"""VLABench, VLM track: plan a long-horizon robot task as a sequence of skill calls.

Benchmark: VLABench (github.com/OpenMOSS/VLABench, MIT licence). Episodes: the Hugging Face
dataset `VLABench/vlm_evaluation_v1.0`, revision b432f0c69ec0731935ff49a916d460936a000624, in
`data/vlabench_vlm/`. The track needs no simulator. The model sees two renders of one scene and an
instruction, and it writes a JSON list of skill calls. The score compares the list with the
reference list.

The prompt is the official zero-shot English prompt of `evaluation/model/vlm/base.py`
(`get_ti_list`) with the system text `configs/prompt/eval_vlm_en.txt`, read from the clone in
`data/VLABench/`. Like the official Qwen wrapper (`qwen_vl.py`), the model gets the plain image
and then the image with the numbered tags, in one user message. Differences from the official run:

- The images pass through `limit_pixels`. The renders are 960 x 960, so this changes nothing today.
- `max_new_tokens` is 1024. The official Qwen wrapper uses 128, which cuts a long plan short.
- The text parts stay separate, as in the official code. The chat template of each family joins them.

The parser and the metrics are ports of the official code (the source of each part is named).
The metrics are fractions from 0 to 1. The official code reports percent.

Ours: the item format, the practice and held-out split, the `parsed` flag, `total_score_intended`,
and the rule for `correct`. `correct` is strict: the reply parses, `exact_match_score` is 1, and the
reply holds the same skill calls as the reference, as a multiset of (name, entity, container),
so a reply with an extra or a missing call is not correct.
"""

import json
from collections import Counter
from pathlib import Path

from PIL import Image

from wsp.evaluate import limit_pixels

GENERATE = {"thinking": False, "max_new_tokens": 1024}

DATA_ROOT = Path("data/vlabench_vlm")
PROMPT_FILE = Path("data/VLABench/VLABench/configs/prompt/eval_vlm_en.txt")
DIMENSIONS = ("M&T", "Semantic", "CommenSence", "Spatial", "PhysicsLaw", "Complex")

# From configs/evaluation/seq_independent_task.json: the subtasks of these tasks have no order.
SEQ_INDEPENDENT_TASKS = ("cook_dishes", "texas_holdem")

# From evaluation/utils.py: SUBTASK_PATTERN. The order matters, because the first match wins.
SUBTASK_PATTERNS = (
    ("pick", "place"),
    ("pick", "insert"),
    ("pick", "pour", "place"),
    ("pick", "pour"),
    ("pick", "pull"),
    ("pick", "lift"),
    ("pick", "push", "pull"),
    ("pick", "push", "place"),
    ("pick", "push"),
    ("pick", "open_door"),
    ("press",),
)

# From evaluation/utils.py: get_final_score. The official total sums only the first two terms.
WEIGHTS = {"skill_match_score": 0.4, "entity_match_score": 0.4, "skill_with_entity_match_score": 0.1, "exact_match_score": 0.1}
METRICS = (*WEIGHTS, "total_score", "total_score_intended")

_PROMPT: list[str] = []


def load(dimension: str, limit: int | None = None, root: Path = DATA_ROOT) -> list[dict]:
    """The examples of one dimension folder, sorted by task name and then by example number."""
    folder = root / dimension
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} does not exist. Download VLABench/vlm_evaluation_v1.0 to {root} first. Folders: {DIMENSIONS}.")
    items = []
    for task in sorted(path.name for path in folder.iterdir() if path.is_dir()):
        examples = sorted((path for path in (folder / task).iterdir() if path.name.startswith("example")), key=lambda path: int(path.name[7:]))
        for example in examples:
            reference = json.loads((example / "output/operation_sequence.json").read_text())["skill_sequence"]
            items.append({
                "id": f"{dimension}/{task}/{example.name}",
                "dimension": dimension,
                "task": task,
                "example": int(example.name[7:]),
                # The official evaluator passes the raw file text, so this keeps it unstripped.
                "instruction": (example / "input/instruction.txt").read_text(encoding="utf-8"),
                "image": str(example / "input/input.png"),
                "mask": str(example / "input/input_mask.png"),
                "reference": reference,
                "dependency": "Seq-independent" if task in SEQ_INDEPENDENT_TASKS else "Sequential",
            })
    return items[:limit]


def split(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Practice items have an even example number. Held-out items have an odd example number."""
    return [item for item in items if item["example"] % 2 == 0], [item for item in items if item["example"] % 2 == 1]


def prompt() -> str:
    """The official system text. The file is read one time."""
    if not _PROMPT:
        if not PROMPT_FILE.is_file():
            raise FileNotFoundError(f"{PROMPT_FILE} does not exist. Clone github.com/OpenMOSS/VLABench to data/VLABench first.")
        _PROMPT.append(PROMPT_FILE.read_text(encoding="utf-8"))
    return _PROMPT[0]


def conversation(item: dict) -> list[dict]:
    """The zero-shot English prompt of evaluation/model/vlm/base.py: get_ti_list."""
    content = [
        {"type": "text", "text": prompt()},
        {"type": "text", "text": "Input picture"},
        {"type": "image", "image": limit_pixels(Image.open(item["image"]).convert("RGB"))},
        {"type": "text", "text": "Input picture with numbered tags"},
        {"type": "image", "image": limit_pixels(Image.open(item["mask"]).convert("RGB"))},
        {"type": "text", "text": "Language instruction:"},
        {"type": "text", "text": item["instruction"]},
        {"type": "text", "text": "Please give the output skill sequence"},
    ]
    return [{"role": "user", "content": content}]


def parse(text: str) -> list[dict] | None:
    """The skill list in the first ```json block, or None.

    The extraction is the one of evaluation/model/vlm/qwen_vl.py. The structure check is ours: the
    official scorer gives zero to a list that it cannot read, and the check names that case.
    """
    try:
        sequence = json.loads(text.split("```json")[1].split("```")[0])
    except (IndexError, json.JSONDecodeError):
        return None
    if not isinstance(sequence, list):
        return None
    for skill in sequence:
        if not (isinstance(skill, dict) and isinstance(skill.get("name"), str) and isinstance(skill.get("params"), dict)):
            return None
        # A list or a dict as a target cannot go into a Counter, so the official scorer fails on it.
        if any(isinstance(skill["params"].get(key), (list, dict)) for key in ("target_entity_name", "target_container_name")):
            return None
    return sequence


def score(item: dict, text: str) -> dict:
    """The official metrics of evaluation/utils.py: get_final_score, as fractions."""
    sequence = parse(text)
    reference = item["reference"]
    if sequence is None or not _subtasks(reference):
        # The official evaluator gives zero for a format error. It also gives zero when the reference
        # has no subtask, because exact_match_percentage then divides by zero inside a try block.
        return {"parsed": sequence is not None, "correct": False, "steps": len(sequence or []), **dict.fromkeys(METRICS, 0.0)}
    metrics = {
        **_skill_and_entity_scores(reference, sequence),
        "skill_with_entity_match_score": _overlap([_call(skill) for skill in reference], [_call(skill) for skill in sequence]),
        "exact_match_score": _exact_match(reference, sequence, item["dependency"]),
    }
    metrics["total_score"] = WEIGHTS["skill_match_score"] * metrics["skill_match_score"] + WEIGHTS["entity_match_score"] * metrics["entity_match_score"]
    metrics["total_score_intended"] = sum(weight * metrics[key] for key, weight in WEIGHTS.items())
    same_calls = Counter(map(_call, reference)) == Counter(map(_call, sequence))
    return {"parsed": True, "correct": same_calls and metrics["exact_match_score"] == 1.0, "steps": len(sequence), **metrics}


def _call(skill: dict) -> tuple:
    return skill["name"], skill["params"].get("target_entity_name"), skill["params"].get("target_container_name")


def _overlap(reference: list, given: list) -> float:
    """The share of the reference that the reply covers, counted with repeats. Extra items cost nothing."""
    return sum((Counter(reference) & Counter(given)).values()) / len(reference) if reference else 0.0


def _skill_and_entity_scores(reference: list[dict], sequence: list[dict]) -> dict:
    """From evaluation/utils.py: calculate_skill_and_entity_scores. Order does not count."""

    def entities(skills: list[dict]) -> list[tuple]:
        found = []
        for skill in skills:
            for role, key in (("target_entity", "target_entity_name"), ("target_container", "target_container_name")):
                if skill["params"].get(key) is not None:
                    found.append((role, skill["params"][key]))
        return found

    return {
        "skill_match_score": _overlap([skill["name"] for skill in reference], [skill["name"] for skill in sequence]),
        "entity_match_score": _overlap(entities(reference), entities(sequence)),
    }


def _subtasks(sequence: list[dict]) -> list[tuple]:
    """From evaluation/utils.py: find_subtasks and the node attributes of build_graph.

    Each subtask is (pattern, last entity, last container). A skill outside every pattern drops out.
    """
    names = [skill["name"] for skill in sequence]
    found, i = [], 0
    while i < len(names):
        pattern = next((pattern for pattern in SUBTASK_PATTERNS if tuple(names[i : i + len(pattern)]) == pattern), None)
        if pattern is None:
            i += 1
            continue
        part = sequence[i : i + len(pattern)]
        entities = [skill["params"].get("target_entity_name") for skill in part if skill["params"].get("target_entity_name") is not None]
        containers = [skill["params"].get("target_container_name") for skill in part if skill["params"].get("target_container_name") is not None]
        found.append(("-".join(pattern), entities[-1] if entities else None, containers[-1] if containers else None))
        i += len(pattern)
    return found


def _exact_match(reference: list[dict], sequence: list[dict], dependency: str) -> float:
    """From evaluation/utils.py: exact_match_percentage on the graphs of build_graph.

    The official code compares the nodes layer by layer. A Sequential graph is a chain, so layer k
    holds subtask k, and a match needs subtask k and its predecessor to be equal. A Seq-independent
    graph is a star under START, so all subtasks are in one layer and the order does not count.
    """
    expected, given = _subtasks(reference), _subtasks(sequence)
    if dependency == "Seq-independent":
        return _overlap(expected, given)
    start = ("START", None, None)
    matches = sum(
        1
        for k, node in enumerate(expected)
        if k < len(given) and given[k] == node and ([start, *given][k] == [start, *expected][k])
    )
    return matches / len(expected)

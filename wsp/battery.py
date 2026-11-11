"""The generalization battery: fixed benchmark subsets that run before and after an edit.

The battery measures abilities outside the target of an edit. `run` returns one result per set.
`compare` reports each change with a paired test and marks a major drop.
"""

from wsp import evaluate
from wsp.bench import boxed_math, chartqa, countbench, mmlu_pro, refcoco
from wsp.stats import paired_flips

# A drop is major when it is this many points or more and the paired p-value is below 0.05.
MAJOR_DROP_POINTS = 5.0

# name: (items loader, conversation, score, generation options, batch size)
SETS = {
    "mmlu_pro": (lambda: mmlu_pro.load(200), mmlu_pro.conversation, mmlu_pro.score, mmlu_pro.GENERATE, 32),
    "gsm8k_thinking": (
        lambda: boxed_math.load_gsm8k_test(100),
        boxed_math.conversation,
        boxed_math.score,
        {**boxed_math.BLOG_SAMPLING, "max_new_tokens": 4096},
        25,
    ),
    "countbench": (lambda: countbench.load(200), countbench.conversation, countbench.score, countbench.GENERATE, 8),
    "chartqa": (lambda: chartqa.load(200), chartqa.conversation, chartqa.score, chartqa.GENERATE, 8),
    "refcoco": (lambda: refcoco.load(200), refcoco.conversation, refcoco.score, refcoco.GENERATE, 8),
}

_ITEMS: dict[str, list[dict]] = {}


def items(name: str) -> list[dict]:
    """The fixed items of one set. The first call loads them, and later calls reuse them."""
    if name not in _ITEMS:
        _ITEMS[name] = SETS[name][0]()
    return _ITEMS[name]


def run(names: list[str] | None = None) -> dict:
    """Run the named sets, or all sets, on the model that the kernel holds now."""
    results = {}
    for name in names or list(SETS):
        _, conversation, score, options, batch_size = SETS[name]
        rows = evaluate.run(items(name), conversation, score, batch_size=batch_size, **options)
        results[name] = {
            "summary": evaluate.summarize(rows)["all"],
            # The verdict needs only the outcome of each item, not the reply text.
            "correct": {row["id"]: bool(row["correct"]) for row in rows},
        }
    return results


def neutral_text(count: int = 64, tokens: int = 256) -> list[list[int]]:
    """Token sequences from WikiText-2 (`Salesforce/wikitext`), for `tools.deviation`."""
    from datasets import load_dataset

    from wsp import tools

    rows = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    # A short row is a heading or an empty line. Keep paragraphs only.
    paragraphs = [row["text"].strip() for row in rows if len(row["text"]) > 600][:count]
    encoded = tools.PROCESSOR.tokenizer(paragraphs, add_special_tokens=False)["input_ids"]
    return [sequence[:tokens] for sequence in encoded]


def compare(before: dict, after: dict) -> dict:
    """The change of each set from `before` to `after`, with the items that the edit fixed and broke."""
    report = {}
    for name in before:
        if name not in after:
            continue
        ids = list(before[name]["correct"])
        flips = paired_flips([before[name]["correct"][i] for i in ids], [after[name]["correct"][i] for i in ids])
        change = after[name]["summary"]["correct_pct"] - before[name]["summary"]["correct_pct"]
        report[name] = {
            "before_pct": before[name]["summary"]["correct_pct"],
            "after_pct": after[name]["summary"]["correct_pct"],
            "change_points": round(change, 1),
            **flips,
            "major_drop": change <= -MAJOR_DROP_POINTS and flips["p_value"] < 0.05,
        }
    return report

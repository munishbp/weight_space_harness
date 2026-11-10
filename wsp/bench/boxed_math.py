"""Math problems with one final answer in a `\\boxed{}`. Used for GSM8K and for the blog's practice set.

The prompt suffix and the sampling follow the evaluation of the weight-space programming blog
(`exp8_math` in https://github.com/RikiyaT/lm_understand-logs).
"""

import json
import re
from pathlib import Path

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."

# The sampling of the blog's evaluation: thinking on, with these options for `tools.generate`.
BLOG_SAMPLING = {"thinking": True, "do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}


def load_jsonl(path: Path) -> list[dict]:
    """Records with `id`, `problem`, and `answer`, one JSON object per line."""
    # `str.splitlines` also splits on Unicode separators inside a JSON string, so split on "\n" only.
    return [json.loads(line) for line in Path(path).read_text().split("\n") if line.strip()]


def conversation(item: dict) -> list[dict]:
    return [{"role": "user", "content": [{"type": "text", "text": item["problem"] + SUFFIX}]}]


def _number(text: str) -> float | None:
    cleaned = text.replace(",", "").replace("$", "").replace("%", "")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    return float(match.group()) if match else None


def score(item: dict, text: str) -> dict:
    """The last boxed value must equal the answer as a number. This fits numeric answers only."""
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    value = _number(boxed[-1]) if boxed else None
    return {"correct": value is not None and value == _number(item["answer"]), "answer_given": boxed[-1] if boxed else None}

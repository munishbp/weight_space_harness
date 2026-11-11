"""ChartQA: answer a question about a chart image.

Dataset: `lmms-lab/ChartQA` on Hugging Face.
The score is the relaxed accuracy of the ChartQA paper: a numeric answer can differ by 5 percent.
"""

import random
import re

from datasets import load_dataset

from wsp.evaluate import limit_pixels

GENERATE = {"thinking": False, "max_new_tokens": 32}


def load(count: int, seed: int = 0) -> list[dict]:
    """A fixed random subset of the test split."""
    rows = load_dataset("lmms-lab/ChartQA", split="test")
    picks = sorted(random.Random(seed).sample(range(len(rows)), count))
    return [{"id": f"chartqa_{i}", **rows[i]} for i in picks]


def conversation(item: dict) -> list[dict]:
    text = f"{item['question']}\nAnswer the question with a single word or number."
    return [{"role": "user", "content": [{"type": "image", "image": limit_pixels(item["image"].convert("RGB"))}, {"type": "text", "text": text}]}]


def _number(text: str) -> float | None:
    try:
        return float(text.replace(",", "").replace("%", "").replace("$", "").strip())
    except ValueError:
        return None


def score(item: dict, text: str) -> dict:
    # Some families (GLM) mark the final answer with box tokens inside a sentence.
    boxed = re.search(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", text, flags=re.DOTALL)
    given, target = (boxed.group(1) if boxed else text).strip().rstrip("."), item["answer"].strip()
    given_number, target_number = _number(given), _number(target)
    if given_number is not None and target_number is not None:
        # A target of zero has no relative error, so it needs an exact match.
        close = given_number == target_number if target_number == 0 else abs(given_number - target_number) <= 0.05 * abs(target_number)
        return {"correct": close}
    return {"correct": given.lower() == target.lower()}

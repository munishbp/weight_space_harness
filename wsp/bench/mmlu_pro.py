"""MMLU-Pro: multiple-choice questions on many subjects, with up to ten options.

Dataset: `TIGER-Lab/MMLU-Pro` on Hugging Face (MIT licence).
"""

import random
import re

from datasets import load_dataset

LETTERS = "ABCDEFGHIJ"
GENERATE = {"thinking": False, "max_new_tokens": 8}


def load(count: int, seed: int = 0) -> list[dict]:
    """A fixed random subset of the test split."""
    rows = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    picks = sorted(random.Random(seed).sample(range(len(rows)), count))
    return [{"id": f"mmlu_pro_{rows[i]['question_id']}", **rows[i]} for i in picks]


def conversation(item: dict) -> list[dict]:
    options = "\n".join(f"{LETTERS[i]}. {option}" for i, option in enumerate(item["options"]))
    text = f"{item['question']}\n\n{options}\n\nAnswer with the letter of the correct option only."
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def score(item: dict, text: str) -> dict:
    """The first option letter in the reply must equal the answer."""
    match = re.search(r"\b([A-J])\b", text)
    return {"correct": bool(match) and match.group(1) == item["answer"]}

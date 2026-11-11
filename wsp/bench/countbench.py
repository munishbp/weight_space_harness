"""CountBench: count the objects of one kind in a photo.

Dataset: `vikhyatk/CountBenchQA` on Hugging Face, which adds a question to each CountBench image.
"""

import random
import re

from datasets import load_dataset

from wsp.evaluate import limit_pixels

GENERATE = {"thinking": False, "max_new_tokens": 16}


def load(count: int, seed: int = 0) -> list[dict]:
    """A fixed random subset of the test split."""
    rows = load_dataset("vikhyatk/CountBenchQA", split="test")
    picks = sorted(random.Random(seed).sample(range(len(rows)), count))
    return [{"id": f"countbench_{i}", **rows[i]} for i in picks]


def conversation(item: dict) -> list[dict]:
    text = f"{item['question']} Answer with a number only."
    return [{"role": "user", "content": [{"type": "image", "image": limit_pixels(item["image"].convert("RGB"))}, {"type": "text", "text": text}]}]


def score(item: dict, text: str) -> dict:
    """The first integer in the reply must equal the count."""
    match = re.search(r"\d+", text)
    return {"correct": bool(match) and int(match.group()) == int(item["number"])}

"""RefCOCO: find the object that a phrase describes in a photo, and return its box.

Dataset: `lmms-lab/RefCOCO` on Hugging Face. A box is `[x, y, width, height]` in pixels.
The prompt follows the single-object example of the Qwen3-VL grounding cookbook. The model writes
`{"bbox_2d": [x1, y1, x2, y2]}` on a 0 to 1000 scale. A box that overlaps the target with an
intersection over union of 0.5 or more is correct.
"""

import random
import re

from datasets import load_dataset

from wsp.evaluate import limit_pixels

GENERATE = {"thinking": False, "max_new_tokens": 64}


def load(count: int, seed: int = 0) -> list[dict]:
    """A fixed random subset of the validation split."""
    rows = load_dataset("lmms-lab/RefCOCO", split="val")
    picks = sorted(random.Random(seed).sample(range(len(rows)), count))
    keep = ("image", "bbox", "answer")
    return [{"id": f"refcoco_{rows[i]['question_id']}", **{key: rows[i][key] for key in keep}} for i in picks]


def conversation(item: dict) -> list[dict]:
    phrase = item["answer"][0].strip().rstrip(".")
    text = f"Locate {phrase}, output its bbox coordinates using JSON format."
    return [{"role": "user", "content": [{"type": "image", "image": limit_pixels(item["image"].convert("RGB"))}, {"type": "text", "text": text}]}]


def parse_box(text: str) -> list[float] | None:
    """The first box in the reply, as shares of the image width and height."""
    match = re.search(r'"bbox_2d"\s*:\s*\[([^\]]*)\]', text)
    if not match:
        return None
    try:
        values = [float(value) / 1000 for value in match.group(1).split(",")]
    except ValueError:
        return None
    return values if len(values) == 4 else None


def overlap(a: list[float], b: list[float]) -> float:
    """Intersection over union of two boxes in `[x1, y1, x2, y2]` form."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    shared = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - shared
    return shared / union


def score(item: dict, text: str) -> dict:
    box = parse_box(text)
    if box is None:
        return {"correct": False, "iou": None}
    image_width, image_height = item["image"].size
    x, y, width, height = item["bbox"]
    target = [x / image_width, y / image_height, (x + width) / image_width, (y + height) / image_height]
    value = overlap(box, target)
    return {"correct": value >= 0.5, "iou": round(value, 3)}

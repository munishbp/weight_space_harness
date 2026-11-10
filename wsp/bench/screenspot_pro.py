"""ScreenSpot Pro: GUI grounding on professional software. One screenshot, one instruction, one click.

Dataset: `likaixin/ScreenSpot-Pro` on Hugging Face (MIT licence).
The prompt and the scoring rule follow the official Qwen3.5 wrapper, `models/qwen3_5.py` in
https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding.
"""

import json
import math
import re
from pathlib import Path

from huggingface_hub import snapshot_download
from PIL import Image

# The system prompt of the official wrapper, character for character.
SYSTEM_PROMPT = """You are a helpful assistant. The user will give you an instruction, and you MUST left click on the corresponding UI element via tool call. If you are not sure about where to click, guess a most likely one.\n\n# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.\n* The screen's resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. \n* You can only use the left_click action to interact with the computer.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `left_click`: Click the left mouse button with coordinate (x, y).", "enum": ["left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=left_click`.", "type": "array"}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

# The official wrapper caps a screenshot at this many pixels before the model sees it.
MAX_PIXELS = 8_294_400
# One visual token of the Qwen3.5 vision encoder covers 32 by 32 pixels.
PIXEL_FACTOR = 32


def load() -> tuple[list[dict], Path]:
    """All records in a fixed order, and the directory that holds the images."""
    root = Path(snapshot_download("likaixin/ScreenSpot-Pro", repo_type="dataset"))
    items = []
    for path in sorted((root / "annotations").glob("*.json")):
        items.extend(json.loads(path.read_text()))
    return sorted(items, key=lambda item: item["id"]), root / "images"


def open_image(item: dict, images: Path, max_pixels: int = MAX_PIXELS) -> Image.Image:
    """The screenshot, scaled down when it is larger than `max_pixels`."""
    image = Image.open(images / item["img_filename"]).convert("RGB")
    scale = min(1.0, math.sqrt(max_pixels / (image.width * image.height)))
    width = max(PIXEL_FACTOR, round(image.width * scale / PIXEL_FACTOR) * PIXEL_FACTOR)
    height = max(PIXEL_FACTOR, round(image.height * scale / PIXEL_FACTOR) * PIXEL_FACTOR)
    return image.resize((width, height))


def conversation(item: dict, images: Path, max_pixels: int = MAX_PIXELS) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": open_image(item, images, max_pixels)},
                {"type": "text", "text": item["instruction"]},
            ],
        },
    ]


def parse_point(text: str) -> tuple[float, float] | None:
    """The click point of the last tool call, as a share of the image width and height.

    The model writes coordinates on a 0 to 1000 scale. A box of four values becomes its centre.
    """
    matches = re.findall(r'"coordinate"\s*:\s*\[([^\]]*)\]', text)
    if not matches:
        return None
    try:
        values = [float(value) for value in matches[-1].split(",")]
    except ValueError:
        return None
    if len(values) == 2:
        return values[0] / 1000, values[1] / 1000
    if len(values) == 4:
        return (values[0] + values[2]) / 2000, (values[1] + values[3]) / 2000
    return None


def score(item: dict, text: str) -> dict:
    """A click inside the target box is correct, edges included. A reply with no point is wrong."""
    point = parse_point(text)
    if point is None:
        return {"correct": False, "point": None}
    width, height = item["img_size"]
    x1, y1, x2, y2 = item["bbox"]
    inside = x1 / width <= point[0] <= x2 / width and y1 / height <= point[1] <= y2 / height
    return {"correct": inside, "point": list(point)}

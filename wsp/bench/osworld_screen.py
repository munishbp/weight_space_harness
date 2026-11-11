"""OSWorld through screenshots: desktop tasks in an Ubuntu virtual machine, solved with mouse and keyboard.

Benchmark: OSWorld (Xie et al., 2024; Apache-2.0). The environment and its evaluators come from the
`desktop-env` package. That package runs in its own process (`wsp/bench/osworld_server.py`), and this
module talks to it through a Unix socket. The task files come from the OSWorld repository
(`evaluation_examples/`), as data.

The interface is ours. The model sees one screenshot and the instruction. It writes one action as JSON:

    {"action": "click", "position": [x, y]}            also "double_click" and "right_click"
    {"action": "drag", "from": [x, y], "to": [x, y]}
    {"action": "scroll", "position": [x, y], "amount": -3}   a negative amount scrolls down
    {"action": "type", "text": "..."}
    {"action": "key", "keys": ["ctrl", "s"]}           one key, or keys that are held together
    {"action": "wait"}
    {"action": "done"}                                 the task is complete
    {"action": "fail"}                                 the task is not possible

x and y run from 0 to 1000 across the screenshot. This module turns a valid action into one line of
`pyautogui` code. The model never writes code that runs in the machine.
"""

import base64
import io
import json
import re
import socket
import time
from pathlib import Path

from PIL import Image

SYSTEM_PROMPT = """You operate an Ubuntu desktop with a mouse and a keyboard. You see a screenshot of the screen.

At each turn you write exactly one action as a JSON object, and nothing else:
{"action": "click", "position": [x, y]}  clicks at that point. x and y run from 0 to 1000: x from the left edge to the right edge of the screenshot, y from the top edge to the bottom edge.
{"action": "double_click", "position": [x, y]}  and  {"action": "right_click", "position": [x, y]}  work in the same way.
{"action": "drag", "from": [x, y], "to": [x, y]}  drags with the left button.
{"action": "scroll", "position": [x, y], "amount": -3}  scrolls at that point. A negative amount scrolls down.
{"action": "type", "text": "..."}  types the text at the cursor. Click a field first to give it the focus.
{"action": "key", "keys": ["ctrl", "s"]}  presses one key, or keys that are held together. Examples of keys: enter, tab, esc, backspace, delete, space, up, down, left, right, ctrl, alt, shift, a, 1, f5.
{"action": "wait"}  waits for the screen to change.
{"action": "done"}  ends the task when it is complete.
{"action": "fail"}  ends the task when it is not possible."""

MAX_STEPS = 15
# The reference runner of OSWorld waits 20 seconds before it evaluates, so that the desktop settles.
SETTLE_SECONDS = 20
SOCKET_PATH = "/tmp/wsp-osworld.sock"
TASK_ROOT = Path("data/OSWorld/evaluation_examples")

NAMED_KEYS = {
    "enter", "tab", "esc", "backspace", "delete", "space", "up", "down", "left", "right", "home", "end",
    "pageup", "pagedown", "ctrl", "alt", "shift", "win", "insert", "capslock", "printscreen",
    *(f"f{number}" for number in range(1, 13)),
}
POINT_ACTIONS = {"click": "click", "double_click": "doubleClick", "right_click": "rightClick"}
END_ACTIONS = {"wait": "WAIT", "done": "DONE", "fail": "FAIL"}


class BridgeError(RuntimeError):
    """The OSWorld server did not answer, or it reported a failure."""


class Client:
    """One connection to `osworld_server.py`."""

    def __init__(self, path: str = SOCKET_PATH) -> None:
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.socket.connect(path)
        except OSError as error:
            raise BridgeError(
                f"No OSWorld server answers on {path} ({error}). No task ran. "
                "Start it with: .venv-osworld/bin/python wsp/bench/osworld_server.py"
            ) from error
        self.stream = self.socket.makefile("rwb")

    def request(self, op: str, **fields) -> dict:
        self.stream.write(json.dumps({"op": op, **fields}).encode() + b"\n")
        self.stream.flush()
        line = self.stream.readline()
        if not line:
            raise BridgeError(f"The OSWorld server closed the connection during `{op}`. Read its log, then start it again.")
        reply = json.loads(line)
        if "error" in reply:
            raise BridgeError(f"The OSWorld server failed during `{op}`: {reply['error']}")
        return reply

    def close(self) -> None:
        self.stream.close()
        self.socket.close()


def tasks(domain: str, root: Path = TASK_ROOT) -> list[dict]:
    """The task files of one domain (for example "chrome" or "os"), in the order of `test_all.json`."""
    index = root / "test_all.json"
    if not index.exists():
        raise FileNotFoundError(f"{index} does not exist. Put the OSWorld repository at data/OSWorld first.")
    ids = json.loads(index.read_text()).get(domain)
    if ids is None:
        raise KeyError(f"OSWorld has no domain {domain!r}. The domains are in {index}.")
    return [json.loads((root / "examples" / domain / f"{task_id}.json").read_text()) | {"domain": domain} for task_id in ids]


def _is_point(value) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1000 for v in value)
    )


def _pixels(point: list, width: int, height: int) -> str:
    return f"{round(point[0] / 1000 * (width - 1))}, {round(point[1] / 1000 * (height - 1))}"


def to_code(action: dict | None, width: int, height: int) -> str | None:
    """One line of `pyautogui` code for a valid action, a special word for wait/done/fail, or None for an invalid action."""
    if not isinstance(action, dict):
        return None
    name = action.get("action")
    if name in END_ACTIONS:
        return END_ACTIONS[name]
    if name in POINT_ACTIONS and _is_point(action.get("position")):
        return f"pyautogui.{POINT_ACTIONS[name]}({_pixels(action['position'], width, height)})"
    if name == "drag" and _is_point(action.get("from")) and _is_point(action.get("to")):
        return (
            f"pyautogui.moveTo({_pixels(action['from'], width, height)}); "
            f"pyautogui.dragTo({_pixels(action['to'], width, height)}, duration=0.5, button='left')"
        )
    amount = action.get("amount")
    if name == "scroll" and _is_point(action.get("position")) and isinstance(amount, int) and not isinstance(amount, bool) and -50 <= amount <= 50:
        return f"pyautogui.scroll({amount}, {_pixels(action['position'], width, height)})"
    text = action.get("text")
    # `pyautogui.write` presses keyboard keys, so it types printable ASCII only.
    if name == "type" and isinstance(text, str) and text and len(text) <= 500 and all(32 <= ord(c) < 127 or c == "\n" for c in text):
        return f"pyautogui.write({text!r}, interval=0.02)"
    keys = action.get("keys")
    if name == "key" and isinstance(keys, list) and 1 <= len(keys) <= 4:
        keys = [key.lower() if isinstance(key, str) else None for key in keys]
        if all(key in NAMED_KEYS or (key is not None and len(key) == 1 and 33 <= ord(key) < 127) for key in keys):
            return f"pyautogui.hotkey({', '.join(repr(key) for key in keys)})"
    return None


def parse_action(text: str) -> dict | None:
    """The first JSON object in the reply, or None when the reply holds no valid JSON object."""
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def conversation(instruction: str, screenshot: Image.Image, history: list[dict]) -> list[dict]:
    """The prompt for the next action: the screenshot, the task, and the actions so far."""
    done = "\n".join(json.dumps(step) for step in history) or "none"
    text = f"Task: {instruction}\nActions so far:\n{done}\nWrite the next action."
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": screenshot}, {"type": "text", "text": text}]},
    ]


def _image(encoded: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def run_episodes(task_list: list[dict], generate, client: Client | None = None, max_steps: int = MAX_STEPS) -> list[dict]:
    """Run each task with a model, one after the other. `generate` maps conversations to replies, as `tools.generate` does.

    One virtual machine runs one task at a time. A result holds `score` (0 to 1 from the OSWorld
    evaluator) and `correct` (a score of 1). When the bridge fails in a task, the result holds `error`
    and no score, so a failure of the machine does not count as a failure of the model.
    """
    client = client or Client()
    results = []
    for task in task_list:
        started = time.time()
        result = {"id": task["id"], "domain": task.get("domain"), "instruction": task["instruction"], "actions": [], "replies": [], "invalid": 0}
        try:
            # A reset goes back to the snapshot, so a second try is safe.
            try:
                state = client.request("reset", task=task)
            except BridgeError:
                state = client.request("reset", task=task)
            for _ in range(max_steps):
                screenshot = _image(state["screenshot"])
                reply = generate([conversation(state["instruction"], screenshot, result["actions"])])[0]
                result["replies"].append(reply["text"])
                action = parse_action(reply["text"])
                code = to_code(action, *screenshot.size)
                if code is None:
                    # An invalid action counts as a step, changes nothing, and stays in the history.
                    result["invalid"] += 1
                    result["actions"].append({"action": "invalid"})
                    continue
                result["actions"].append(action)
                state = client.request("step", code=code) | {"instruction": state["instruction"]}
                if state["done"]:
                    break
            time.sleep(SETTLE_SECONDS)
            result["score"] = client.request("evaluate")["score"]
            result["correct"] = result["score"] >= 1.0
        except BridgeError as error:
            result["error"] = str(error)
            result["correct"] = False
        result["seconds"] = round(time.time() - started)
        results.append(result)
    return results

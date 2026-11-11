"""MiniWoB++ through screenshots: short web tasks that a model solves with clicks and keys.

Benchmark: MiniWoB++ (`miniwob` on PyPI, MIT licence). Selenium drives a headless Chrome, and
Selenium downloads the ChromeDriver that matches the installed Chrome.

The interface is ours. The model sees one screenshot and the instruction. It writes one action as JSON:

    {"action": "click", "position": [x, y]}   click at a point of the screenshot
    {"action": "type", "text": "..."}         type text into the element that has the focus
    {"action": "key", "key": "Enter"}         press one key
    {"action": "done"}                        stop the episode

x and y run from 0 to 1000 across the screenshot, as in the grounding format of Qwen3.5.

The page ends an episode after 10 seconds and lowers the reward with time. A local model is slower
than that, so this module removes the limit. An episode succeeds when the page gives a positive reward.
"""

import json
import re

import gymnasium
import miniwob
from miniwob.action import ActionTypes
from miniwob.constants import TYPING_MAX_LENGTH
from PIL import Image

gymnasium.register_envs(miniwob)

SYSTEM_PROMPT = """You operate a web page with a mouse and a keyboard. You see a screenshot of the page. The yellow bar at the top shows the task.

At each turn you write exactly one action as a JSON object, and nothing else:
{"action": "click", "position": [x, y]}  clicks at that point. x and y run from 0 to 1000: x from the left edge to the right edge of the screenshot, y from the top edge to the bottom edge.
{"action": "type", "text": "..."}  types the text into the field that has the focus. Click a field first to give it the focus.
{"action": "key", "key": "Enter"}  presses one key: Enter, Tab, Backspace, Delete, Space, ArrowUp, ArrowDown, ArrowLeft, or ArrowRight.
{"action": "done"}  ends the task."""

KEYS = ("Enter", "Tab", "Backspace", "Delete", "Space", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight")
MAX_ACTIONS = 10
# The page is 160 by 210 pixels. The model sees it four times larger.
ZOOM = 4
NO_LIMIT_MS = 100_000_000

# One browser per task type. A browser start takes about 3 seconds, so episodes of a task share it.
_ENVS: dict[str, gymnasium.Env] = {}


def environment(task: str) -> gymnasium.Env:
    """The browser of one task type, with the episode time limit removed."""
    if task not in _ENVS:
        env = gymnasium.make(f"miniwob/{task}-v1")
        # The page exists only after the first reset. The next reset starts its timer with the new limit.
        env.reset(seed=0)
        env.unwrapped.instance.driver.execute_script(f"core.EPISODE_MAX_TIME = {NO_LIMIT_MS};")
        _ENVS[task] = env
    return _ENVS[task]


def close_all() -> None:
    """Close every browser that this module started."""
    for env in _ENVS.values():
        env.close()
    _ENVS.clear()


class Episode:
    """One run of one task type from one seed. The seed fixes the content of the page."""

    def __init__(self, task: str, seed: int) -> None:
        self.task, self.seed = task, seed
        self.env = environment(task)
        self.obs, _ = self.env.reset(seed=seed)
        self.utterance = self.obs["utterance"]
        self.height, self.width = self.obs["screenshot"].shape[:2]
        self.success = False
        self.ended = False
        self.history: list[dict] = []
        self.invalid = 0

    @property
    def over(self) -> bool:
        return self.ended or len(self.history) >= MAX_ACTIONS or (
            bool(self.history) and self.history[-1].get("action") == "done"
        )

    def frame(self) -> Image.Image:
        image = Image.fromarray(self.obs["screenshot"])
        return image.resize((self.width * ZOOM, self.height * ZOOM), Image.LANCZOS)

    def _step(self, kind: ActionTypes, **fields) -> None:
        action = self.env.unwrapped.create_action(kind, **fields)
        self.obs, reward, terminated, truncated, _ = self.env.step(action)
        if terminated or truncated:
            self.ended = True
            self.success = reward > 0

    def apply(self, action: dict | None) -> None:
        """Execute one action. An invalid action counts, changes nothing, and stays in the history."""
        name = action.get("action") if isinstance(action, dict) else None
        position = action.get("position") if isinstance(action, dict) else None
        text = action.get("text") if isinstance(action, dict) else None
        key = action.get("key") if isinstance(action, dict) else None
        if name == "click" and _is_point(position):
            x, y = position[0] / 1000 * self.width, position[1] / 1000 * self.height
            self._step(ActionTypes.CLICK_COORDS, coords=[x, y])
            self.history.append({"action": "click", "position": [int(position[0]), int(position[1])]})
        elif name == "type" and isinstance(text, str) and text and len(text) <= TYPING_MAX_LENGTH and text.isascii():
            self._step(ActionTypes.TYPE_TEXT, text=text)
            self.history.append({"action": "type", "text": text})
        elif name == "key" and key in KEYS:
            allowed = self.env.unwrapped.action_space_config.allowed_keys
            self._step(ActionTypes.PRESS_KEY, key=allowed.index(f"<{key}>"))
            self.history.append({"action": "key", "key": key})
        elif name == "done":
            self.history.append({"action": "done"})
        else:
            self.invalid += 1
            self.history.append({"action": "invalid"})


def _is_point(value) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(v, (int, float)) and 0 <= v <= 1000 for v in value)
    )


def conversation(episode: Episode) -> list[dict]:
    """The prompt for the next action: the screenshot, the task, and the actions so far."""
    done = "\n".join(json.dumps(step) for step in episode.history) or "none"
    text = f"Task: {episode.utterance}\nActions so far:\n{done}\nWrite the next action."
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": episode.frame()}, {"type": "text", "text": text}]},
    ]


def parse_action(text: str) -> dict | None:
    """The first JSON object in the reply, or None when the reply holds no valid JSON object."""
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def run_episodes(specs: list[tuple[str, int]], generate, batch_size: int = 16) -> list[dict]:
    """Run each (task, seed) episode with a model. `generate` maps conversations to replies, as `tools.generate` does.

    One batch holds at most one episode per task type, because a task type has one browser.
    All active episodes of a batch ask for their next action in one forward pass.
    """
    results = []
    waiting = list(specs)
    while waiting:
        batch, tasks = [], set()
        for spec in waiting:
            if spec[0] not in tasks and len(batch) < batch_size:
                batch.append(spec)
                tasks.add(spec[0])
        waiting = [spec for spec in waiting if spec not in batch]
        episodes = [Episode(task, seed) for task, seed in batch]
        replies_of: dict[int, list[str]] = {index: [] for index in range(len(episodes))}
        while True:
            active = [index for index, episode in enumerate(episodes) if not episode.over]
            if not active:
                break
            replies = generate([conversation(episodes[index]) for index in active])
            for index, reply in zip(active, replies):
                replies_of[index].append(reply["text"])
                episodes[index].apply(parse_action(reply["text"]))
        for index, episode in enumerate(episodes):
            results.append(
                {
                    "id": f"{episode.task}/{episode.seed}",
                    "task": episode.task,
                    "seed": episode.seed,
                    "utterance": episode.utterance,
                    "correct": bool(episode.success),
                    "actions": episode.history,
                    "invalid": episode.invalid,
                    "replies": replies_of[index],
                }
            )
    return results

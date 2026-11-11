"""Meta-World through text waypoints: a robot arm that a text model controls one primitive at a time.

Simulator: Meta-World 3 (`metaworld` on PyPI, MIT licence), a Sawyer arm in MuJoCo.

The interface is ours. The model sees one camera image, the task instruction, and the positions of
the gripper, the object, and the goal in millimetres. It writes one primitive as JSON:

    {"action": "move", "position": [x, y, z]}    move the gripper to a position
    {"action": "close"}  {"action": "open"}      close or open the gripper
    {"action": "done"}                           stop the episode

A proportional controller executes the primitive with the 4-D action of Meta-World. The loop
repeats with a new image until the task succeeds, the model stops, or a limit ends the episode.
The prompt holds positions from the simulator state, as EmbodiedBench EB-Manipulation does.
"""

import json
import os
import re

# MuJoCo reads this variable at import. Without it, rendering needs a display.
os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym  # noqa: E402
import metaworld  # noqa: E402, F401  (the import registers the Meta-World environments)
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

# The ten tasks of the standard MT10 set, each with the instruction that the model reads.
INSTRUCTIONS = {
    "reach-v3": "Move the gripper to the goal position.",
    "push-v3": "Push the puck along the table until it is at the goal position.",
    "pick-place-v3": "Pick up the puck and bring it to the goal position.",
    "door-open-v3": "Open the door. Pull the door handle to the goal position.",
    "drawer-open-v3": "Open the drawer. Pull the drawer handle to the goal position.",
    "drawer-close-v3": "Close the drawer. Push the drawer handle to the goal position.",
    "button-press-topdown-v3": "Press the button down from above until it is at the goal position.",
    "peg-insert-side-v3": "Pick up the peg and insert it into the hole at the goal position.",
    "window-open-v3": "Open the window. Slide the window handle to the goal position.",
    "window-close-v3": "Close the window. Slide the window handle to the goal position.",
}

SYSTEM_PROMPT = """You control a robot arm with a two-finger gripper above a table. You see the scene from a camera.

Positions are [x, y, z] in millimetres. x grows to the right, y grows away from the robot base, and z grows upward. The table surface is at z = 0.

At each turn you write exactly one primitive as a JSON object, and nothing else:
{"action": "move", "position": [x, y, z]}  moves the gripper in a straight line to that position.
{"action": "close"}  closes the gripper to grasp the object between the fingers.
{"action": "open"}  opens the gripper.
{"action": "done"}  ends the task.

The gripper grasps an object when the gripper is at the position of the object and then closes. To push an object, place the gripper beside the object and move through it toward the goal."""

MAX_PRIMITIVES = 12
MOVE_STEPS = 80
GRIP_STEPS = 12
ARRIVE_M = 0.005


class Episode:
    """One run of one task from one seed. The seed fixes the start positions of the object and the goal."""

    def __init__(self, task: str, seed: int) -> None:
        self.task, self.seed = task, seed
        self.env = gym.make("Meta-World/MT1", env_name=task, render_mode="rgb_array", camera_name="corner2", seed=seed)
        self.obs, _ = self.env.reset(seed=seed)
        self.grip = -1.0  # -1 holds the gripper open, +1 holds it closed
        self.success = False
        self.out_of_time = False
        self.history: list[dict] = []
        self.invalid = 0
        # The observation from `reset` can hold a stale object position (window-close-v3 does).
        # One step with no motion gives the true start positions.
        self._step(np.zeros(3))

    @property
    def over(self) -> bool:
        return self.success or self.out_of_time or len(self.history) >= MAX_PRIMITIVES or (
            bool(self.history) and self.history[-1].get("action") == "done"
        )

    def positions(self) -> dict[str, list[int]]:
        """The gripper, the object, and the goal in whole millimetres."""
        mm = lambda vector: [int(round(1000 * float(value))) for value in vector]
        return {"gripper": mm(self.obs[0:3]), "object": mm(self.obs[4:7]), "goal": mm(self.obs[-3:])}

    def frame(self) -> Image.Image:
        # The corner camera of Meta-World renders upside down.
        return Image.fromarray(np.flipud(self.env.render()).copy())

    def _step(self, motion: np.ndarray) -> None:
        self.obs, _, _, truncated, info = self.env.step(np.append(motion, self.grip).astype(np.float32))
        self.success = self.success or info["success"] > 0
        self.out_of_time = self.out_of_time or bool(truncated)

    def apply(self, primitive: dict | None) -> None:
        """Execute one primitive. An invalid primitive counts, changes nothing, and stays in the history."""
        action = primitive.get("action") if isinstance(primitive, dict) else None
        position = primitive.get("position") if isinstance(primitive, dict) else None
        if action == "move" and _is_position(position):
            target = np.array(position, dtype=float) / 1000
            for _ in range(MOVE_STEPS):
                delta = target - self.obs[0:3]
                if np.linalg.norm(delta) < ARRIVE_M or self.success or self.out_of_time:
                    break
                # One unit of action moves the gripper about 10 mm, so the gain slows the arm near the target.
                self._step(np.clip(delta * 25, -1, 1))
            self.history.append({"action": "move", "position": [int(value) for value in position]})
        elif action in ("close", "open"):
            self.grip = 1.0 if action == "close" else -1.0
            for _ in range(GRIP_STEPS):
                if self.success or self.out_of_time:
                    break
                self._step(np.zeros(3))
            self.history.append({"action": action})
        elif action == "done":
            self.history.append({"action": "done"})
        else:
            self.invalid += 1
            self.history.append({"action": "invalid"})

    def close(self) -> None:
        self.env.close()


def _is_position(value) -> bool:
    return isinstance(value, list) and len(value) == 3 and all(isinstance(v, (int, float)) for v in value)


def conversation(episode: Episode) -> list[dict]:
    """The prompt for the next primitive: the image, the instruction, the positions, and the primitives so far."""
    positions = episode.positions()
    done = "\n".join(json.dumps(step) for step in episode.history) or "none"
    text = (
        f"Task: {INSTRUCTIONS[episode.task]}\n"
        f"Gripper position: {positions['gripper']}\n"
        f"Object position: {positions['object']}\n"
        f"Goal position: {positions['goal']}\n"
        f"Gripper is {'closed' if episode.grip > 0 else 'open'}.\n"
        f"Primitives so far:\n{done}\n"
        "Write the next primitive."
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": episode.frame()}, {"type": "text", "text": text}]},
    ]


def parse_primitive(text: str) -> dict | None:
    """The first JSON object in the reply, or None when the reply holds no valid JSON object."""
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def reference_plan(task: str, positions: dict[str, list[int]]) -> list[dict] | None:
    """A hand-written plan of primitives that solves the task from the start positions, or None.

    The plans show that the interface can solve a task. They also give the examples for a
    training baseline. A task with no plan returns None.
    """
    obj, goal = np.array(positions["object"], float), np.array(positions["goal"], float)
    move = lambda vector: {"action": "move", "position": [int(round(float(value))) for value in vector]}

    def unit(vector) -> np.ndarray:
        return vector / (np.linalg.norm(vector) + 1e-9)

    if task == "reach-v3":
        return [move(goal)]
    if task == "push-v3":
        # Start 70 mm behind the puck on the line to the goal, then move through the puck.
        behind = obj - 70 * unit((goal - obj) * [1, 1, 0])
        return [move([behind[0], behind[1], obj[2] + 120]), move([behind[0], behind[1], obj[2]]), move([goal[0], goal[1], obj[2]])]
    if task == "pick-place-v3":
        return [move(obj + [0, 0, 120]), move(obj), {"action": "close"}, move(goal)]
    if task == "drawer-close-v3":
        return [move([obj[0], obj[1] - 120, obj[2] + 80]), move([obj[0], obj[1] - 120, obj[2]]), move([goal[0], goal[1], obj[2]])]
    if task == "drawer-open-v3":
        # Put the fingers behind the handle, then pull toward the robot base.
        return [move(obj + [0, 0, 120]), move(obj + [0, 0, -20]), move([goal[0], goal[1] - 40, obj[2] - 20])]
    if task == "button-press-topdown-v3":
        return [move(obj + [0, 0, 120]), move([obj[0], obj[1], goal[2] - 30])]
    if task in ("window-open-v3", "window-close-v3"):
        side = unit((goal - obj) * [1, 0, 0])
        beside = obj - 80 * side
        return [move([beside[0], beside[1], obj[2]]), move([goal[0] + 40 * side[0], obj[1], obj[2]])]
    if task == "peg-insert-side-v3":
        # Grasp the peg, lift it to the height of the hole, then push it into the hole along -x.
        grasp = obj + [-20, 0, 0]
        return [
            move(grasp + [0, 0, 120]),
            move(grasp),
            {"action": "close"},
            move([obj[0], obj[1], goal[2]]),
            move([goal[0] + 300, goal[1], goal[2]]),
            move([goal[0] + 80, goal[1], goal[2]]),
        ]
    # door-open-v3 has no plan: one move of 80 control steps cannot drag the door to the goal.
    return None


def run_episodes(specs: list[tuple[str, int]], generate, batch_size: int = 16) -> list[dict]:
    """Run each (task, seed) episode with a model. `generate` maps conversations to replies, as `tools.generate` does.

    All active episodes ask for their next primitive in one batch, so one batch is one forward pass.
    """
    results = []
    for start in range(0, len(specs), batch_size):
        episodes = [Episode(task, seed) for task, seed in specs[start : start + batch_size]]
        replies_of: dict[int, list[str]] = {index: [] for index in range(len(episodes))}
        while True:
            active = [index for index, episode in enumerate(episodes) if not episode.over]
            if not active:
                break
            replies = generate([conversation(episodes[index]) for index in active])
            for index, reply in zip(active, replies):
                replies_of[index].append(reply["text"])
                episodes[index].apply(parse_primitive(reply["text"]))
        for index, episode in enumerate(episodes):
            results.append(
                {
                    "id": f"{episode.task}/{episode.seed}",
                    "task": episode.task,
                    "seed": episode.seed,
                    "correct": bool(episode.success),
                    "primitives": episode.history,
                    "invalid": episode.invalid,
                    "replies": replies_of[index],
                    "final": episode.positions(),
                }
            )
            episode.close()
    return results

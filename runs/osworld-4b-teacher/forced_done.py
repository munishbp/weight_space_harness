# Forced replay with no model: repeat the recorded actions of a failed task up to the first repeated
# action, then write `done`. The score shows if the stop alone would solve the task.
# Run from the repository root: PYTHONPATH=. uv run python runs/osworld-4b-teacher/forced_done.py <results.json> <socket path>
# The script keeps its results in forced_done.json and skips a task that it holds, so a second run continues.
import json
import sys
from pathlib import Path

from wsp.bench import osworld_screen as ow

RUN = Path("runs/osworld-4b-teacher")
OUT = RUN / "forced_done.json"


def first_repeat(actions: list[dict]) -> int | None:
    """The index of the first action that equals the action before it."""
    for index in range(1, len(actions)):
        if actions[index] == actions[index - 1]:
            return index
    return None


def scripted(actions: list[dict]):
    """A stand-in for `tools.generate` that writes the given actions in order, then `done`."""
    replies = [json.dumps(a) for a in actions] + [json.dumps({"action": "done"})]
    position = iter(replies)
    return lambda conversations: [{"text": next(position)}]


def main(source: str, socket_path: str) -> None:
    results = json.loads(Path(source).read_text())
    tasks = {t["id"]: t for domain in {r["domain"] for r in results} for t in ow.tasks(domain)}
    done = json.loads(OUT.read_text()) if OUT.exists() else []
    held = {r["id"] for r in done}
    client = ow.Client(socket_path)
    for result in results:
        cut = first_repeat(result["actions"])
        if result["correct"] or "error" in result or cut is None or result["id"] in held:
            continue
        # An invalid action in the record has no code, so the stand-in writes it as text that does not parse.
        replay = ow.run_episodes([tasks[result["id"]]], scripted(result["actions"][:cut]), client)[0]
        replay["cut"] = cut
        done.append(replay)
        OUT.write_text(json.dumps(done, indent=1))
        print(result["domain"], result["id"][:8], "cut", cut, "score", replay.get("score"), replay.get("error", ""), flush=True)
    client.close()
    print("solved by the stop:", sum(r["correct"] for r in done), "of", len(done))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

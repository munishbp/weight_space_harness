# Shared state of the later OSWorld scripts (b13 and up). Run after setup.py in a new kernel.
import torch
from PIL import Image

SOCKETS = ("/tmp/wsp-osworld.sock", "/tmp/wsp-osworld2.sock", "/tmp/wsp-osworld3.sock")
ALL_APPS = ["chrome", "gimp", "libreoffice_calc", "libreoffice_impress", "libreoffice_writer", "multi_apps", "os", "thunderbird", "vlc", "vs_code"]
# The split of every application: even positions in `test_all.json` are practice, odd positions are held-out.
TASKS = {t["id"]: t | {"half": "practice" if i % 2 == 0 else "heldout"} for app in ALL_APPS for i, t in enumerate(ow.tasks(app))}
INFEASIBLE = {i for i, t in TASKS.items() if t["evaluator"].get("func") == "infeasible"}
NAMES = ["click", "double_click", "right_click", "drag", "scroll", "type", "key", "wait", "done", "fail"]
PREFIX = tok.encode('{"action": "', add_special_tokens=False)
FIRST = {n: tok.encode(n, add_special_tokens=False)[0] for n in NAMES}
LAST = parts["layers"][-1]
scan = json.load(open(f"{RUN}/scan.json"))
BASE = {r["id"]: r for r in scan + json.load(open(f"{RUN}/infeasible_base.json"))}


def shot(task_id, step, label):
    """The saved screenshot of one step. The tasks of the first sample keep their screenshots in the `sample` directory."""
    path = f"{SHOTS}/{label}/{task_id}-{step}.png"
    return Image.open(path if os.path.exists(path) else f"{SHOTS}/sample/{task_id}-{step}.png").convert("RGB")


def with_system(extra):
    """A `generate` function for the base prompt with more text at the end of the system prompt."""
    def generate(conversations):
        return gen([[{"role": "system", "content": [{"type": "text", "text": ow.SYSTEM_PROMPT + "\n\n" + extra}]}, c[1]] for c in conversations])
    return generate


def compare(label, tasks_, rows):
    """Solved counts against the base scan, for possible tasks and for tasks that are not possible."""
    by = {r["id"]: r for r in rows}
    for name, ids in (("possible", [t["id"] for t in tasks_ if t["id"] not in INFEASIBLE]), ("not possible", [t["id"] for t in tasks_ if t["id"] in INFEASIBLE])):
        if not ids: continue
        before, after = [BASE[i]["correct"] for i in ids], [by[i]["correct"] for i in ids]
        print(f"{label} | {name}: base {sum(before)} of {len(ids)}, now {sum(after)} | {paired_flips(before, after)} | ends with fail {sum(by[i]['actions'][-1].get('action') == 'fail' for i in ids)}", flush=True)


print("common done:", len(TASKS), "tasks,", len(INFEASIBLE), "not possible")

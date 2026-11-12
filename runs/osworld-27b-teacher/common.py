# Shared state of the OSWorld run on the 27B student. Run after setup.py in a new kernel.
import torch
from PIL import Image

SOCKETS = ("/tmp/wsp-osworld.sock", "/tmp/wsp-osworld2.sock", "/tmp/wsp-osworld3.sock")
ALL_APPS = ["chrome", "gimp", "libreoffice_calc", "libreoffice_impress", "libreoffice_writer", "multi_apps", "os", "thunderbird", "vlc", "vs_code"]
# The same split as in the 4B run: even positions in `test_all.json` are practice, odd positions are held-out.
TASKS = {t["id"]: t | {"half": "practice" if i % 2 == 0 else "heldout"} for app in ALL_APPS for i, t in enumerate(ow.tasks(app))}
INFEASIBLE = {i for i, t in TASKS.items() if t["evaluator"].get("func") == "infeasible"}
NAMES = ["click", "double_click", "right_click", "drag", "scroll", "type", "key", "wait", "done", "fail"]
PREFIX = tok.encode('{"action": "', add_special_tokens=False)
FIRST = {n: tok.encode(n, add_special_tokens=False)[0] for n in NAMES}
LAST = parts["layers"][-1]
# The base results of the 4B student on the same tasks, for a comparison of the two students.
BASE_4B = {r["id"]: r for f in ("scan.json", "infeasible_base.json") for r in json.load(open(f"runs/osworld-4b-teacher/{f}"))}


def practice_of(*apps):
    return [t for t in TASKS.values() if t["domain"] in apps and t["half"] == "practice"]


def summary(label, rows):
    ok = [r for r in rows if "error" not in r]
    for name, part in (("possible", [r for r in ok if r["id"] not in INFEASIBLE]), ("not possible", [r for r in ok if r["id"] in INFEASIBLE])):
        if part: print(f"{label} | {name}: solved {sum(r['correct'] for r in part)} of {len(part)} | 4B base on the same tasks {sum(BASE_4B[r['id']]['correct'] for r in part if r['id'] in BASE_4B)} | ends with fail {sum(r['actions'][-1].get('action') == 'fail' for r in part if r['actions'])} | ends with done {sum(r['actions'][-1].get('action') == 'done' for r in part if r['actions'])} | mean seconds {sum(r['seconds'] for r in part) // len(part)}", flush=True)


print("common done:", len(TASKS), "tasks,", len(INFEASIBLE), "not possible")

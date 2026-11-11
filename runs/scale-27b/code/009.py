# State and helpers of the MiniWoB run on the packed 27B model. Run this file in the kernel after each restart.
import json, time
from collections import Counter, defaultdict
import gymnasium
from wsp import attribution, battery, edits, layout
from wsp.bench import miniwob_screen as mb
from wsp.stats import paired_flips, wilson_interval

RUN = "runs/scale-27b"
tok = processor.tokenizer
LAY = layout.find(model)
# Task types that clicks, text, and single keys can solve. Drag, draw, hover, scroll, and timing tasks are out.
CANDIDATES = """ascending-numbers buy-ticket choose-date-easy choose-date-medium choose-date choose-list click-button-sequence
click-button click-checkboxes-large click-checkboxes-soft click-checkboxes click-collapsible-2 click-collapsible click-color
click-dialog-2 click-dialog click-link click-menu-2 click-menu click-option click-pie click-scroll-list click-shades click-shape
click-tab-2-easy click-tab-2-medium click-tab-2-hard click-tab-2 click-tab click-test-2 click-test click-widget count-shape
count-sides daily-calendar email-inbox-delete email-inbox-forward email-inbox-important email-inbox-noscroll email-inbox-reply
email-inbox-star-reply email-inbox enter-date enter-password enter-text-2 enter-text-dynamic enter-text enter-time find-greatest
find-word focus-text-2 focus-text form-sequence-2 form-sequence-3 form-sequence generate-number grid-coordinate guess-number
identify-shape login-user-popup login-user multi-layouts multi-orderings navigate-tree number-checkboxes odd-or-even order-food
phone-book read-table-2 read-table search-engine sign-agreement simple-algebra simple-arithmetic social-media-all
social-media-some social-media text-transform tic-tac-toe unicode-test use-autocomplete use-spinner visual-addition""".split()
TASKS = [task for task in CANDIDATES if f"miniwob/{task}-v1" in gymnasium.registry]

def gen(conversations):
    return tools.generate(conversations, max_new_tokens=48, thinking=False)

def table(results):
    rows = {}
    for task in dict.fromkeys(r["task"] for r in results):
        rs = [r for r in results if r["task"] == task]
        rows[task] = {"n": len(rs), "success": sum(r["correct"] for r in rs), "invalid": sum(r["invalid"] for r in rs),
                      "mean_actions": round(sum(len(r["actions"]) for r in rs) / len(rs), 1)}
    n, k = len(results), sum(r["correct"] for r in results)
    rows["all"] = {"n": n, "success": k, "success_pct": round(100 * k / n, 1), "interval_95": [round(v, 1) for v in wilson_interval(k, n)]}
    return rows

def episodes(label, specs, path=None, reference=None, chunk=8, quiet=False):
    """Run episodes with the model as it is now. Browsers of `chunk` task types are open at one time."""
    t = time.time(); results = []
    tasks = list(dict.fromkeys(task for task, _ in specs))
    for i in range(0, len(tasks), chunk):
        part = set(tasks[i:i + chunk])
        results += mb.run_episodes([s for s in specs if s[0] in part], gen, batch_size=chunk)
        if len(tasks) > chunk: mb.close_all()
    order = {spec: i for i, spec in enumerate(specs)}
    results.sort(key=lambda r: order[(r["task"], r["seed"])])
    rows = table(results)
    if not quiet:
        for task, row in rows.items(): print(f"{label} | {task:26s} {row}", flush=True)
    else: print(f"{label} | all {rows['all']}", flush=True)
    if reference:
        print(f"{label} | against reference:", paired_flips([r["correct"] for r in reference], [r["correct"] for r in results]), flush=True)
    print(f"{label} | {round(time.time() - t)} s", flush=True)
    tools.log("episodes", label=label, **rows["all"], seconds=round(time.time() - t))
    if path: json.dump(results, open(path, "w"))
    return results

def show(results, task, count=3, failed=True):
    for r in [r for r in results if r["task"] == task and (not r["correct"]) == failed][:count]:
        print(r["id"], "|", r["utterance"]); [print("   ", json.dumps(a)) for a in r["actions"]]

# The target portion (chosen after the base scan) and its seeds.
EMAIL = ["email-inbox-reply", "email-inbox-forward", "email-inbox", "email-inbox-noscroll", "email-inbox-star-reply"]
OTHER = [task for task in TASKS if task not in EMAIL]
PRACTICE = [(task, seed) for task in EMAIL for seed in range(20)]
HELDOUT = [(task, seed) for task in EMAIL for seed in range(100, 120)]
HELD_OTHER = [(task, seed) for task in OTHER for seed in range(100, 104)]
acts = lambda r: " ".join({"click": "c", "type": "t", "key": "k", "done": "d", "invalid": "x"}[a["action"]] for a in r["actions"])

print("setup done:", len(TASKS), "task types of", len(CANDIDATES), "candidates")

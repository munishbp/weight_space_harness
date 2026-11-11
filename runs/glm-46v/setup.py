# State and helpers of the Meta-World run on GLM-4.6V-Flash. Run this file in the kernel after each restart.
import json, time
from collections import Counter, defaultdict
from wsp import attribution, edits, evaluate, layout
from wsp.bench import metaworld_text as mw
from wsp.stats import paired_flips, wilson_interval

RUN = "runs/glm-46v"
tok = processor.tokenizer
LAY = layout.find(model)
TASKS = list(mw.INSTRUCTIONS)
# The seed fixes the start positions. Practice and held-out episodes use different seeds.
PRACTICE = [(task, seed) for task in TASKS for seed in range(0, 20)]
HELDOUT = [(task, seed) for task in TASKS for seed in range(100, 120)]

def gen(conversations):
    return tools.generate(conversations, max_new_tokens=48, thinking=False)

def table(results):
    rows = defaultdict(list)
    for r in results: rows[r["task"]].append(r)
    out = {}
    for task, rs in rows.items():
        out[task] = {"n": len(rs), "success": sum(r["correct"] for r in rs), "invalid": sum(r["invalid"] for r in rs),
                     "mean_primitives": round(sum(len(r["primitives"]) for r in rs) / len(rs), 1)}
    wins = sum(r["correct"] for r in results); low, high = wilson_interval(wins, len(results))
    out["all"] = {"n": len(results), "success": wins, "success_pct": round(100 * wins / len(results), 1), "interval_95": [round(low, 1), round(high, 1)]}
    return out

def episodes(label, specs, path=None, reference=None):
    """Run the episodes, print the table per task, and save the results."""
    t = time.time(); results = mw.run_episodes(specs, gen, batch_size=16); tab = table(results)
    for task, row in tab.items(): print(f"{label} | {task:26s} {row}", flush=True)
    if reference:
        ref = {r["id"]: r["correct"] for r in reference}
        print(label, "| against reference:", paired_flips([ref[r["id"]] for r in results], [r["correct"] for r in results]), flush=True)
    tools.log("trial", trial=label, **tab["all"], seconds=round(time.time() - t))
    print(label, "|", round(time.time() - t), "s", flush=True)
    if path: json.dump(results, open(path, "w"))
    return results

print("setup done:", len(PRACTICE), "practice episodes,", len(HELDOUT), "held-out episodes")

def show(results, task, count=3, failed=True):
    """Print the primitives of some episodes of one task, with the start positions from the first prompt."""
    picked = [r for r in results if r["task"] == task and r["correct"] != failed][:count]
    for r in picked:
        ep = mw.Episode(r["task"], r["seed"]); start = ep.positions(); plan = mw.reference_plan(r["task"], start); ep.close()
        print(f"--- {r['id']} success={r['correct']} start={start}")
        print("    model:    ", " | ".join(json.dumps(p) for p in r["primitives"]))
        print("    reference:", " | ".join(json.dumps(p) for p in plan or []))
        print("    final:    ", r["final"])

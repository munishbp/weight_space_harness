# State and helpers of the VLABench run on the 4B model. Run this file in the kernel after each restart.
import importlib, json, os, time
from collections import Counter, defaultdict
from wsp import attribution, battery, edits, evaluate, habit, layout
from wsp.bench import vlabench_plan as vp
from wsp.stats import paired_flips, wilson_interval

RUN = "runs/vlabench-4b-teacher"; tok = processor.tokenizer; LAY = layout.find(model)
T0 = time.time(); stamp = lambda text: print(f"[{(time.time() - T0) / 60:5.1f} min] {text}", flush=True)
# The long-horizon portion: the five composite tasks. Even example numbers are practice, odd numbers are held-out.
PRACTICE, HELDOUT = vp.split(vp.load("Complex"))
# A near-transfer set: the first 10 held-out examples of each task of the five short dimensions.
NEAR = [item for d in vp.DIMENSIONS if d != "Complex" for item in vp.split(vp.load(d))[1] if item["example"] < 20]

def table(rows):
    out = {}
    for task in sorted({r["task"] for r in rows}):
        rs = [r for r in rows if r["task"] == task]
        out[task] = {"n": len(rs), "correct": sum(r["correct"] for r in rs), "parsed": sum(r["parsed"] for r in rs),
                     **{m[:-6]: round(sum(r[m] for r in rs) / len(rs), 3) for m in vp.METRICS}}
    wins = sum(r["correct"] for r in rows); low, high = wilson_interval(wins, len(rows))
    out["all"] = {"n": len(rows), "correct": wins, "correct_pct": round(100 * wins / len(rows), 1), "interval_95": [round(low, 1), round(high, 1)]}
    return out

def trial(label, items, path=None, reference=None, batch=8):
    t = time.time(); rows = evaluate.run(items, vp.conversation, vp.score, batch_size=batch, **vp.GENERATE); tab = table(rows)
    for task, row in tab.items(): print(f"{label} | {task:32s} {row}", flush=True)
    if reference: print(label, "| against reference:", paired_flips([r["correct"] for r in reference], [r["correct"] for r in rows]), flush=True)
    tools.log("trial", trial=label, **tab["all"], seconds=round(time.time() - t)); print(label, "|", round(time.time() - t), "s", flush=True)
    if path: json.dump([{k: v for k, v in r.items() if k != "item"} for r in rows], open(path, "w"))
    return rows

print("setup done:", len(PRACTICE), "practice,", len(HELDOUT), "held-out,", len(NEAR), "near-transfer items")

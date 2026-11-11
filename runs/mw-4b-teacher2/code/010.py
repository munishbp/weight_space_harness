# The held-out verdict for the edit of the second teacher. The main session runs this file, not the second teacher.
# It runs the base model and the edited model on the held-out seeds in this kernel, so both see the same batches.
import json, time
from collections import defaultdict
from wsp import battery
from wsp.bench import metaworld_text as mw
from wsp.stats import paired_flips, wilson_interval

RUN2 = "runs/mw-4b-teacher2"
HELD = [(task, seed) for task in mw.INSTRUCTIONS for seed in range(100, 120)]
gen2 = lambda conversations: tools.generate(conversations, max_new_tokens=48, thinking=False)

def table2(results):
    rows = defaultdict(list)
    for r in results: rows[r["task"]].append(r)
    out = {task: sum(r["correct"] for r in rs) for task, rs in rows.items()}
    wins = sum(r["correct"] for r in results); low, high = wilson_interval(wins, len(results))
    out["all"] = {"n": len(results), "success": wins, "success_pct": round(100 * wins / len(results), 1), "interval_95": [round(low, 1), round(high, 1)]}
    return out

t0 = time.time(); out2 = {}
tools.reset_edit()
base = mw.run_episodes(HELD, gen2, batch_size=16); json.dump(base, open(f"{RUN2}/base_heldout.json", "w"))
print("base held-out", table2(base), round(time.time() - t0), "s", flush=True)
tools.load_edit(f"{RUN2}/edit_final.safetensors")
manifest = json.load(open(f"{RUN2}/edit/manifest.json"))
out2["edit"] = {k: v for k, v in manifest.items() if k != "tensors"}
out2["tensors"] = [{k: v for k, v in t.items() if not isinstance(v, (list, dict))} for t in manifest["tensors"]]
print("edit:", out2["edit"], flush=True)
edit = mw.run_episodes(HELD, gen2, batch_size=16); json.dump(edit, open(f"{RUN2}/edit_heldout.json", "w"))
flips = paired_flips([r["correct"] for r in base], [r["correct"] for r in edit])
print("edit held-out", table2(edit), flips, flush=True)
out2["heldout"] = {"base": table2(base), "edit": table2(edit), "flips": flips}
out2["deviation"] = tools.deviation(battery.neutral_text(64, 256)); print("deviation:", {k: round(v, 4) for k, v in out2["deviation"].items()}, flush=True)
after = battery.run(); out2["battery"] = battery.compare(json.load(open("runs/ss-pro-4b-teacher/battery_base.json")), after)
for k, v in out2["battery"].items(): print("battery", k, v, flush=True)
json.dump(out2, open(f"{RUN2}/verdict.json", "w")); print("verdict done", round((time.time() - t0) / 60, 1), "min", flush=True)

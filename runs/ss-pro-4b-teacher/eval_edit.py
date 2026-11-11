# Measure one edit file with the six measures of the map. Run in the kernel after setup.py.
# Set EDIT_PATH and LABEL before this file runs.
import random, shutil
from wsp import battery
out = {"label": LABEL}
t0 = time.time()
entries = tools.load_edit(EDIT_PATH)
manifest = json.load(open("runs/ss-pro-4b-teacher/edit/manifest.json"))
out["edit"] = {k: v for k, v in manifest.items() if k != "tensors"}
ranks = [e["delta_rank"] for e in manifest["tensors"] if "delta_rank" in e]
out["edit"]["delta_rank_min_max"] = [min(ranks), max(ranks)]
out["edit"]["relative_change_per_tensor_min_max"] = [round(min(e["relative_change"] for e in manifest["tensors"]), 4), round(max(e["relative_change"] for e in manifest["tensors"]), 4)]
print(LABEL, "| loaded in", round(time.time() - t0), "s |", out["edit"], flush=True)
out["deviation"] = tools.deviation(battery.neutral_text(64, 256)); print("deviation:", {k: round(v, 4) for k, v in out["deviation"].items()}, flush=True)
text_items = random.Random(21).sample([i for i in items if i["ui_type"] == "text"], 200)
for name, its in (("heldout_icons", heldout), ("text_targets", text_items), ("practice_icons", practice)):
    res = trial(f"{LABEL}: {name}", its)
    out[name] = {"n": len(its), "base_pct": acc(its), "correct_pct": round(100 * sum(r["correct"] for r in res) / len(res), 1),
                 **paired_flips([base_res[i["id"]]["correct"] for i in its], [r["correct"] for r in res]), "rows": {r["id"]: bool(r["correct"]) for r in res}}
    json.dump(out, open(OUT_PATH, "w"))
after = battery.run(); base_b = json.load(open("runs/ss-pro-4b-teacher/battery_base.json"))
out["battery"] = battery.compare(base_b, after); out["battery_raw"] = after
json.dump(out, open(OUT_PATH, "w"))
for k, v in out["battery"].items(): print("battery", k, v, flush=True)
tools.reset_edit()
print("evaluation done in", round((time.time() - t0) / 60), "min")

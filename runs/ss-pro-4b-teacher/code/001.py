import json, time, copy
from collections import defaultdict
from functools import partial
from wsp import evaluate, layout
from wsp.bench import screenspot_pro as sp
from wsp.stats import paired_flips
tools.log("teacher_start", note="ScreenSpot Pro icon slice")
torch.cuda.empty_cache()
items, images = sp.load()
base_res = json.load(open("runs/ss-pro-4b-baseline/baseline_nothink.json"))
# The slice: icon targets. Split inside each application: even rank to practice, odd rank to held-out.
icons = defaultdict(list)
for it in items:
    if it["ui_type"] == "icon": icons[it["application"]].append(it)
practice, heldout = [], []
for app in sorted(icons):
    for rank, it in enumerate(icons[app]): (practice if rank % 2 == 0 else heldout).append(it)
quick = practice[::2]                                   # the fast trial set
json.dump({"practice": [i["id"] for i in practice], "heldout": [i["id"] for i in heldout], "quick": [i["id"] for i in quick]}, open("runs/ss-pro-4b-teacher/split.json", "w"))
acc = lambda its: round(100 * sum(base_res[i["id"]]["correct"] for i in its) / len(its), 1)
print("icons:", sum(map(len, icons.values())), "| practice", len(practice), acc(practice), "% | held-out", len(heldout), acc(heldout), "% | quick", len(quick), acc(quick), "%")
L = layout.find(model); print("layout:", {k: (v if isinstance(v, str) else f"{len(v)} names, e.g. {v[0]}") for k, v in L.items()})

def conv_with(suffix="", system_extra=""):
    def build(item):
        c = sp.conversation(item, images)
        if suffix: c[1]["content"][1]["text"] = item["instruction"] + suffix
        if system_extra: c[0]["content"][0]["text"] = sp.SYSTEM_PROMPT + system_extra
        return c
    return build
def trial(label, its, build, **opts):
    t = time.time()
    res = evaluate.run(its, build, sp.score, batch_size=4, **{"max_new_tokens": 64, "thinking": False, **opts})
    s = evaluate.summarize(res)["all"]
    flips = paired_flips([base_res[i["id"]]["correct"] for i in its], [r["correct"] for r in res])
    out = {"trial": label, "n": s["n"], "correct_pct": s["correct_pct"], "interval_95": s["interval_95"], "base_pct": acc(its), **flips, "no_point": sum(r["point"] is None for r in res), "seconds": round(time.time() - t)}
    tools.log("trial", **out); print(out, flush=True)
    return res
HINT = " (The target is an icon or a small symbol, not a text label.)"
r_hint = trial("prompt hint after the instruction: target is an icon", quick, conv_with(suffix=HINT))
r_think = trial("thinking on", quick, conv_with(), thinking=True, max_new_tokens=1024)
json.dump({"hint": r_hint, "think": r_think}, open("runs/ss-pro-4b-teacher/prompt_baselines_quick.json", "w"))

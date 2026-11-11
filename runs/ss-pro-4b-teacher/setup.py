# State and helpers of the teacher run. Run this file in the kernel after each restart.
import json, math, time
from collections import Counter, defaultdict
from wsp import attribution, edits, evaluate, layout
from wsp.bench import screenspot_pro as sp
from wsp.stats import paired_flips

tok = processor.tokenizer
items, images = sp.load()
by = {i["id"]: i for i in items}
base_res = json.load(open("runs/ss-pro-4b-baseline/baseline_nothink.json"))
split = json.load(open("runs/ss-pro-4b-teacher/split.json"))
practice, heldout, quick = ([by[i] for i in split[k]] for k in ("practice", "heldout", "quick"))
acc = lambda its: round(100 * sum(base_res[i["id"]]["correct"] for i in its) / len(its), 1)
LAY = layout.find(model)
DIG = list(range(15, 25)); COMMA = tok.encode(",", add_special_tokens=False)[0]; CLOSE = tok.encode("]", add_special_tokens=False)[0]
WANT = DIG + [COMMA, CLOSE]

def conv_with(suffix="", system_extra=""):
    def build(item):
        c = sp.conversation(item, images)
        if suffix: c[1]["content"][1]["text"] = item["instruction"] + suffix
        if system_extra: c[0]["content"][0]["text"] = sp.SYSTEM_PROMPT + system_extra
        return c
    return build

def trial(label, its, build=None, reference=None, **opts):
    """Generate on `its`, score, and compare with the reference results (the base model by default)."""
    t = time.time(); reference = reference or base_res
    res = evaluate.run(its, build or conv_with(), sp.score, batch_size=4, **{"max_new_tokens": 64, "thinking": False, **opts})
    s = evaluate.summarize(res)["all"]
    flips = paired_flips([reference[i["id"]]["correct"] for i in its], [r["correct"] for r in res])
    out = {"trial": label, "n": s["n"], "correct_pct": s["correct_pct"], "interval_95": s["interval_95"],
           "base_pct": round(100 * sum(reference[i["id"]]["correct"] for i in its) / len(its), 1), **flips,
           "no_point": sum(r["point"] is None for r in res), "seconds": round(time.time() - t)}
    tools.log("trial", **out); print(out, flush=True)
    return res

def target_ranges(it):
    W, H = it["img_size"]; x1, y1, x2, y2 = it["bbox"]
    return (1000 * x1 / W, 1000 * x2 / W), (1000 * y1 / H, 1000 * y2 / H)

def viable_first_digits(lo, hi):
    """The first digits of the whole numbers inside a target range."""
    return sorted({int(str(v)[0]) for v in range(math.ceil(lo), math.floor(hi) + 1)})
print("setup done:", len(practice), "practice,", len(heldout), "held-out,", len(quick), "quick")

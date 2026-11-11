# State and helpers of the thinking-fault run on the 4B model. Run this file in the kernel after each restart.
import json, random, time
from collections import Counter
from wsp import attribution, battery, edits, evaluate, layout
from wsp.bench import boxed_math
from wsp.stats import paired_flips

RUN = "runs/gsm8k-4b-teacher"
tok = processor.tokenizer
THINK_END = tok.convert_tokens_to_ids("</think>")
LAY = layout.find(model)
rows = boxed_math.load_jsonl("data/lm_understand-logs/exp8_math/practice/gsm8k_train.jsonl")
random.Random(0).shuffle(rows)
# The same items as the 0.8B calibration: 64 practice items for the search, 128 held-out items for the verdict.
practice, heldout = rows[1000:1064], rows[100:228]

def think_len(gen):
    """The median number of thinking tokens of the replies that close their thinking."""
    v = sorted(r["ids"].index(THINK_END) for r in gen if THINK_END in r["ids"])
    return v[len(v) // 2] if v else None

def behaviour(gen):
    n = len(gen); closed = [THINK_END in r["ids"] for r in gen]
    return {"correct_pct": round(100 * sum(r["correct"] for r in gen) / n, 1),
            "at_limit_pct": round(100 * sum(r["hit_limit"] for r in gen) / n, 1),
            "closed_thinking_pct": round(100 * sum(closed) / n, 1),
            "median_think_len": think_len(gen), "mean_tokens": round(sum(r["new_tokens"] for r in gen) / n)}

def run(items, cap=2048, batch=32, **options):
    return evaluate.run(items, boxed_math.conversation, boxed_math.score, batch_size=batch, max_new_tokens=cap, **{**boxed_math.BLOG_SAMPLING, **options})

def trial(label, items=None, reference=None, cap=2048, **options):
    """Generate on the practice items with the model as it is now, and print the behaviour."""
    t = time.time(); items = items or practice; gen = run(items, cap, **options)
    m = json.load(open(f"{RUN}/edit/manifest.json")) if (tools.RUN_DIR / "edit" / "manifest.json").exists() else {}
    out = {"trial": label, "n": len(items), **behaviour(gen), "weights_changed_pct": round(m.get("changed_weights_pct", 0), 4),
           "model_rel_change": round(m.get("relative_change_of_model", 0), 5), "seconds": round(time.time() - t)}
    if reference: out.update(paired_flips([r["correct"] for r in reference], [r["correct"] for r in gen]))
    tools.log("trial", **out); print(out, flush=True)
    return gen

print("setup done:", len(practice), "practice,", len(heldout), "held-out | </think> id", THINK_END)

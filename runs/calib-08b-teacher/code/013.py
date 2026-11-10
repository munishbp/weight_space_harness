def think_len(gen):
    v = sorted(r["ids"].index(THINK_END) for r in gen if THINK_END in r["ids"])
    return v[len(v) // 2] if v else None
# Baseline: the unedited model with thinking turned off in the prompt.
sampling_off = {**boxed_math.BLOG_SAMPLING, "thinking": False}
t = time.time()
nothink = evaluate.run(practice, boxed_math.conversation, boxed_math.score, batch_size=64, max_new_tokens=2048, **sampling_off)
n = len(nothink)
print({"trial": "BASE, thinking off", "finished_pct": round(100 * sum(not r["hit_limit"] for r in nothink) / n, 1), "correct_pct": round(100 * sum(r["correct"] for r in nothink) / n, 1), "mean_tokens": round(sum(r["new_tokens"] for r in nothink) / n), "seconds": round(time.time() - t)})
sweep = {}
for a in (1.25, 1.5, 1.75):
    sweep[a] = trial_names(f"g out of MLP 21, 22, att 18, alpha {a}", g, [M21, M22, A18], alpha=a)
    print("   median thinking length when closed:", think_len(sweep[a]), "| finished-and-correct share of finished:", round(sum(r["correct"] for r in sweep[a]) / max(1, sum(not r["hit_limit"] for r in sweep[a])), 2))
print("alpha 1.0: median thinking length", think_len(g6), "| alpha 2.0:", think_len(g8))
json.dump({"nothink": nothink, **{f"alpha_{a}": v for a, v in sweep.items()}}, open("runs/calib-08b-teacher/trials_sweep.json", "w"))

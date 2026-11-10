import json, time
N, CAP = 128, 8192
problems = rows[100:100 + N]
def summary(label, res, seconds):
    n = len(res)
    s = {"model": label, "n": n, "cap": CAP, "hit_limit_pct": 100 * sum(r["hit_limit"] for r in res) / n,
         "correct_pct": 100 * sum(r["correct"] for r in res) / n,
         "mean_new_tokens": sum(r["new_tokens"] for r in res) / n, "seconds": round(seconds, 1)}
    print(s, flush=True); return s
t = time.time(); edited = run_math(problems, CAP, seed=0, batch=N); s_edit = summary("edited (published p50c)", edited, time.time() - t)
tools.reset_edit()
t = time.time(); base = run_math(problems, CAP, seed=0, batch=N); s_base = summary("base", base, time.time() - t)
for name, tensor in published.items(): tools.save_edit(name, tensor.float())
json.dump({"base": s_base, "edited": s_edit, "base_rows": base, "edited_rows": edited},
          open("runs/calib-08b-reproduce/math_gsm8k_128.json", "w"))
print("saved; edit is active again")

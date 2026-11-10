import json, time
from wsp.stats import paired_flips
N, CAP, BATCH = 128, 4096, 64
problems = rows[100:100 + N]
def measure(label):
    t = time.time()
    res = evaluate.run(problems, boxed_math.conversation, boxed_math.score, batch_size=BATCH, max_new_tokens=CAP, **boxed_math.BLOG_SAMPLING)
    s = {"model": label, "cap": CAP, "seconds": round(time.time() - t, 1), **evaluate.summarize(res)["all"]}
    print(s, flush=True)
    return res, s
edited, s_edit = measure("edited (published p50c)")
saved = {name: model.get_parameter(name).detach().clone() for name in tools.ORIGINALS}
tools.reset_edit()
base, s_base = measure("base")
for name, tensor in saved.items(): tools.save_edit(name, tensor.float())
flips = {"correct": paired_flips([r["correct"] for r in base], [r["correct"] for r in edited]),
         "finished": paired_flips([not r["hit_limit"] for r in base], [not r["hit_limit"] for r in edited])}
print("paired:", flips)
json.dump({"base": s_base, "edited": s_edit, "paired": flips, "base_rows": base, "edited_rows": edited},
          open("runs/calib-08b-reproduce/math_gsm8k_128.json", "w"))
print("saved; the edit is active again")

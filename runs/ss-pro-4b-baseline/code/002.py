import json, time
from collections import defaultdict
tools.log("baseline_start", thinking=False)
by_app = defaultdict(list)
for it in items: by_app[it["application"]].append(it)
done = {}
t0 = time.time()
for app, its in sorted(by_app.items()):
    res = evaluate.run(its, conv, sp.score, batch_size=4, max_new_tokens=64, thinking=False)
    done.update({r["id"]: r for r in res})
    json.dump(done, open("runs/ss-pro-4b-baseline/baseline_nothink.json", "w"))
    s = evaluate.summarize(res)["all"]
    print(f"{app:16s} n={s['n']:3d} correct={s['correct_pct']:5.1f}  [{time.time()-t0:5.0f} s]", flush=True)
allres = list(done.values())
print("OVERALL", evaluate.summarize(allres)["all"], "| no point:", sum(r["point"] is None for r in allres))
print("by group:", {k: (v["n"], v["correct_pct"]) for k, v in evaluate.summarize(allres, items, by="group").items()})
print("by ui_type:", {k: (v["n"], v["correct_pct"]) for k, v in evaluate.summarize(allres, items, by="ui_type").items()})
tools.log("baseline_end", seconds=round(time.time() - t0))

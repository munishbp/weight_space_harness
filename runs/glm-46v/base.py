# The base GLM-4.6V-Flash model on the Meta-World practice episodes, with thinking off.
T0 = time.time(); stamp = lambda text: print(f"[{(time.time() - T0) / 60:5.1f} min] {text}", flush=True)
tools.log("teacher_start", note="Meta-World MT10 through text waypoints, base run on GLM-4.6V-Flash")
base_practice = episodes("base practice", PRACTICE, f"{RUN}/base_practice.json"); stamp("base practice done")
for task in TASKS:
    plans = Counter(" ".join(p.get("action", "?") for p in r["primitives"]) for r in base_practice if r["task"] == task)
    print(f"{task:26s}", plans.most_common(3), flush=True)

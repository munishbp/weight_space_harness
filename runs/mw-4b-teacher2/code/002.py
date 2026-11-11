from collections import Counter

parts = layout.find(model)
print({k: (v if isinstance(v, str) else (len(v), v[-1])) for k, v in parts.items()})
for task in TASKS:
    rows = [r for r in base if r["task"] == task]
    shapes = Counter(" ".join(p["action"][0] for p in r["primitives"]) + (" OK" if r["correct"] else "") for r in rows)
    print(task, sum(r["correct"] for r in rows), shapes.most_common(4))

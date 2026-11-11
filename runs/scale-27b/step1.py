# Sort the practice episodes of the 27B model by their sequence of action names.
T0 = time.time(); tools.log("search_start", note="27B on Meta-World: look for a wrong action name")
stamp = lambda label: print(f"[{(time.time() - T0) / 60:5.1f} min] {label}", flush=True)
acts = lambda r: " ".join(p["action"][0] for p in r["primitives"])
by_task = defaultdict(list)
for r in base: by_task[r["task"]].append(r)
for task, rows in by_task.items():
    print(f"{task:26s} ok {sum(r['correct'] for r in rows):2d} | ok:", Counter(acts(r) for r in rows if r["correct"]).most_common(2), "| failed:", Counter(acts(r) for r in rows if not r["correct"]).most_common(3))

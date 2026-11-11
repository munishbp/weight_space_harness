# Why the forced replay fails on the button task: the plans next to the reference plan.
for r in forced[:3]:
    ep = mw.Episode(r["task"], r["seed"]); start = ep.positions(); plan = mw.reference_plan(r["task"], start); ep.close()
    print(f"--- {r['id']} success={r['correct']} start={start}")
    print("    model:    ", " | ".join(json.dumps(p) for p in r["primitives"]))
    print("    reference:", " | ".join(json.dumps(p) for p in plan or []))
    print("    final:    ", r.get("final"))
for task in ("drawer-close-v3", "pick-place-v3", "peg-insert-side-v3"):
    show(base_practice, task, count=2)

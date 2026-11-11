bk = {r["id"]: r for r in base}
def acts(r): return " ".join(p["action"][0] for p in r["primitives"])
for task in ("peg-insert-side-v3", "pick-place-v3", "push-v3", "door-open-v3"):
    rows = [r for r in stack_prac if r["task"] == task]
    print(task)
    print("  base  ok :", Counter(acts(bk[r["id"]]) for r in rows if bk[r["id"]]["correct"]).most_common(4))
    print("  base  bad:", Counter(acts(bk[r["id"]]) for r in rows if not bk[r["id"]]["correct"]).most_common(4))
    print("  stack ok :", Counter(acts(r) for r in rows if r["correct"]).most_common(4))
    print("  stack bad:", Counter(acts(r) for r in rows if not r["correct"]).most_common(4))

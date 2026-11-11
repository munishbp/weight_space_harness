def replay(task, seed, plan):
    """Run a fixed plan of primitives in the simulator, with no model. Return the success flag."""
    episode = mw.Episode(task, seed)
    for primitive in plan:
        if episode.over:
            break
        episode.apply(primitive)
    ok = bool(episode.success)
    final = episode.positions()
    episode.close()
    return ok, final


for task in ["button-press-topdown-v3", "drawer-open-v3"]:
    rows = [r for r in base if r["task"] == task]
    for r in rows[:2]:
        print(task, r["seed"], mw.Episode(task, r["seed"]).positions(), r["primitives"], r["final"])
    for name, change in [
        ("drop close", lambda p: [x for x in p if x["action"] != "close"]),
        ("drop close+open", lambda p: [x for x in p if x["action"] not in ("close", "open")]),
        ("drop open", lambda p: [x for x in p if x["action"] != "open"]),
    ]:
        wins = sum(replay(task, r["seed"], change(r["primitives"]))[0] for r in rows)
        print(task, name, wins, "/", len(rows))
    plan_wins = sum(replay(task, r["seed"], mw.reference_plan(task, mw.Episode(task, r["seed"]).positions()))[0] for r in rows)
    print(task, "reference plan", plan_wins)

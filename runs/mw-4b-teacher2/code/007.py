def flips(a, b):
    """Per task: the score of a, the score of b, the episodes that b fixed, and the episodes that b broke."""
    out = {}
    for task in TASKS:
        ra = {r["seed"]: r["correct"] for r in a if r["task"] == task}
        rb = {r["seed"]: r["correct"] for r in b if r["task"] == task}
        out[task.split("-v3")[0]] = (sum(ra.values()), sum(rb.values()),
                                    sum(rb[s] and not ra[s] for s in ra), sum(ra[s] and not rb[s] for s in ra))
    return out


MARGIN = 12.0
habit.fit(MLP_OUT, [(keys, scale, g)], keep, protect, margin=MARGIN)
edit12 = run_set(PRACTICE, "edit12_practice")
print("edit12 vs base", flips(base, edit12), flush=True)
tools.reset_edit()

torch.manual_seed(1234)
g_rand = torch.randn_like(g)
g_rand = g_rand * (g.norm() / g_rand.norm())
habit.fit(MLP_OUT, [(keys, scale, g_rand)], keep, protect, margin=MARGIN)
rand12 = run_set(PRACTICE, "rand12_practice")
print("random vs base", flips(base, rand12), flush=True)
tools.reset_edit()

base2 = run_set(PRACTICE, "base2_practice")
print("base2 vs base", flips(base, base2), flush=True)

# The protection, the keys, and a margin sweep on the MLP output matrix of layer 34.
MLP_OUT = LAY["mlp_outputs"][34]
P_ALL = habit.protection(prompts, NEUTRAL, MLP_OUT); stamp("protection done")
K_PICK, S_PICK = habit.step_keys([c for _, c in pick1], MLP_OUT, LAST, PREFIX, batch=10)
SEEN = torch.tensor([s < 10 for s, _ in pick1])
K_KEEP = habit.step_keys(other1 + later, MLP_OUT, LAST, PREFIX, batch=10)[0]; stamp("keys done")
other1_unseen = [mw.replay(r, 1) for t in TASKS if t != PICK for r in by_task[t] if 10 <= r["seed"] < 14 and len(r["primitives"]) > 1]
pick2_unseen = [mw.replay(r, 2) for r in forced_by[PICK] if r["seed"] >= 10]
def report(label):
    p = probs([c for _, c in pick1]); q = probs(other1_unseen); c2 = probs(pick2_unseen)
    print(f"{label} | pick-place step 1, seeds 0-9 {fmt(p[SEEN])} | seeds 10-19 {fmt(p[~SEEN])} | step 1 of other tasks {fmt(q)} | pick-place step 2 {fmt(c2)}", flush=True)
    return q
q0 = report("base")
W0 = tools.get_weight(MLP_OUT + ".weight").clone()
for margin in (4.0, 8.0, 16.0):
    e = habit.fit(MLP_OUT, [(K_PICK[SEEN], S_PICK[SEEN], g)], K_KEEP, P_ALL, margin)
    q = report(f"margin {margin:g}, relative change {e['relative_change']:.3f}")
    print("    largest change of p(move) at step 1 of other tasks:", round(float((q[:, 0] - q0[:, 0]).abs().max()), 3), flush=True)
    tools.reset_edit()
stamp("margin sweep done")

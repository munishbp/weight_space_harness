# More push seeds: the simulator gives new practice episodes at no cost. Seeds 0-39 are for the fit, 40-79 for the check.
t = time.time()
push_more = mw.run_episodes([("push-v3", s) for s in range(20, 80)], gen, batch_size=16)
PUSH = by_task["push-v3"] + push_more
json.dump(PUSH, open(f"{RUN}/base_push_80.json", "w"))
PX = []
for r in PUSH:
    conv = context(r, 0); text = conv[1]["content"][1]["text"]
    obj = json.loads(text.split("Object position: ")[1].split("\n")[0]); goal = json.loads(text.split("Goal position: ")[1].split("\n")[0])
    first = r["primitives"][0].get("position") or [9999, 9999, 0]
    PX.append({"seed": r["seed"], "obj": obj, "goal": goal, "ok": r["correct"], "conv": conv, "o": first_tokens(obj)[0], "g": first_tokens(goal)[0],
               "to_obj": abs(first[0] - obj[0]) + abs(first[1] - obj[1]) < 10, "to_goal": abs(first[0] - goal[0]) + abs(first[1] - goal[1]) < 10})
for lo, hi in ((0, 40), (40, 80)):
    sub = [x for x in PX if lo <= x["seed"] < hi]
    print(f"push seeds {lo}-{hi - 1}: success {sum(x['ok'] for x in sub)} of {len(sub)} | first move to the object {sum(x['to_obj'] for x in sub)} (success {sum(x['ok'] for x in sub if x['to_obj'])}) | to the goal {sum(x['to_goal'] for x in sub)} (success {sum(x['ok'] for x in sub if x['to_goal'])})")
print(round(time.time() - t), "s")
# Is the choice visible as a state at the step before the first number? Residual after each layer, object-first minus goal-first.
LAYERS = [LAY["layers"][i] for i in (8, 12, 16, 20, 22, 24, 26, 28, 30)]
R = {name: [] for name in LAYERS}
for i in range(0, len(PX), 20):
    out = tools.read([x["conv"] for x in PX[i:i + 20]], LAYERS, append_ids=[PREFIX_MOVE] * len(PX[i:i + 20]))
    for name in LAYERS: R[name].append(out[name])
R = {name: torch.cat(v) for name, v in R.items()}
is_train = torch.tensor([x["seed"] < 40 for x in PX]); to_obj = torch.tensor([x["to_obj"] for x in PX]); to_goal = torch.tensor([x["to_goal"] for x in PX])
DIRS = {}
for name in LAYERS:
    d = R[name][is_train & to_obj].mean(0) - R[name][is_train & to_goal].mean(0); DIRS[name] = d
    proj = R[name] @ (d / d.norm())
    a, b = proj[~is_train & to_obj], proj[~is_train & to_goal]
    auc = float((a[:, None] > b[None, :]).float().mean())
    print(f"{name.split('.')[-1]:>3s} | |difference| {float(d.norm()):6.2f} | residual norm {float(R[name].norm(dim=1).mean()):7.1f} | AUC on seeds 40-79: {auc:.2f}")

# A timed repeat of the recipe on a second habit: in door-open the model opens the gripper after the grasp.
import importlib; importlib.reload(edits)
T0 = time.time(); tools.log("repeat_start", note="door-open: no `open` after the grasp")
DOOR = "door-open-v3"
stamp = lambda label: print(f"[{(time.time() - T0) / 60:5.1f} min] {label}", flush=True)
door3 = CTX3[DOOR]
lp = tools.read(door3, [], append_ids=[PREFIX] * len(door3), logprobs_of=list(NAMES.values()))["logprobs"].exp()
stamp("P at turn 3 of door-open: " + str({w: round(float(lp[:, i].mean()), 3) for i, w in enumerate(NAMES)}))
g_door = attribution.token_direction(NAMES["move"], [NAMES["open"]])
# Protection: the same step of every other context. Door-open turn 3 is the target, so it stays out.
def protect_without(mlp_out, target_task, target_ctx):
    rows = []
    for task in TASKS:
        for name, ctx in (("1", CTX1), ("2", CTX2), ("3", CTX3)):
            if ctx is target_ctx and task == target_task: continue
            if ctx[task]: rows.append(keys_at(ctx[task], mlp_out)[0])
    robot = torch.cat(rows).double()
    text = torch.cat(tools.read_tokens(NEUTRAL, [mlp_out], inputs=True)[mlp_out]).double()
    return robot.T @ robot / len(robot) + text.T @ text / len(text)
P_DOOR = protect_without(MLP_OUT, DOOR, CTX3); stamp("protection set ready")
KD, SD = keys_at(door3, MLP_OUT)
def p_move3(label):
    row = {}
    for task in TASKS:
        if not CTX3[task]: continue
        q = tools.read(CTX3[task], [], append_ids=[PREFIX] * len(CTX3[task]), logprobs_of=list(NAMES.values()))["logprobs"].exp()
        row[task.split("-v3")[0]] = (round(float(q[:10, 0].mean()), 2), round(float(q[10:, 0].mean()), 2))
    stamp(f"{label} | P(move) at turn 3, first half and second half: {row}")
p_move3("base")
for margin in (8.0, 12.0):
    e = edits.fit_keys_to_values(MLP_OUT + ".weight", KD[:10], edits.logit_shift_values(g_door, SD[:10], margin), P_DOOR, 1.0)
    p_move3(f"fit on door seeds 0-9, margin {margin}, relative change {e['relative_change']:.3f}")
    tools.reset_edit()

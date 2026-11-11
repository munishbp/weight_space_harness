from wsp import battery
NEUTRAL = battery.neutral_text(64, 256)
print("turn 3 contexts:", {k.split("-v3")[0]: len(v) for k, v in CTX3.items()})
BUTTON = "button-press-topdown-v3"
gdir = attribution.token_direction(NAMES["move"], [NAMES["close"]])

def protect_of(mlp_out):
    """The second moment of the neuron activations on data that the edit must leave alone."""
    robot = []
    for task in TASKS:
        if task != BUTTON: robot.append(keys_at(CTX2[task], mlp_out)[0])
        robot.append(keys_at(CTX1[task], mlp_out)[0])
        if CTX3[task] and task != BUTTON: robot.append(keys_at(CTX3[task], mlp_out)[0])
    robot = torch.cat(robot).double()
    text = torch.cat(tools.read_tokens(NEUTRAL, [mlp_out], inputs=True)[mlp_out]).double()
    print(f"   protection set: {len(robot)} robot steps, {len(text)} text positions")
    return robot.T @ robot / len(robot) + text.T @ text / len(text)

def p_move(label):
    """P(move) and P(close) at the second turn of every task, with the model as it is now."""
    row = {}
    for task in TASKS:
        lp = torch.cat([tools.read(CTX2[task][i:i + 20], [], append_ids=[PREFIX] * 20, logprobs_of=list(NAMES.values()))["logprobs"] for i in range(0, 20, 20)]).exp()
        row[task.split("-v3")[0]] = (round(float(lp[:10, 0].mean()), 2), round(float(lp[10:, 0].mean()), 2))
    print(label, "| P(move) at turn 2, seeds 0-9 and 10-19:", row, flush=True)
    return row

L = 28
MLP_OUT = LAY["mlp_outs"][L] if "mlp_outs" in LAY else f"model.language_model.layers.{L}.mlp.down_proj"
t = time.time(); PROTECT = {L: protect_of(MLP_OUT)}; print("protect in", round(time.time() - t), "s")
KB, SB = keys_at(CTX2[BUTTON], MLP_OUT)
print("keys", tuple(KB.shape), "| final rms", round(float(SB.mean()), 2), "| |g|", round(float(gdir.norm()), 3))
def values_for(S, margin): return margin * S[:, None] * gdir[None, :] / gdir.pow(2).sum()
base_row = p_move("base")
for strength, margin in ((1.0, 4.0), (0.1, 4.0), (0.01, 4.0)):
    e = edits.fit_keys_to_values(MLP_OUT + ".weight", KB[:10], values_for(SB[:10], margin), PROTECT[L], strength)
    print(f"fit on button seeds 0-9 | L{L} | strength {strength} | margin {margin} | relative change {e['relative_change']:.4f} | rank {e['delta_rank']}")
    p_move("   edited")
    tools.reset_edit()

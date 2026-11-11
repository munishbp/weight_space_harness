# The door fit raised `open` after `close` in the grasp tasks. Add their turn-3 keys to the fit with a zero value.
tools.reset_edit()
OTHER3 = {task: keys_at(CTX3[task], MLP_OUT)[0] for task in TASKS if task != DOOR and CTX3[task]}
half = lambda x, first: x[: len(x) // 2] if first else x[len(x) // 2 :]
def p3(label):
    row = {}
    for task in TASKS:
        if not CTX3[task]: continue
        q = tools.read(CTX3[task], [], append_ids=[PREFIX] * len(CTX3[task]), logprobs_of=list(NAMES.values()))["logprobs"].exp()
        q = half(q, False)  # the second half of the seeds is not in the fit
        row[task.split("-v3")[0][:8]] = (round(float(q[:, 0].mean()), 2), round(float(q[:, 2].mean()), 2))
    stamp(f"{label} | unseen seeds, (P(move), P(open)) at turn 3: {row}")
p3("base")
VD = edits.logit_shift_values(g_door, SD, 12.0)
for w2 in (0.0, 10.0, 100.0):
    tools.load_edit(f"{RUN}/edit_button_L28_m12.safetensors")
    zero_keys = torch.cat([half(k, True) for k in OTHER3.values()]) * w2**0.5
    keys = torch.cat([half(KD, True), zero_keys]); values = torch.cat([half(VD, True), torch.zeros(len(zero_keys), VD.shape[1], dtype=VD.dtype, device=VD.device)])
    e = edits.fit_keys_to_values(MLP_OUT + ".weight", keys, values, P_DOOR, 1.0)
    p3(f"zero-value weight {w2:g}, matrix relative change {e['relative_change']:.3f}")
    tools.reset_edit()

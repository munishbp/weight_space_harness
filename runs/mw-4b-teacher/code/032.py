# Protection from every position of the robot prompts, not only the action step. Then the door fit again.
tools.reset_edit()
def moment_all_positions(convs, drop_last):
    """The sum of k k^T over all positions of each prompt, and the number of positions. `drop_last` leaves out the action step."""
    total, count = None, 0
    for conv in convs:
        K = tools.read([conv], [MLP_OUT], append_ids=[PREFIX], last_positions=100000, inputs=True)[MLP_OUT][0]
        K = (K[:-1] if drop_last else K).to(model.device).double()
        total = K.T @ K if total is None else total + K.T @ K; count += len(K)
    return total, count
total, count = None, 0
for task in TASKS:
    for ctx in (CTX1, CTX2, CTX3):
        if not ctx[task]: continue
        # Seeds 0 to 5 of each context give the protection. Door-open turn 3 is the target, so its action step stays out.
        s, n = moment_all_positions(ctx[task][:6], drop_last=(ctx is CTX3 and task == DOOR))
        total = s if total is None else total + s; count += n
text = torch.cat(tools.read_tokens(NEUTRAL, [MLP_OUT], inputs=True)[MLP_OUT]).to(model.device).double()
P_ALL = total / count + text.T @ text / len(text)
stamp(f"protection from {count} robot positions and {len(text)} text positions")
def written(task, index=15):
    """The total amount of the direction that the current edit adds over one prompt, and the amount at the action step."""
    K = tools.read([CTX3[task][index]], [MLP_OUT], append_ids=[PREFIX], last_positions=100000, inputs=True)[MLP_OUT][0].to(model.device).double()
    c = (K @ (tools.get_weight(MLP_OUT + ".weight").double().to(model.device) - WB).T) @ gn
    return round(float(c[:-1].abs().sum()), 1), round(float(c[-1]), 2)
for strength in (1.0, 10.0):
    tools.load_edit(f"{RUN}/edit_button_L28_m12.safetensors")
    WB = tools.get_weight(MLP_OUT + ".weight").double().to(model.device).clone()
    e = edits.fit_keys_to_values(MLP_OUT + ".weight", half(KD, True), half(VD, True), P_ALL.cpu(), strength)
    print("strength", strength, "| (other positions, action step):", {t.split("-v3")[0]: written(t) for t in ("pick-place-v3", "peg-insert-side-v3", DOOR)})
    p3(f"button + door, all-position protection, strength {strength:g}, matrix relative change {e['relative_change']:.3f}")
    tools.reset_edit()

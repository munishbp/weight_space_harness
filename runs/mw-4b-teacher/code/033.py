tools.reset_edit()
def written(task, index=15):
    """The total amount of the direction that the current edit adds over one prompt, and the amount at the action step."""
    K = tools.read([CTX3[task][index]], [MLP_OUT], append_ids=[PREFIX], last_positions=100000, inputs=True)[MLP_OUT][0].to(model.device).double()
    c = (K @ (tools.get_weight(MLP_OUT + ".weight").double().to(model.device) - WB).T) @ gn.to(model.device)
    return round(float(c[:-1].abs().sum()), 1), round(float(c[-1]), 2)
for strength in (1.0, 10.0):
    tools.load_edit(f"{RUN}/edit_button_L28_m12.safetensors")
    WB = tools.get_weight(MLP_OUT + ".weight").double().to(model.device).clone()
    e = edits.fit_keys_to_values(MLP_OUT + ".weight", half(KD, True), half(VD, True), P_ALL.cpu(), strength)
    print("strength", strength, "| (other positions, action step):", {t.split("-v3")[0]: written(t) for t in ("pick-place-v3", "peg-insert-side-v3", DOOR)})
    p3(f"button + door, all-position protection, strength {strength:g}, matrix relative change {e['relative_change']:.3f}")
    tools.reset_edit()

# The protection now also holds the positions of the reply after the action name (the click position, the text).
tools.reset_edit()
def add_moment_reply(state, conv, reply, drop_action_step):
    """All positions of the prompt and of the recorded reply. `drop_action_step` leaves out the step that writes the action name."""
    ids = tok.encode(reply.strip(), add_special_tokens=False)
    if ids[:len(PREFIX)] != PREFIX: ids = PREFIX          # a reply with another start: keep the prompt and the prefix only
    K = tools.read([conv], [MLP_OUT], append_ids=[ids], last_positions=100000, inputs=True)[MLP_OUT][0].to(model.device).double()
    if drop_action_step:
        step = len(K) - (len(ids) - len(PREFIX)) - 1
        K = torch.cat([K[:step], K[step + 1:]])
    state[0] = K.T @ K if state[0] is None else state[0] + K.T @ K; state[1] += len(K)
state = [None, 0]; prompts = 0
for i in range(0, len(OTHER), 26):
    for r in [r for r in scan if r["task"] in OTHER[i:i + 26] and r["seed"] < 2]:
        for k in range(min(3, len(r["actions"]))):
            conv = context(r, k)
            if conv is not None: add_moment_reply(state, conv, r["replies"][k], False); prompts += 1
    mb.close_all()
for r in base:
    if r["seed"] < 10: add_moment_reply(state, context(r, 0), r["replies"][0], False); prompts += 1
for name, k in STEP.items():
    for r in mail:
        if r["seed"] < 6 and len(r["replies"]) > k:
            conv = context(r, k)
            if conv is not None: add_moment_reply(state, conv, r["replies"][k], drop_action_step=not name.startswith("T")); prompts += 1
P_ALL = (state[0] / state[1] + text.T @ text / len(text)).cpu()
stamp(f"protection with reply positions: {prompts} prompts, {state[1]} positions")
e = fit2(12.0, 1.0); p_table(f"reply positions protected, matrix change {e['relative_change']:.3f}"); tools.reset_edit()
e = fit2(12.0, 1.0, mask=ALL)
m = json.load(open(f"{RUN}/edit/manifest.json"))
print("edit:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items() if k not in ("tensors", "layers_changed")}, "| matrix relative change", round(e["relative_change"], 3), flush=True)
shutil.copy(f"{RUN}/edit/edit.safetensors", f"{RUN}/edit_email_v3_L28_m12.safetensors"); shutil.copy(f"{RUN}/edit/manifest.json", f"{RUN}/manifest_email_v3_L28_m12.json")
edit_prac3 = episodes("edit 3 email practice", PRACTICE, f"{RUN}/edit3_practice.json", reference=base, quiet=True)
stamp("practice trial 3 done")
edit_other3 = episodes("edit 3, other task types", [(r["task"], r["seed"]) for r in scan_other], f"{RUN}/edit3_other_scan.json", reference=scan_other, quiet=True)
for b, a in zip(scan_other, edit_other3):
    if b["correct"] != a["correct"]: print(f"{b['id']:26s} {'fixed ' if a['correct'] else 'BROKEN'} | base: {acts(b):22s} | edit: {acts(a)}")
stamp("scan 3 of the other task types done")
tools.reset_edit()
base_again = episodes("base again, other task types", [(r["task"], r["seed"]) for r in scan_other], f"{RUN}/base_other_scan_repeat.json", reference=scan_other, quiet=True)
stamp("base repeat done")

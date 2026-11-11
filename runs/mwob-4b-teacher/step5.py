# The protection from every position of other prompts, then the fit at the three fault steps.
import importlib; importlib.reload(edits)
def keys_at(convs, mlp_out=MLP_OUT):
    """Neuron activations (the input of the MLP output matrix) and the scale of the final residual, at the action-name step."""
    K, S = [], []
    for i in range(0, len(convs), 16):
        part = convs[i:i + 16]
        out = tools.read(part, [mlp_out, LAST], append_ids=[PREFIX] * len(part), inputs=False)
        S.append(out[LAST].pow(2).mean(1).sqrt())
        K.append(tools.read(part, [mlp_out], append_ids=[PREFIX] * len(part), inputs=True)[mlp_out])
    return torch.cat(K), torch.cat(S)
def add_moment(state, conv, drop_last):
    K = tools.read([conv], [MLP_OUT], append_ids=[PREFIX], last_positions=100000, inputs=True)[MLP_OUT][0]
    K = (K[:-1] if drop_last else K).to(model.device).double()
    state[0] = K.T @ K if state[0] is None else state[0] + K.T @ K; state[1] += len(K)
state = [None, 0]; prompts = 0
# Other task types: seeds 0 and 1 of the base scan, at the first three steps.
for i in range(0, len(OTHER), 26):
    for r in [r for r in scan if r["task"] in OTHER[i:i + 26] and r["seed"] < 2]:
        for k in range(min(3, len(r["actions"]) + 1)):
            conv = context(r, k)
            if conv is not None: add_moment(state, conv, False); prompts += 1
    mb.close_all()
stamp(f"other task types: {prompts} prompts, {state[1]} positions")
# Email prompts: the first step of every practice episode with a seed below 10, and the mail steps with a seed below 6.
for r in base:
    if r["seed"] < 10: add_moment(state, context(r, 0), False); prompts += 1
for name, rows in CTX.items():
    for seed, conv in rows:
        if seed < 6: add_moment(state, conv, drop_last=not name.startswith("T")); prompts += 1
NEUTRAL = battery.neutral_text(64, 256)
text = torch.cat(tools.read_tokens(NEUTRAL, [MLP_OUT], inputs=True)[MLP_OUT]).to(model.device).double()
P_ALL = (state[0] / state[1] + text.T @ text / len(text)).cpu()
stamp(f"protection from {prompts} prompts, {state[1]} positions, and {len(text)} text positions")
g_type = attribution.token_direction(NAMES["click"], [NAMES["type"]]); g_key = attribution.token_direction(NAMES["click"], [NAMES["key"]])
KEYS = {name: keys_at([c for _, c in rows]) for name, rows in CTX.items()}
SEEN = {name: torch.tensor([s < 10 for s, _ in rows]) for name, rows in CTX.items()}
def fit(margin, zero_weight, mask=lambda name: SEEN[name], strength=1.0):
    ks, vs = [], []
    for name, g in (("A", g_type), ("B", g_type), ("C", g_key)):
        full = [n for n in CTX if n.startswith(name)][0]; K, S = KEYS[full]; m = mask(full)
        ks.append(K[m]); vs.append(edits.logit_shift_values(g, S[m], margin))
    full = [n for n in CTX if n.startswith("T")][0]; KT = KEYS[full][0][mask(full)]
    ks.append(KT * zero_weight**0.5); vs.append(torch.zeros(len(KT), vs[0].shape[1], dtype=vs[0].dtype, device=vs[0].device))
    return edits.fit_keys_to_values(MLP_OUT + ".weight", torch.cat(ks), torch.cat(vs), P_ALL, strength)
for margin, zero_weight in ((8.0, 1.0), (12.0, 1.0), (12.0, 10.0)):
    e = fit(margin, zero_weight)
    p_table(f"margin {margin:g}, zero weight {zero_weight:g}, matrix change {e['relative_change']:.3f}")
    tools.reset_edit()
stamp("fit sweep done")

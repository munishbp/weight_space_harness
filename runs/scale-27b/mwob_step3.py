# The protection, the keys, and a fit sweep on one MLP output matrix of the 27B model.
importlib.reload(attribution); importlib.reload(habit)
MLP_OUT = "model.language_model.layers.60.mlp.down_proj"
ids_of = lambda reply: (lambda ids: ids if ids[:len(PREFIX)] == PREFIX else PREFIX)(tok.encode(reply.strip(), add_special_tokens=False))
OTHER_CTX, prompts = [], []
for i in range(0, len(OTHER), 26):
    for r in [r for r in scan4 if r["task"] in OTHER[i:i + 26] and r["seed"] < 2]:
        for k in range(min(3, len(r["actions"]))):
            conv = context(r, k)
            if conv is not None: OTHER_CTX.append(conv); prompts.append((conv, ids_of(r["replies"][k]), None))
    mb.close_all()
for r in forced2:
    if r["seed"] < 10: conv = context(r, 0); OTHER_CTX.append(conv); prompts.append((conv, ids_of(r["replies"][0]), None))
for name, k in STEP.items():
    for r in mail:
        if r["seed"] < 6 and len(r["replies"]) > k:
            conv = context(r, k)
            if conv is not None: prompts.append((conv, ids_of(r["replies"][k]), None if name.startswith("T") else len(PREFIX) - 1))
stamp(f"{len(prompts)} prompts for the protection, {len(OTHER_CTX)} action steps to keep")
NEUTRAL = battery.neutral_text(64, 256)
P_ALL = habit.protection(prompts, NEUTRAL, MLP_OUT); stamp("protection done")
g_type = attribution.token_direction(NAMES["click"], [NAMES["type"]]); g_done = attribution.token_direction(NAMES["click"], [NAMES["done"]])
KEYS = {name: habit.step_keys([c for _, c in rows], MLP_OUT, LAST, PREFIX, batch=8) for name, rows in CTX.items()}
SEEN = {name: torch.tensor([s < 10 for s, _ in rows]) for name, rows in CTX.items()}
K_KEEP = habit.step_keys(OTHER_CTX, MLP_OUT, LAST, PREFIX, batch=8)[0]; stamp("keys done")
full = lambda s: [n for n in CTX if n.startswith(s)][0]
def fit(margin, mask=lambda name: SEEN[name]):
    targets = [(KEYS[full(s)][0][mask(full(s))], KEYS[full(s)][1][mask(full(s))], g) for s, g in (("A", g_type), ("B", g_type), ("C", g_done))]
    keep = torch.cat([KEYS[full("T")][0][mask(full("T"))], K_KEEP])
    return habit.fit(MLP_OUT, targets, keep, P_ALL, margin)
for margin in (8.0, 16.0):
    t = time.time(); e = fit(margin)
    print(f"margin {margin:g}: fit in {time.time() - t:.0f} s |", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items() if not isinstance(v, (list, dict))}, flush=True)
    p_table(f"layer 60, margin {margin:g}"); tools.reset_edit()
stamp("fit sweep done")

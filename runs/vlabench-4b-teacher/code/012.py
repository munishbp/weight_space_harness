# The stacked habit edit: one fit on one MLP matrix, one target group per step of each procedure.
MLP_OUT = [name for name in LAY["mlp_outputs"] if ".27." in name][0]
ORIGINAL_W = tools.get_weight(MLP_OUT + ".weight").clone()
NORM, HEAD = model.get_submodule(LAY["final_norm"]), model.get_output_embeddings()
print("matrix", MLP_OUT)


def read_plan(item, ids):
    """Per position of the prompt and the plan: the MLP input, the RMS of the final residual, and the final residual."""
    conv = vp.conversation(item)
    keys = habit.all_position_keys(conv, ids, MLP_OUT)
    final = tools.read([conv], [LAST], append_ids=[ids], last_positions=1_000_000)[LAST][0]
    offset = len(keys) - len(ids)
    return keys, final, offset


def logprobs_at(final_rows):
    """The next-token log-probabilities from rows of the final residual."""
    weight = HEAD.weight
    with torch.no_grad():
        return torch.log_softmax(HEAD(NORM(final_rows.to(weight.device, weight.dtype))).float(), dim=-1).cpu()


def collect(rows):
    """Group the target positions of forced plans: (task, kind, step) -> keys, scales, and the token that the procedure writes."""
    groups = defaultdict(lambda: {"keys": [], "scale": [], "final": [], "token": None})
    for r in rows:
        keys, final, offset = read_plan(ALL_ITEMS[r["id"]], r["ids"])
        for position, kind, step, token in step_positions(r["ids"]):
            g = groups[(r["task"], kind, step)]
            g["keys"].append(keys[offset + position]); g["final"].append(final[offset + position])
            g["scale"].append(final[offset + position].pow(2).mean().sqrt()); g["token"] = token
    return {k: {"keys": torch.stack(g["keys"]), "scale": torch.stack(g["scale"]), "final": torch.stack(g["final"]), "token": g["token"]} for k, g in groups.items()}


def direction_of(kind, token):
    """Correct token against the tokens that can take its place: the other skill names, or the other separator."""
    if kind == "name": return attribution.token_direction(token, [t for t in SKILL_FIRST if t != token])
    return attribution.token_direction(token, [END if token == GO else GO])


fit_groups = collect(forced_fit); stamp("keys of the fit items")
other_rows = [r for r in other_base if r["parsed"]]
keep_keys, protect_prompts = [], []
for r in other_rows:
    item = ALL_ITEMS[r["id"]]
    keys = habit.all_position_keys(vp.conversation(item), r["ids"], MLP_OUT)
    offset = len(keys) - len(r["ids"])
    keep_keys += [keys[offset + position] for position, _, _, _ in step_positions(r["ids"])]
    if len(protect_prompts) < 48 and r["id"].split("/")[2] in ("example0", "example2", "example4", "example6"):
        protect_prompts.append((vp.conversation(item), r["ids"], None))
# The prompt positions of the target tasks (images and template), with the code fence only: no target position is inside.
protect_prompts += [(vp.conversation(i), FENCE, None) for i in FIT_ITEMS[::5]]
keep = torch.stack(keep_keys)
print(len(fit_groups), "target groups,", sum(len(g["keys"]) for g in fit_groups.values()), "target keys,", len(keep), "keep keys,", len(protect_prompts), "protection prompts")
NEUTRAL = battery.neutral_text(64, 256)
protect = habit.protection(protect_prompts, NEUTRAL, MLP_OUT); stamp("protection done")
targets = [(g["keys"], g["scale"], direction_of(kind, g["token"])) for (task, kind, step), g in sorted(fit_groups.items())]

read_groups = collect(forced_read); stamp("keys of the read items")


def read_check(label):
    """The probability of the token of the procedure at each target position of the read items, with the current weights."""
    summary = defaultdict(list)
    for r in forced_read:
        _, final, offset = read_plan(ALL_ITEMS[r["id"]], r["ids"])
        spots = step_positions(r["ids"])
        logp = logprobs_at(torch.stack([final[offset + p] for p, _, _, _ in spots]))
        for row, (_, kind, step, token) in zip(logp, spots):
            summary[(r["task"], kind)].append(float(row[token].exp()))
    print(label, "| mean probability of the procedure token:", {f"{t[:6]} {k}": round(sum(v) / len(v), 3) for (t, k), v in sorted(summary.items())}, flush=True)


def keep_check(label, rows=None):
    """In plans of other contexts: the probability of the token that the student wrote, at name and separator positions."""
    values = []
    for r in (rows or other_rows[::4]):
        _, final, offset = read_plan(ALL_ITEMS[r["id"]], r["ids"])
        spots = step_positions(r["ids"])
        if not spots: continue
        logp = logprobs_at(torch.stack([final[offset + p] for p, _, _, _ in spots]))
        values += [float(row[token].exp()) for row, (_, _, _, token) in zip(logp, spots)]
    print(label, f"| other contexts: mean probability of the written token {sum(values) / len(values):.3f}, share below 0.5: {sum(v < 0.5 for v in values) / len(values):.3f} (n={len(values)})", flush=True)


read_check("base"); keep_check("base")
for margin in (8.0, 12.0, 16.0):
    tools.reset_edit()
    entry = habit.fit(MLP_OUT, targets, keep, protect, margin=margin)
    read_check(f"margin {margin}"); keep_check(f"margin {margin}")
    print("   entry:", {k: v for k, v in entry.items() if k in ("relative_change", "rank", "name")})
tools.reset_edit(); stamp("margin sweep done")

# The `fail` edit of the 4B run, carried to the 27B student: the same contexts and the same recipe on layer 60 of 64.
# A read of one prompt with a full-size screenshot is slow here, so the fit uses up to 7 steps of each task.
MLP_OUT = "model.language_model.layers.60.mlp.down_proj"
FAIL = FIRST["fail"]
g_fail = attribution.token_direction(FAIL, [t for n, t in FIRST.items() if n != "fail"])
scan_rows = json.load(open(f"{RUN}/scan_practice.json")); more_rows = json.load(open(f"{RUN}/more_base.json"))
LABEL = {r["id"]: "scan" for r in scan_rows} | {r["id"]: "more_base" for r in more_rows} | {r["id"]: "smoke" for r in json.load(open(f"{RUN}/smoke.json"))}
practice = [r for r in scan_rows + more_rows if TASKS[r["id"]]["half"] == "practice" and "error" not in r]
hard = sorted((r for r in practice if r["id"] in INFEASIBLE), key=lambda r: r["id"])
easy = sorted((r for r in practice if r["id"] not in INFEASIBLE), key=lambda r: r["id"])
SOLVED = {r["id"]: r["correct"] for r in practice}
print(len(hard), "practice tasks that are not possible:", Counter(r["domain"] for r in hard), "|", len(easy), "possible practice tasks, base solved", sum(SOLVED[r["id"]] for r in easy))
STEPS_FIT, STEPS_READ = (0, 1, 2, 4, 7, 10, 13), (0, 1, 2, 7, 13)


def shot(task_id, step):
    return Image.open(f"{SHOTS}/{LABEL[task_id]}/{task_id}-{step}.png").convert("RGB")


def prompts(results, steps):
    """One prompt per chosen recorded step: (conversation, task id, step)."""
    return [(ow.conversation(r["instruction"], shot(r["id"], s), r["actions"][:s]), r["id"], s) for r in results for s in steps if s < len(r["actions"])]


def keys_of(rows):
    return habit.step_keys([c for c, _, _ in rows], MLP_OUT, LAST, PREFIX, batch=4)


def fail_probs(rows):
    out = []
    for start in range(0, len(rows), 4):
        part = [c for c, _, _ in rows[start:start + 4]]
        out.append(tools.read(part, [LAST], append_ids=[PREFIX] * len(part), logprobs_of=[FAIL])["logprobs"].exp().float().cpu()[:, 0])
    return torch.cat(out)


def detail(label, rows_hard, rows_easy):
    for name, rows in (("not possible", rows_hard), ("possible", rows_easy)):
        p = fail_probs(rows); by_task = defaultdict(list)
        for (_, task_id, step), value in zip(rows, p): by_task[task_id].append((step, float(value)))
        hit = {i: min(s for s, v in vs if v > 0.5) for i, vs in by_task.items() if any(v > 0.5 for _, v in vs)}
        print(f"{label} | {name}: `fail` wins at a read step in {len(hit)} of {len(by_task)} tasks | first such step {sorted(hit.values())} | of these the base solved {sum(SOLVED[i] for i in hit)} | median p(fail) {float(p.median()):.4f}", flush=True)


t = time.time()
fit_hard, read_hard = prompts(hard[0::2], STEPS_FIT), prompts(hard[1::2], STEPS_READ)
fit_easy, read_easy = prompts(easy[0::2], STEPS_FIT), prompts(easy[1::2], STEPS_READ)
print(len(fit_hard), "target steps,", len(fit_easy), "keep steps for the fit;", len(read_hard), "and", len(read_easy), "steps for the read")
keys, scale = keys_of(fit_hard); keep, _ = keys_of(fit_easy); stamp(f"keys done ({(time.time() - t) / (len(fit_hard) + len(fit_easy)):.1f} s per prompt)")
reply_of = {(r["id"], s): tok.encode(r["replies"][s].strip(), add_special_tokens=False) for r in practice for s in range(len(r["actions"]))}
protect_prompts = [(c, reply_of[(i, s)], None) for c, i, s in fit_easy[:: max(1, len(fit_easy) // 10)]][:10]
protect_prompts += [(c, reply_of[(i, s)], len(PREFIX) - 1) for c, i, s in fit_hard[:: max(1, len(fit_hard) // 10)]][:10]
NEUTRAL = battery.neutral_text(64, 256)
protect = habit.protection(protect_prompts, NEUTRAL, MLP_OUT); stamp("protection done")
detail("base", read_hard, read_easy); stamp("base read done")

def weight_in_range(direction, target_keys, target_scale, keep_keys, margin):
    """The new weight of the fit of `habit.fit`, clamped to the range of the stored format (NVFP4). It writes nothing.

    The first sweep stopped here: `habit.fit` raised, because one new weight passed the largest value
    that the stored format holds. The clamp sits a little inside the range, because float32 rounds the limit up.
    The second sweep ended when the host had no memory left: the system of this solve has 21,760 rows, one copy
    in float64 takes 3.8 GB, and three virtual machines held 12 GB. So the sum is done in place, and the fits
    run with no virtual machine up.
    """
    from wsp.modelopt import E2M1_MAX, E4M3_MAX
    k = torch.cat([target_keys, keep_keys]).double()
    values = edits.logit_shift_values(direction, target_scale, margin)
    values = torch.cat([values, torch.zeros(len(keep_keys), values.shape[1], dtype=values.dtype, device=values.device)]).double()
    system = k.T @ k; system.add_(protect.to(system), alpha=len(k))
    delta = torch.linalg.solve(system, k.T @ values).T.float(); del system
    limit = E2M1_MAX * E4M3_MAX * float(model.get_submodule(MLP_OUT).weight_scale_2)
    new = tools.get_weight(MLP_OUT + ".weight") + delta.to(tools.get_weight(MLP_OUT + ".weight").device)
    over = int((new.abs() > limit).sum())
    print(f"margin {margin}: {over} of {new.numel()} weights outside the range before the clamp | largest {float(new.abs().max()):.4f} | limit {limit:.4f}", flush=True)
    return new.clamp(-limit * (1 - 1e-6), limit * (1 - 1e-6)), over


for margin in (8.0, 12.0):
    tools.reset_edit()
    entry = tools.save_edit(MLP_OUT + ".weight", weight_in_range(g_fail, keys, scale, keep, margin)[0])
    print("relative change", round(entry["relative_change"], 4), flush=True)
    detail(f"margin {margin}", read_hard, read_easy)
tools.reset_edit(); stamp("margin sweep done")

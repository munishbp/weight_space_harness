# The habit edit for `fail`: at the action-name position of a task that is not possible, raise `fail`
# against the other action names. Keys: every recorded step of practice tasks that are not possible.
# Keep: every recorded step of practice tasks that are possible. Fit on one half, read on the other half.
MLP_OUT = [name for name in parts["mlp_outputs"] if ".27." in name][0]
ORIGINAL_W = tools.get_weight(MLP_OUT + ".weight").clone()
FAIL = FIRST["fail"]
g_fail = attribution.token_direction(FAIL, [t for n, t in FIRST.items() if n != "fail"])
infeasible_base = json.load(open(f"{RUN}/infeasible_base.json"))
LABEL = {r["id"]: "scan" for r in scan} | {r["id"]: "infeasible" for r in infeasible_base}
practice = [r for r in scan + infeasible_base if TASKS[r["id"]]["half"] == "practice" and "error" not in r]
hard = sorted((r for r in practice if r["id"] in INFEASIBLE), key=lambda r: r["id"])
easy = sorted((r for r in practice if r["id"] not in INFEASIBLE), key=lambda r: r["id"])
print(len(hard), "practice tasks that are not possible:", Counter(r["domain"] for r in hard), "|", len(easy), "possible practice tasks")


def prompts(results):
    """One prompt per recorded step: (conversation, task id, step)."""
    return [(ow.conversation(r["instruction"], shot(r["id"], s, LABEL[r["id"]]), r["actions"][:s]), r["id"], s) for r in results for s in range(len(r["actions"]))]


def keys_of(rows, batch=8):
    return habit.step_keys([c for c, _, _ in rows], MLP_OUT, LAST, PREFIX, batch=batch)


def fail_probs(rows):
    out = []
    for start in range(0, len(rows), 8):
        part = [c for c, _, _ in rows[start:start + 8]]
        out.append(tools.read(part, [LAST], append_ids=[PREFIX] * len(part), logprobs_of=[FAIL])["logprobs"].exp().float().cpu()[:, 0])
    return torch.cat(out)


def report(label, rows_hard, rows_easy):
    """Per task: the first step where `fail` is the most likely action name would end the episode there."""
    line = []
    for name, rows in (("not possible", rows_hard), ("possible", rows_easy)):
        p = fail_probs(rows); by_task = defaultdict(list)
        for (_, task_id, step), value in zip(rows, p): by_task[task_id].append(float(value))
        line.append(f"{name}: tasks with p(fail) above 0.5 at some step {sum(max(v) > 0.5 for v in by_task.values())} of {len(by_task)}, at step 0 {sum(v[0] > 0.5 for v in by_task.values())}, median over steps {float(p.median()):.3f}")
    print(label, "|", " | ".join(line), flush=True)


fit_hard, read_hard = prompts(hard[0::2]), prompts(hard[1::2])
fit_easy, read_easy = prompts(easy[0::2]), prompts(easy[1::2])
print(len(fit_hard), "target steps,", len(fit_easy), "keep steps for the fit;", len(read_hard), "and", len(read_easy), "steps for the read")
keys, scale = keys_of(fit_hard); keep, _ = keys_of(fit_easy); stamp("keys done")
# The protection: all positions of 12 possible steps and 12 target steps with the reply of the student. The target position is left out.
reply_of = {(r["id"], s): tok.encode(r["replies"][s].strip(), add_special_tokens=False) for r in practice for s in range(len(r["actions"]))}
protect_prompts = [(c, reply_of[(i, s)], None) for c, i, s in fit_easy[:: max(1, len(fit_easy) // 12)]][:12]
protect_prompts += [(c, reply_of[(i, s)], len(PREFIX) - 1) for c, i, s in fit_hard[:: max(1, len(fit_hard) // 12)]][:12]
NEUTRAL = battery.neutral_text(64, 256)
protect = habit.protection(protect_prompts, NEUTRAL, MLP_OUT); stamp("protection done")
report("base", read_hard, read_easy)
for margin in (8.0, 12.0, 16.0):
    tools.reset_edit()
    entry = habit.fit(MLP_OUT, [(keys, scale, g_fail)], keep, protect, margin=margin)
    report(f"margin {margin}", read_hard, read_easy)
tools.reset_edit(); stamp("margin sweep done")

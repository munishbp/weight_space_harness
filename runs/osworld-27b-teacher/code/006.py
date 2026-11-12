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
for margin in (12.0, 16.0, 24.0):
    tools.reset_edit()
    entry = habit.fit(MLP_OUT, [(keys, scale, g_fail)], keep, protect, margin=margin)
    detail(f"margin {margin}", read_hard, read_easy)
tools.reset_edit(); stamp("margin sweep done")

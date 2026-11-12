# Which tasks does `fail` end, and at which step? For a possible task the cost is real only when the base solved it.
SOLVED = {r["id"]: r["correct"] for r in practice}


def detail(label):
    for name, rows in (("not possible", read_hard), ("possible", read_easy)):
        p = fail_probs(rows); by_task = defaultdict(list)
        for (_, task_id, step), value in zip(rows, p): by_task[task_id].append(float(value))
        first = {i: next((s for s, v in enumerate(vs) if v > 0.5), None) for i, vs in by_task.items()}
        hit = {i: s for i, s in first.items() if s is not None}
        print(f"{label} | {name}: `fail` ends {len(hit)} of {len(first)} tasks | first step {sorted(hit.values())} | of these the base solved {sum(SOLVED[i] for i in hit)} | base solved in all {sum(SOLVED[i] for i in first)}", flush=True)


for margin in (16.0, 20.0, 24.0):
    tools.reset_edit(); habit.fit(MLP_OUT, [(keys, scale, g_fail)], keep, protect, margin=margin); detail(f"margin {margin}")
    print("   deviation:", {k: round(v, 4) for k, v in tools.deviation(NEUTRAL[:32]).items() if k in ("mean_kl", "top1_agreement_pct")})
tools.reset_edit(); stamp("detail done")

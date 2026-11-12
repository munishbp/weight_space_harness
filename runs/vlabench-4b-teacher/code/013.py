# Free generation in the plan-only mode with the stacked edit: read items, other contexts, deviation on neutral text.
base_read = plan_trial("plan only, base, read items", READ_ITEMS, f"{RUN}/plan_read_base.json")
trials = {}
for margin in (8.0, 12.0):
    tools.reset_edit()
    habit.fit(MLP_OUT, targets, keep, protect, margin=margin)
    trials[margin] = plan_trial(f"margin {margin}, read items", READ_ITEMS, f"{RUN}/plan_read_m{int(margin)}.json", reference=base_read)
    other = plan_trial(f"margin {margin}, other contexts", OTHER_ITEMS, f"{RUN}/plan_other_m{int(margin)}.json", reference=other_base)
    for metric in ("skill_match_score", "entity_match_score", "total_score"):
        print(f"   other contexts {metric}: base {sum(r[metric] for r in other_base) / len(other_base):.3f} edit {sum(r[metric] for r in other) / len(other):.3f}")
    print("   changed plans in other contexts:", sum(vp.parse(a["text"]) != vp.parse(b["text"]) for a, b in zip(other_base, other)), "of", len(other))
    dev = tools.deviation(NEUTRAL); print("   deviation:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in dev.items()})
    stamp(f"margin {margin} done")
tools.reset_edit()

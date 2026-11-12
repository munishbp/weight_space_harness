# A diagnosis after the verdict. It reads held-out data, so it is not a verdict.
# Question: the sweep read 5 of 15 steps and showed a false `fail` in 3 of 19 possible tasks, but the edit
# gave up on 10 of 17 possible held-out tasks. Does a read of every step show the problem, and at which margin does it stop?
# The values of the fit are linear in the margin, so the weight at margin m is W0 + (m / 12) * (W_EDIT - W0):
# one solve gives every margin.
tools.reset_edit()
W0 = tools.get_weight(MLP_OUT + ".weight").clone(); DELTA = W_EDIT.to(W0) - W0
curve = {}
for margin in (12.0, 8.0, 6.0):
    tools.reset_edit(); tools.save_edit(MLP_OUT + ".weight", W0 + (margin / MARGIN) * DELTA)
    first = {}
    for task in held_hard + held_easy:
        r = base_by_id[task["id"]]
        # One task at a time: the prompts of all tasks hold 2.5 GB of screenshots.
        p = fail_probs(prompts([r], range(len(r["actions"]))))
        first[r["id"]] = next((s for s, v in enumerate(p) if v > 0.5), None)
    hard_hit = [first[t["id"]] for t in held_hard if first[t["id"]] is not None]
    easy_hit = [t["id"] for t in held_easy if first[t["id"]] is not None]
    curve[margin] = {"not_possible_fail_wins": len(hard_hit), "not_possible_n": len(held_hard), "possible_fail_wins": len(easy_hit), "possible_n": len(held_easy),
                     "possible_fail_wins_base_solved": sum(base_by_id[i]["correct"] for i in easy_hit), "possible_base_solved": sum(base_by_id[t["id"]]["correct"] for t in held_easy),
                     "first_step_not_possible": sorted(hard_hit), "first_step_possible": sorted(first[i] for i in easy_hit)}
    print("margin", margin, curve[margin], flush=True); stamp(f"margin {margin} read")
tools.reset_edit(); out["posthoc_strength_curve"] = curve
json.dump(out, open(f"{RUN}/verdict.json", "w"), indent=1); stamp("strength curve done")

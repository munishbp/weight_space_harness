# Both fits on the MLP of layer 28: the button fit on all practice seeds, and the object-first fit on seeds 0-9 only.
mlp_out = MLP_OUT
def apply_button(margin=12.0):
    return edits.fit_keys_to_values(mlp_out + ".weight", KB, values_for(SB, margin), PROTECT[28], 1.0)
def apply_object_first(xs, margin):
    K, S = keys_first([x["conv"] for x in xs], mlp_out)
    D = torch.stack([(W_U[x["o"]] - W_U[x["g"]]) * nw for x in xs])
    return edits.fit_keys_to_values(mlp_out + ".weight", K, margin * S[:, None] * D / D.pow(2).sum(1, keepdim=True), PROTECT[28], 1.0)
def split_rows(results, label):
    for task in ("push-v3", "window-open-v3"):
        a = sum(r["correct"] for r in results if r["task"] == task and r["seed"] < 10); b = sum(r["correct"] for r in results if r["task"] == task and r["seed"] >= 10)
        a0 = sum(r["correct"] for r in base if r["task"] == task and r["seed"] < 10); b0 = sum(r["correct"] for r in base if r["task"] == task and r["seed"] >= 10)
        print(f"{label} | {task}: fit seeds 0-9: {a0} -> {a} of 10 | unseen seeds 10-19: {b0} -> {b} of 10")
apply_button(); e = apply_object_first(train, 12.0)
print("both fits | relative change of the matrix", round(e["relative_change"], 3), "| rank", e["delta_rank"])
t2 = episodes("button fit + object-first fit (margin 12, seeds 0-9)", PRACTICE, f"{RUN}/trial_both_m12.json", reference=base)
split_rows(t2, "margin 12")
tools.reset_edit()

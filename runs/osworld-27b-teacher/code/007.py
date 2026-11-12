# The sweep part of b04_fit_fail.py, run again after the range failure. The kernel holds the keys.
def fit_in_range(direction, target_keys, target_scale, keep_keys, margin):
    """The fit of `habit.fit`, with the new weights clamped to the range of the stored format (NVFP4).

    The first sweep stopped here: `habit.fit` raised, because one new weight passed the largest value
    that the stored format holds. The clamp sits a little inside the range, because float32 rounds the limit up.
    """
    from wsp.modelopt import E2M1_MAX, E4M3_MAX
    k = torch.cat([target_keys, keep_keys]).double()
    values = edits.logit_shift_values(direction, target_scale, margin)
    values = torch.cat([values, torch.zeros(len(keep_keys), values.shape[1], dtype=values.dtype, device=values.device)]).double()
    delta = torch.linalg.solve(len(k) * protect.double() + k.T @ k, k.T @ values).T.float()
    limit = E2M1_MAX * E4M3_MAX * float(model.get_submodule(MLP_OUT).weight_scale_2)
    new = tools.get_weight(MLP_OUT + ".weight") + delta
    over = new.abs() > limit
    print(f"margin {margin}: {int(over.sum())} of {new.numel()} weights outside the range before the clamp | largest {float(new.abs().max()):.4f} | limit {limit:.4f}", flush=True)
    return tools.save_edit(MLP_OUT + ".weight", new.clamp(-limit * (1 - 1e-6), limit * (1 - 1e-6))) | {"clamped": int(over.sum())}


for margin in (12.0, 16.0, 24.0):
    tools.reset_edit()
    entry = fit_in_range(g_fail, keys, scale, keep, margin)
    print("relative change", round(entry["relative_change"], 4), flush=True)
    detail(f"margin {margin}", read_hard, read_easy)
tools.reset_edit(); stamp("margin sweep done")

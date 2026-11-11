# Random control on the 27B model: the same fit with random directions of the same length.
# The first try failed: one new weight passed the largest value that the stored format holds (1.172).
# The clamp sits a little inside the range, because float32 rounds the limit up.
# This run clamps the new weights to that range and prints how many weights the clamp changes.
from wsp.modelopt import E2M1_MAX, E4M3_MAX
tools.reset_edit(); torch.manual_seed(0)
rand_like = lambda g: (lambda r: r / r.norm() * g.norm())(torch.randn_like(g))
r_type, r_done = rand_like(g_type), rand_like(g_done)
groups = [(KEYS[full(s)][0], KEYS[full(s)][1], g) for s, g in (("A", r_type), ("B", r_type), ("C", r_done))]
keep = torch.cat([KEYS[full("T")][0], K_KEEP])
keys = torch.cat([k for k, _, _ in groups] + [keep]).double()
values = [edits.logit_shift_values(g, scale, MARGIN) for _, scale, g in groups]
values = torch.cat(values + [torch.zeros(len(keep), values[0].shape[1], dtype=values[0].dtype, device=values[0].device)]).double()
delta = torch.linalg.solve(len(keys) * P_ALL.double() + keys.T @ keys, keys.T @ values).T.float()
limit = E2M1_MAX * E4M3_MAX * float(model.get_submodule(MLP_OUT).weight_scale_2)
new = tools.get_weight(MLP_OUT + ".weight") + delta
print("weights outside the range before the clamp:", int((new.abs() > limit).sum()), "| limit", round(limit, 4), flush=True)
e = tools.save_edit(MLP_OUT + ".weight", new.clamp(-limit * (1 - 1e-6), limit * (1 - 1e-6)))
print("control fit:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items() if not isinstance(v, (list, dict))}, flush=True)
p_table("27B random control")
ctl = episodes("27B random control email held-out", HELDOUT, f"{RUN}/mwob_control_heldout.json", reference=base_held)
out["random_control"] = {"relative_change": e["relative_change"], "email_heldout": table(ctl)["all"],
                         "flips": paired_flips([r["correct"] for r in base_held], [r["correct"] for r in ctl])}
tools.reset_edit(); json.dump(out, open(f"{RUN}/mwob_verdict.json", "w")); mb.close_all(); stamp("random control done")

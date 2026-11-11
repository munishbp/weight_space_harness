# Random control on the 27B model: the same fit with random directions of the same length.
tools.reset_edit(); torch.manual_seed(0)
rand_like = lambda g: (lambda r: r / r.norm() * g.norm())(torch.randn_like(g))
r_type, r_done = rand_like(g_type), rand_like(g_done)
targets = [(KEYS[full(s)][0], KEYS[full(s)][1], g) for s, g in (("A", r_type), ("B", r_type), ("C", r_done))]
e = habit.fit(MLP_OUT, targets, torch.cat([KEYS[full("T")][0], K_KEEP]), P_ALL, MARGIN)
print("control fit:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items() if not isinstance(v, (list, dict))}, flush=True)
p_table("27B random control")
ctl = episodes("27B random control email held-out", HELDOUT, f"{RUN}/mwob_control_heldout.json", reference=base_held)
out["random_control"] = {"relative_change": e["relative_change"], "email_heldout": table(ctl)["all"],
                         "flips": paired_flips([r["correct"] for r in base_held], [r["correct"] for r in ctl])}
tools.reset_edit(); json.dump(out, open(f"{RUN}/mwob_verdict.json", "w")); mb.close_all(); stamp("random control done")

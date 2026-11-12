# A check after the verdict, not a verdict: the margin comes from a read of held-out steps (b07_strength_curve.py).
# The same edit at margin 8, as W0 + (8 / 12) * (W_EDIT - W0), on the same held-out tasks in the virtual machine.
POSTHOC = 8.0
tools.reset_edit(); entry8 = tools.save_edit(MLP_OUT + ".weight", W0 + (POSTHOC / MARGIN) * DELTA)
rows8 = episodes("edit margin8 heldout", held_hard + held_easy, f"{RUN}/edit_margin8_heldout.json", sockets=SOCKETS)
post = {"margin": POSTHOC, "relative_change": entry8["relative_change"]}
post["not_possible"] = compare("margin 8, not possible", held_hard, rows8); post["possible"] = compare("margin 8, possible", held_easy, rows8)
out["posthoc_margin8"] = post; json.dump(out, open(f"{RUN}/verdict.json", "w"), indent=1); stamp("margin 8 held-out tasks done")
post["deviation"] = deviation_in_parts(NEUTRAL); print("deviation:", {k: round(v, 5) for k, v in post["deviation"].items()}, flush=True)
battery8 = short_battery()
post["battery"] = {name: {"before_pct": float(sum(battery_base[name])), "after_pct": float(sum(battery8[name])), **paired_flips(battery_base[name], battery8[name])} for name in battery_base}
for k, v in post["battery"].items(): print("battery", k, v, flush=True)
tools.reset_edit(); state1 = model.get_submodule(MLP_OUT).state()
out["reset_restores_stored_bytes_after_margin8"] = all(torch.equal(raw(state0[k]), raw(state1[k])) for k in state0)
print("reset restores the stored bytes:", out["reset_restores_stored_bytes_after_margin8"])
json.dump(out, open(f"{RUN}/verdict.json", "w"), indent=1); stamp("margin 8 check done")

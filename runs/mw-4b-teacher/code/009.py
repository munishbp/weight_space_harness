e = edits.fit_keys_to_values(MLP_OUT + ".weight", KB, values_for(SB, 12.0), PROTECT[L], 1.0)
torch.save({"keys": KB, "scales": SB}, f"{RUN}/button_keys_L28.pt")
print(f"fit on the 20 practice button episodes | L{L} | margin 12 | relative change {e['relative_change']:.4f}")
t1 = episodes("edit: no close before a press (L28 fit, margin 12)", PRACTICE, f"{RUN}/trial_fit_L28_m12.json", reference=base)

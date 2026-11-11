# The verdict for the button edit: held-out seeds, a random control, the battery, and the deviation.
tools.reset_edit()
out = {}
base_held = episodes("base held-out", HELDOUT, f"{RUN}/base_heldout.json")
# Random control: the same fit with a random direction of the same length in place of move-minus-close.
torch.manual_seed(7); rand = torch.randn_like(gdir); rand = rand * gdir.norm() / rand.norm()
e = edits.fit_keys_to_values(MLP_OUT + ".weight", KB, 12.0 * SB[:, None] * rand[None, :] / rand.pow(2).sum(), PROTECT[28], 1.0)
ctl = mw.run_episodes([(BUTTON, s) for s in range(100, 120)], gen, batch_size=20)
out["random_control"] = {"relative_change": e["relative_change"], "button_heldout_success": sum(r["correct"] for r in ctl)}
print("random control | relative change", round(e["relative_change"], 4), "| button held-out success", out["random_control"]["button_heldout_success"], "of 20", flush=True)
tools.reset_edit()
entries = tools.load_edit(f"{RUN}/edit_button_L28_m12.safetensors")
manifest = json.load(open(f"{RUN}/edit/manifest.json")); out["edit"] = {k: v for k, v in manifest.items() if k != "tensors"}; out["edit"]["tensor"] = manifest["tensors"][0]
print("edit:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out["edit"].items() if k != "tensor"}, flush=True)
edit_held = episodes("edit held-out", HELDOUT, f"{RUN}/edit_heldout.json", reference=base_held)
out["heldout"] = {"base": table(base_held), "edit": table(edit_held)}
out["deviation"] = tools.deviation(NEUTRAL); print("deviation:", {k: round(v, 4) for k, v in out["deviation"].items()}, flush=True)
json.dump(out, open(f"{RUN}/verdict.json", "w"))
after = battery.run(); base_b = json.load(open("runs/ss-pro-4b-teacher/battery_base.json"))
out["battery"] = battery.compare(base_b, after); out["battery_raw"] = after
for k, v in out["battery"].items(): print("battery", k, v, flush=True)
json.dump(out, open(f"{RUN}/verdict.json", "w"))
tools.reset_edit(); print("verdict done")

# The verdict for the stacked edit (button + door): held-out seeds, a random control, the deviation, and the battery.
tools.log("repeat_end", minutes=round((time.time() - T0) / 60, 1))
out = {"repeat_minutes_to_practice_trial": 14.8}
base_held = json.load(open(f"{RUN}/base_heldout.json"))
# Random control for the door step: the same fit on the button edit with a random direction of the same length.
tools.reset_edit(); tools.load_edit(f"{RUN}/edit_button_L28_m12.safetensors")
torch.manual_seed(7); rand = torch.randn_like(g_door); rand = rand * g_door.norm() / rand.norm()
e = edits.fit_keys_to_values(MLP_OUT + ".weight", KD, edits.logit_shift_values(rand, SD, 12.0), P_ALL.cpu(), 1.0)
ctl = mw.run_episodes([(DOOR, s) for s in range(100, 120)], gen, batch_size=20)
out["random_control"] = {"relative_change": e["relative_change"], "door_heldout_success": sum(r["correct"] for r in ctl)}
print("random control | relative change", round(e["relative_change"], 4), "| door held-out success", out["random_control"]["door_heldout_success"], "of 20", flush=True)
tools.reset_edit(); tools.load_edit(f"{RUN}/edit_button_door_L28.safetensors")
manifest = json.load(open(f"{RUN}/edit/manifest.json")); out["edit"] = {k: v for k, v in manifest.items() if k != "tensors"}; out["edit"]["tensor"] = manifest["tensors"][0]
print("edit:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out["edit"].items() if k != "tensor"}, flush=True)
D2 = tools.get_weight(MLP_OUT + ".weight").double() - W0.to(tools.get_weight(MLP_OUT + ".weight").device)
sv = torch.linalg.svdvals(D2.float()); out["rank_of_change"] = int((sv > 0.01 * sv[0]).sum()); print("rank of the stacked change:", out["rank_of_change"])
stack_held = episodes("button + door held-out", HELDOUT, f"{RUN}/stack_heldout.json", reference=base_held)
out["heldout"] = {"base": table(base_held), "edit": table(stack_held)}
out["practice"] = table(stack_prac)
out["deviation"] = tools.deviation(NEUTRAL); print("deviation:", {k: round(v, 4) for k, v in out["deviation"].items()}, flush=True)
json.dump(out, open(f"{RUN}/verdict_stack.json", "w"))
after = battery.run(); base_b = json.load(open("runs/ss-pro-4b-teacher/battery_base.json"))
out["battery"] = battery.compare(base_b, after); out["battery_raw"] = after
for k, v in out["battery"].items(): print("battery", k, v, flush=True)
json.dump(out, open(f"{RUN}/verdict_stack.json", "w"))
tools.reset_edit(); print("verdict done")

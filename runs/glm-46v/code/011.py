# The verdict for the pick-place edit on GLM-4.6V-Flash: a practice trial, held-out seeds, a random control, the deviation, and the battery.
import shutil
MARGIN = 12.0; out = {"margin": MARGIN, "student": "zai-org/GLM-4.6V-Flash"}
tools.reset_edit()
e = habit.fit(MLP_OUT, [(K_PICK, S_PICK, g)], K_KEEP, P_ALL, MARGIN); report("margin 12, all practice seeds")
out["fit"] = {k: v for k, v in e.items() if not isinstance(v, (list, dict))}
manifest = json.load(open(f"{RUN}/edit/manifest.json")); out["edit"] = {k: v for k, v in manifest.items() if k != "tensors"}
print("edit:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out["edit"].items()}, "| matrix relative change", round(e["relative_change"], 4), flush=True)
shutil.copy(f"{RUN}/edit/edit.safetensors", f"{RUN}/edit_pick_L34_m12.safetensors")
edit_practice = episodes("edit practice", PRACTICE, f"{RUN}/edit_practice.json", reference=base_practice); stamp("practice trial done")
tools.log("search_end", minutes=round((time.time() - T0) / 60, 1))
out["practice"] = {"base": table(base_practice), "edit": table(edit_practice)}
out["deviation"] = tools.deviation(NEUTRAL); print("deviation:", {k: round(v, 4) for k, v in out["deviation"].items()}, flush=True)
tools.reset_edit()
base_held = episodes("base held-out", HELDOUT, f"{RUN}/base_heldout.json"); stamp("base held-out done")
tools.load_edit(f"{RUN}/edit_pick_L34_m12.safetensors")
edit_held = episodes("edit held-out", HELDOUT, f"{RUN}/edit_heldout.json", reference=base_held); stamp("edit held-out done")
out["heldout"] = {"base": table(base_held), "edit": table(edit_held), "flips": paired_flips([r["correct"] for r in base_held], [r["correct"] for r in edit_held])}
# Random control: the same fit with a random direction of the same length.
tools.reset_edit(); torch.manual_seed(0); rand = torch.randn_like(g); rand = rand * g.norm() / rand.norm()
ec = habit.fit(MLP_OUT, [(K_PICK, S_PICK, rand)], K_KEEP, P_ALL, MARGIN)
ctl = mw.run_episodes([(PICK, s) for s in range(100, 120)], gen, batch_size=16)
out["random_control"] = {"relative_change": ec["relative_change"], "pick_heldout_success": sum(r["correct"] for r in ctl)}
print("random control | relative change", round(ec["relative_change"], 4), "| pick-place held-out", out["random_control"]["pick_heldout_success"], "of 20", flush=True)
tools.reset_edit(); json.dump(out, open(f"{RUN}/verdict.json", "w")); stamp("verdict without the battery saved")
before = battery.run(); json.dump(before, open(f"{RUN}/battery_base.json", "w")); stamp("battery of the base model done")
tools.load_edit(f"{RUN}/edit_pick_L34_m12.safetensors"); after = battery.run(); tools.reset_edit()
out["battery"] = battery.compare(before, after)
for k, v in out["battery"].items(): print("battery", k, v, flush=True)
json.dump(out, open(f"{RUN}/verdict.json", "w")); stamp("verdict done")

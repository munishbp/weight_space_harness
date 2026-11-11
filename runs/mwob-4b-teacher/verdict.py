# The verdict for the email edit: held-out seeds of the email tasks and of the other task types, the deviation, and the battery.
EDIT = f"{RUN}/edit_email_v3_L28_m12.safetensors"
tools.reset_edit(); out = {}
HELD_OTHER = [(task, seed) for task in OTHER for seed in range(100, 104)]
base_held = episodes("base email held-out", HELDOUT, f"{RUN}/base_heldout.json")
base_held_other = episodes("base other held-out", HELD_OTHER, f"{RUN}/base_heldout_other.json", quiet=True)
# Random control: the same fit with random directions of the same length.
torch.manual_seed(7); keep_g = (g_type.clone(), g_key.clone())
for g in (g_type, g_key):
    r = torch.randn_like(g); g.copy_(r * g.norm() / r.norm())
e = fit2(12.0, 1.0, mask=ALL); g_type.copy_(keep_g[0]); g_key.copy_(keep_g[1])
ctl = episodes("random control email held-out", HELDOUT, None, reference=base_held, quiet=True)
out["random_control"] = {"relative_change": e["relative_change"], "email_heldout": table(ctl)["all"]}
tools.reset_edit(); tools.load_edit(EDIT)
manifest = json.load(open(f"{RUN}/edit/manifest.json")); out["edit"] = {k: v for k, v in manifest.items() if k != "tensors"}; out["edit"]["tensor"] = manifest["tensors"][0]
edit_held = episodes("edit email held-out", HELDOUT, f"{RUN}/edit_heldout.json", reference=base_held)
edit_held_other = episodes("edit other held-out", HELD_OTHER, f"{RUN}/edit_heldout_other.json", reference=base_held_other, quiet=True)
for b, a in zip(base_held_other, edit_held_other):
    if b["correct"] != a["correct"]: print(f"{b['id']:26s} {'fixed ' if a['correct'] else 'BROKEN'} | base: {acts(b):22s} | edit: {acts(a)}")
out["heldout"] = {"base": table(base_held), "edit": table(edit_held), "flips": paired_flips([r["correct"] for r in base_held], [r["correct"] for r in edit_held])}
out["heldout_other"] = {"base": table(base_held_other)["all"], "edit": table(edit_held_other)["all"], "flips": paired_flips([r["correct"] for r in base_held_other], [r["correct"] for r in edit_held_other])}
out["deviation"] = tools.deviation(NEUTRAL); print("deviation:", {k: round(v, 4) for k, v in out["deviation"].items()}, flush=True)
json.dump(out, open(f"{RUN}/verdict.json", "w"))
after = battery.run(); base_b = json.load(open("runs/ss-pro-4b-teacher/battery_base.json"))
out["battery"] = battery.compare(base_b, after); out["battery_raw"] = after
for k, v in out["battery"].items(): print("battery", k, v, flush=True)
json.dump(out, open(f"{RUN}/verdict.json", "w"))
tools.reset_edit(); print("verdict done")

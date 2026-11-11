# Strength sweep past 2, and a random control: the same removal with a random direction.
for strength in (3.0, 4.0):
    tools.reset_edit(); edits.remove_direction([L30], g, strength)
    trials[strength] = trial(f"remove g from L30 mlp, strength {strength:g}", reference=base_gen)
tools.reset_edit(); torch.manual_seed(0); rand = torch.randn_like(g)
edits.remove_direction([L30], rand, 2.0); ctl = trial("random control: remove a random direction from L30 mlp, strength 2", reference=base_gen)
tools.reset_edit(); json.dump({str(k): v for k, v in trials.items()} | {"control": ctl}, open(f"{RUN}/trials_L30.json", "w")); stamp("sweep and control done")

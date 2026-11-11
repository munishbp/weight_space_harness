# Trial: remove the direction "closing token minus the token that wins" from the output of the MLP of layer 30.
L30 = LAY["mlp_outputs"][30] + ".weight"
trials = {}
for strength in (1.0, 2.0):
    tools.reset_edit(); edits.remove_direction([L30], g, strength)
    trials[strength] = trial(f"remove g from L30 mlp, strength {strength:g}", reference=base_gen)
tools.reset_edit(); json.dump({str(k): v for k, v in trials.items()}, open(f"{RUN}/trials_L30.json", "w")); stamp("L30 trials done")

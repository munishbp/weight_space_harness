# The door fit stacked on the button edit, then a practice trial in the simulator.
import shutil
tools.reset_edit(); T1 = time.time()
tools.load_edit(f"{RUN}/edit_button_L28_m12.safetensors")
e = edits.fit_keys_to_values(MLP_OUT + ".weight", KD, edits.logit_shift_values(g_door, SD, 12.0), P_DOOR, 1.0)
m = json.load(open(f"{RUN}/edit/manifest.json"))
print("stacked edit:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items() if k not in ("tensors", "layers_changed")}, "| matrix relative change", round(e["relative_change"], 3), flush=True)
p_move3("button + door")
q = tools.read(CTX2["button-press-topdown-v3"], [], append_ids=[PREFIX] * 20, logprobs_of=list(NAMES.values()))["logprobs"].exp()
print("button turn 2 with the stack: P(move) %.2f, P(close) %.2f" % (float(q[:, 0].mean()), float(q[:, 1].mean())), flush=True)
shutil.copy(f"{RUN}/edit/edit.safetensors", f"{RUN}/edit_button_door_L28.safetensors"); shutil.copy(f"{RUN}/edit/manifest.json", f"{RUN}/manifest_button_door_L28.json")
stack_prac = episodes("button + door practice", PRACTICE, f"{RUN}/stack_practice.json", reference=base)
tools.log("repeat_trial", minutes_since_repeat_start=round((time.time() - T0) / 60, 1))
print("trial done in", round((time.time() - T1) / 60, 1), "min")

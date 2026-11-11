# The stacked edit with all-position protection: fit on the 20 practice door keys, then a practice trial.
import shutil
tools.reset_edit(); tools.load_edit(f"{RUN}/edit_button_L28_m12.safetensors")
e = edits.fit_keys_to_values(MLP_OUT + ".weight", KD, VD, P_ALL.cpu(), 1.0)
m = json.load(open(f"{RUN}/edit/manifest.json"))
print("stacked edit:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items() if k not in ("tensors", "layers_changed")}, "| matrix relative change of the door step", round(e["relative_change"], 3), flush=True)
shutil.copy(f"{RUN}/edit/edit.safetensors", f"{RUN}/edit_button_door_L28.safetensors"); shutil.copy(f"{RUN}/edit/manifest.json", f"{RUN}/manifest_button_door_L28.json")
stamp("stacked edit saved")
stack_prac = episodes("button + door practice", PRACTICE, f"{RUN}/stack_practice.json", reference=base)
stamp("practice trial done")
for task in TASKS:
    rows = [r for r in stack_prac if r["task"] == task and not r["correct"]]
    if rows: print(task, "failed:", Counter(acts(r) for r in rows).most_common(3))

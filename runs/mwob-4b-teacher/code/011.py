# The fit on all practice seeds, then the practice trial and the scan of the other task types.
import shutil
ALL = lambda name: torch.ones(len(SEEN[name]), dtype=torch.bool)
tools.reset_edit(); e = fit(12.0, 1.0, mask=ALL)
m = json.load(open(f"{RUN}/edit/manifest.json"))
print("edit:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items() if k not in ("tensors", "layers_changed")}, "| matrix relative change", round(e["relative_change"], 3), flush=True)
shutil.copy(f"{RUN}/edit/edit.safetensors", f"{RUN}/edit_email_L28_m12.safetensors"); shutil.copy(f"{RUN}/edit/manifest.json", f"{RUN}/manifest_email_L28_m12.json")
stamp("edit saved")
edit_prac = episodes("edit email practice", PRACTICE, f"{RUN}/edit_practice.json", reference=base)
for task in EMAIL:
    rows = [r for r in edit_prac if r["task"] == task]
    print(task, "| ok:", Counter(acts(r) for r in rows if r["correct"]).most_common(3), "| failed:", Counter(acts(r) for r in rows if not r["correct"]).most_common(4))
stamp("practice trial done")
tools.log("search_end", minutes=round((time.time() - T0) / 60, 1))
scan_other = [r for r in scan if r["task"] in OTHER]
edit_other = episodes("edit, other task types", [(r["task"], r["seed"]) for r in scan_other], f"{RUN}/edit_other_scan.json", reference=scan_other, quiet=True)
changed = {}
for b, a in zip(scan_other, edit_other):
    if b["correct"] != a["correct"]: changed.setdefault(b["task"], []).append("fixed" if a["correct"] else "broken")
print("task types with a changed episode:", changed)
stamp("scan of the other task types done")

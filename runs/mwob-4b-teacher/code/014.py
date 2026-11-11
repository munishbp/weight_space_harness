# The refit on all practice seeds, then the practice trial and the scan of the other task types again.
tools.reset_edit(); e = fit2(12.0, 1.0, mask=ALL)
m = json.load(open(f"{RUN}/edit/manifest.json"))
print("edit:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items() if k not in ("tensors", "layers_changed")}, "| matrix relative change", round(e["relative_change"], 3), flush=True)
shutil.copy(f"{RUN}/edit/edit.safetensors", f"{RUN}/edit_email_keep_L28_m12.safetensors"); shutil.copy(f"{RUN}/edit/manifest.json", f"{RUN}/manifest_email_keep_L28_m12.json")
edit_prac2 = episodes("edit 2 email practice", PRACTICE, f"{RUN}/edit2_practice.json", reference=base)
stamp("practice trial 2 done")
edit_other2 = episodes("edit 2, other task types", [(r["task"], r["seed"]) for r in scan_other], f"{RUN}/edit2_other_scan.json", reference=scan_other, quiet=True)
for b, a in zip(scan_other, edit_other2):
    if b["correct"] != a["correct"]: print(f"{b['id']:26s} {'fixed ' if a['correct'] else 'BROKEN'} | base: {acts(b):22s} | edit: {acts(a)}")
half = lambda rows, lo: [r["correct"] for r in rows if (r["seed"] >= 2) == lo]
print("seeds 0-1 (in the keep set):", paired_flips(half(scan_other, False), half(edit_other2, False)), "| seeds 2-3 (not in it):", paired_flips(half(scan_other, True), half(edit_other2, True)))
stamp("scan 2 of the other task types done")

# The same measures for the LoRA model of 15 minutes.
tools.reset_edit(); out = {}; t0 = time.time()
entries = tools.load_edit("runs/mw-4b-lora15/lora_edit.safetensors")
import shutil; shutil.copy(f"{RUN}/edit/manifest.json", "runs/mw-4b-lora15/manifest.json")
manifest = json.load(open(f"{RUN}/edit/manifest.json")); out["edit"] = {k: v for k, v in manifest.items() if k != "tensors"}
print("LoRA loaded in", round(time.time() - t0), "s |", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out["edit"].items() if k != "layers_changed"}, flush=True)
base_held = json.load(open(f"{RUN}/base_heldout.json"))
lora_held = episodes("LoRA held-out", HELDOUT, "runs/mw-4b-lora15/heldout.json", reference=base_held)
lora_prac = episodes("LoRA practice", PRACTICE, "runs/mw-4b-lora15/practice.json", reference=base)
out["heldout"] = table(lora_held); out["practice"] = table(lora_prac)
out["deviation"] = tools.deviation(NEUTRAL); print("deviation:", {k: round(v, 4) for k, v in out["deviation"].items()}, flush=True)
json.dump(out, open("runs/mw-4b-lora15/evaluation.json", "w"))
after = battery.run(); base_b = json.load(open("runs/ss-pro-4b-teacher/battery_base.json"))
out["battery"] = battery.compare(base_b, after); out["battery_raw"] = after
for k, v in out["battery"].items(): print("battery", k, v, flush=True)
json.dump(out, open("runs/mw-4b-lora15/evaluation.json", "w"))
tools.reset_edit(); print("LoRA evaluation done in", round((time.time() - t0) / 60), "min")

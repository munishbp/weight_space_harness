# The same measures for the LoRA model of 15 minutes (examples: the successful forced-replay episodes of the practice seeds).
import shutil
tools.reset_edit(); out = {}; t0 = time.time()
LORA = "runs/mwob-4b-lora15"
tools.load_edit(f"{LORA}/lora_edit.safetensors"); shutil.copy(f"{RUN}/edit/manifest.json", f"{LORA}/manifest.json")
manifest = json.load(open(f"{RUN}/edit/manifest.json")); out["edit"] = {k: v for k, v in manifest.items() if k != "tensors"}
print("LoRA loaded in", round(time.time() - t0), "s |", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out["edit"].items() if k != "layers_changed"}, flush=True)
base_held = json.load(open(f"{RUN}/base_heldout.json")); base_held_other = json.load(open(f"{RUN}/base_heldout_other.json"))
lora_held = episodes("LoRA email held-out", HELDOUT, f"{LORA}/heldout.json", reference=base_held)
lora_other = episodes("LoRA other held-out", HELD_OTHER, f"{LORA}/heldout_other.json", reference=base_held_other, quiet=True)
changed = Counter()
for b, a in zip(base_held_other, lora_other):
    if b["correct"] != a["correct"]: changed[(b["task"], "fixed" if a["correct"] else "broken")] += 1
print("other task types with a changed episode:", dict(changed))
out["heldout"] = {"base": table(base_held), "lora": table(lora_held), "flips": paired_flips([r["correct"] for r in base_held], [r["correct"] for r in lora_held])}
out["heldout_other"] = {"base": table(base_held_other)["all"], "lora": table(lora_other)["all"], "flips": paired_flips([r["correct"] for r in base_held_other], [r["correct"] for r in lora_other])}
NEUTRAL = battery.neutral_text(64, 256)
out["deviation"] = tools.deviation(NEUTRAL); print("deviation:", {k: round(v, 4) for k, v in out["deviation"].items()}, flush=True)
json.dump(out, open(f"{LORA}/evaluation.json", "w"))
after = battery.run(); base_b = json.load(open("runs/ss-pro-4b-teacher/battery_base.json"))
out["battery"] = battery.compare(base_b, after); out["battery_raw"] = after
for k, v in out["battery"].items(): print("battery", k, v, flush=True)
json.dump(out, open(f"{LORA}/evaluation.json", "w"))
tools.reset_edit(); mb.close_all(); print("LoRA evaluation done in", round((time.time() - t0) / 60), "min")

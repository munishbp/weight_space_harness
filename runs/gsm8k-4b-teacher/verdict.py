# The verdict for the closing-token edit on the 4B model: 128 held-out items at a 4,096 token limit, the deviation, and the battery.
import shutil
tools.log("search_end", minutes=round((time.time() - T0) / 60, 1))
out = {"strength": 2.0, "tensor": L30, "practice": {str(k): behaviour(v) for k, v in trials.items()} | {"base": behaviour(base_gen), "thinking_off": behaviour(off_gen), "random_control": behaviour(ctl)}}
tools.reset_edit()
held_base = run(heldout, cap=4096); print("held-out base:", behaviour(held_base), flush=True); stamp("held-out base done")
held_off = run(heldout, cap=4096, thinking=False); print("held-out base, thinking off:", behaviour(held_off), flush=True)
edits.remove_direction([L30], g, 2.0); shutil.copy(f"{RUN}/edit/edit.safetensors", f"{RUN}/edit_close_L30_s2.safetensors")
manifest = json.load(open(f"{RUN}/edit/manifest.json")); out["edit"] = {k: v for k, v in manifest.items() if k != "tensors"}
out["edit"]["tensor"] = {k: v for k, v in manifest["tensors"][0].items() if not isinstance(v, (list, dict))}
print("edit:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out["edit"].items() if k != "tensor"}, flush=True)
held_edit = run(heldout, cap=4096); flips = paired_flips([r["correct"] for r in held_base], [r["correct"] for r in held_edit])
print("held-out edit:", behaviour(held_edit), flips, flush=True); stamp("held-out edit done")
out["heldout"] = {"base": behaviour(held_base), "thinking_off": behaviour(held_off), "edit": behaviour(held_edit), "flips": flips,
                  "edit_against_thinking_off": paired_flips([r["correct"] for r in held_off], [r["correct"] for r in held_edit])}
out["deviation"] = tools.deviation(battery.neutral_text(64, 256)); print("deviation:", {k: round(v, 4) for k, v in out["deviation"].items()}, flush=True)
json.dump(out, open(f"{RUN}/verdict.json", "w")); json.dump({"base": held_base, "off": held_off, "edit": held_edit}, open(f"{RUN}/heldout_gen.json", "w"))
after = battery.run(); out["battery"] = battery.compare(json.load(open("runs/ss-pro-4b-teacher/battery_base.json")), after)
for k, v in out["battery"].items(): print("battery", k, v, flush=True)
tools.reset_edit(); json.dump(out, open(f"{RUN}/verdict.json", "w")); stamp("verdict done")

# The verdict for the email edit on the 27B model: held-out email seeds, one seed of each other task type, the deviation, and a short battery.
import shutil
from wsp import evaluate
MARGIN = 12.0
out = {"margin": MARGIN}; tools.reset_edit()
HELD_OTHER_1 = [(task, 100) for task in OTHER]
def short_battery():
    """100 items of each battery set with short answers. GSM8K with thinking is too slow at 3 tokens per second."""
    scores = {}
    for name in ("mmlu_pro", "countbench", "chartqa", "refcoco"):
        _, conversation, score, options, _ = battery.SETS[name]
        rows = evaluate.run(battery.items(name)[:100], conversation, score, batch_size=6, **options)
        scores[name] = [bool(r["correct"]) for r in rows]
        print("battery", name, round(100 * sum(scores[name]) / len(rows), 1), flush=True)
    return scores
base_held = episodes("27B base email held-out", HELDOUT, f"{RUN}/mwob_base_heldout.json")
base_other = episodes("27B base other held-out", HELD_OTHER_1, f"{RUN}/mwob_base_heldout_other.json", quiet=True)
battery_base = short_battery(); stamp("base measures done")
W0 = tools.get_weight(MLP_OUT + ".weight").clone(); state0 = {k: v.clone() for k, v in model.get_submodule(MLP_OUT).state().items()}
ALL = lambda name: torch.ones(len(SEEN[name]), dtype=torch.bool)
e = fit(MARGIN, mask=ALL); out["fit"] = {k: v for k, v in e.items() if not isinstance(v, (list, dict))}
manifest = json.load(open(f"{RUN}/edit/manifest.json")); out["edit"] = {k: v for k, v in manifest.items() if k != "tensors"}; out["edit"]["tensor"] = manifest["tensors"][0]
print("edit:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out["edit"].items() if k != "tensor"}, flush=True)
print("tensor:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in out["edit"]["tensor"].items() if not isinstance(v, (list, dict))}, flush=True)
shutil.copy(f"{RUN}/edit/edit.safetensors", f"{RUN}/edit_email_27b_L60.safetensors")
p_table("27B edit")
edit_held = episodes("27B edit email held-out", HELDOUT, f"{RUN}/mwob_edit_heldout.json", reference=base_held)
edit_other = episodes("27B edit other held-out", HELD_OTHER_1, f"{RUN}/mwob_edit_heldout_other.json", reference=base_other, quiet=True)
out["heldout"] = {"base": table(base_held), "edit": table(edit_held), "flips": paired_flips([r["correct"] for r in base_held], [r["correct"] for r in edit_held])}
out["heldout_other"] = {"base": table(base_other)["all"], "edit": table(edit_other)["all"], "flips": paired_flips([r["correct"] for r in base_other], [r["correct"] for r in edit_other])}
def deviation_in_parts(sequences, size=8):
    """`tools.deviation` on a few sequences at a time, joined. The loaded kernel runs all sequences in one pass, which does not fit here."""
    import math
    parts = [tools.deviation(sequences[i:i + size]) for i in range(0, len(sequences), size)]
    n = sum(p["positions"] for p in parts); mean = lambda key: sum(p[key] * p["positions"] for p in parts) / n
    log_ppl = lambda key: math.exp(sum(math.log(p[key]) * p["positions"] for p in parts) / n)
    return {"mean_kl": mean("mean_kl"), "max_kl": max(p["max_kl"] for p in parts), "top1_agreement_pct": mean("top1_agreement_pct"),
            "perplexity_original": log_ppl("perplexity_original"), "perplexity_edited": log_ppl("perplexity_edited"), "positions": n}
out["deviation"] = deviation_in_parts(NEUTRAL); print("deviation:", {k: round(v, 4) for k, v in out["deviation"].items()}, flush=True)
battery_edit = short_battery()
out["battery"] = {name: {"before_pct": 100 * sum(battery_base[name]) / 100, "after_pct": 100 * sum(battery_edit[name]) / 100, **paired_flips(battery_base[name], battery_edit[name])} for name in battery_base}
for k, v in out["battery"].items(): print("battery", k, v, flush=True)
tools.reset_edit(); state1 = model.get_submodule(MLP_OUT).state()
raw = lambda x: x.contiguous().reshape(-1).view(torch.uint8)
out["reset_restores_stored_bytes"] = all(torch.equal(raw(state0[k]), raw(state1[k])) for k in state0)
print("reset restores the stored bytes:", out["reset_restores_stored_bytes"])
json.dump(out, open(f"{RUN}/mwob_verdict.json", "w")); mb.close_all(); stamp("verdict done")

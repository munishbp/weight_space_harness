# Part 1 of the verdict for the `fail` edit on the 27B student (layer 60). Run after b04_fit_fail.py, with no virtual machine up:
# a fit needs about 12 GB of host memory, and three virtual machines hold 12 GB more.
# It measures the base battery and makes the two new weights: the teacher fit and the same fit with a random direction.
import gc
from wsp import evaluate
MARGIN = 12.0
tools.reset_edit()
ends_with_fail = lambda r: r["actions"][-1].get("action") == "fail"


def short_battery():
    """100 items of each battery set with short answers. GSM8K with thinking is too slow at 3 tokens per second."""
    scores = {}
    for name in ("mmlu_pro", "countbench", "chartqa", "refcoco"):
        _, conversation, score, options, _ = battery.SETS[name]
        rows = evaluate.run(battery.items(name)[:100], conversation, score, batch_size=6, **options)
        scores[name] = [bool(r["correct"]) for r in rows]
        print("battery", name, round(100 * sum(scores[name]) / len(rows), 1), flush=True)
    return scores


def deviation_in_parts(sequences, size=8):
    """`tools.deviation` on a few sequences at a time, joined. All sequences in one pass do not fit in the memory."""
    import math
    chunks = [tools.deviation(sequences[i:i + size]) for i in range(0, len(sequences), size)]
    n = sum(p["positions"] for p in chunks); mean = lambda key: sum(p[key] * p["positions"] for p in chunks) / n
    log_ppl = lambda key: math.exp(sum(math.log(p[key]) * p["positions"] for p in chunks) / n)
    return {"mean_kl": mean("mean_kl"), "max_kl": max(p["max_kl"] for p in chunks), "top1_agreement_pct": mean("top1_agreement_pct"),
            "perplexity_original": log_ppl("perplexity_original"), "perplexity_edited": log_ppl("perplexity_edited"), "positions": n}


def compare(label, tasks_, rows):
    """Paired counts of one held-out group against the base run of the same tasks."""
    ids = [t["id"] for t in tasks_]; by = {r["id"]: r for r in rows}
    before, after = [base_by_id[i]["correct"] for i in ids], [by[i]["correct"] for i in ids]
    result = {"n": len(ids), "base_solved": sum(before), "solved": sum(after), "flips": paired_flips(before, after),
              "base_ends_with_fail": sum(ends_with_fail(base_by_id[i]) for i in ids), "ends_with_fail": sum(ends_with_fail(by[i]) for i in ids),
              "fail_step": sorted(len(by[i]["actions"]) - 1 for i in ids if ends_with_fail(by[i])), "bridge_errors": sum("error" in by[i] for i in ids)}
    print(label, result, flush=True)
    return result


battery_base = short_battery(); stamp("base battery done")
state0 = {k: v.clone() for k, v in model.get_submodule(MLP_OUT).state().items()}
all_hard, all_easy = prompts(hard, STEPS_FIT), prompts(easy, STEPS_FIT)
keys_all, scale_all = keys_of(all_hard); keep_all, _ = keys_of(all_easy)
print(len(all_hard), "target steps of", len(hard), "tasks;", len(all_easy), "keep steps of", len(easy), "tasks")
W_EDIT, over_edit = weight_in_range(g_fail, keys_all, scale_all, keep_all, MARGIN); stamp("teacher weight done")
# The random control: the same keys, the same margin, a random direction of the same length.
torch.manual_seed(0); g_rand = torch.randn_like(g_fail); g_rand = g_rand * g_fail.norm() / g_rand.norm()
W_RAND, over_rand = weight_in_range(g_rand, keys_all, scale_all, keep_all, MARGIN); stamp("control weight done")
# A copy on disk: a kernel that stops in part 2 does not need the fits again.
torch.save({"margin": MARGIN, "edit": W_EDIT.cpu(), "random": W_RAND.cpu()}, f"{SHOTS}/verdict_weights.pt")
del protect; gc.collect(); torch.cuda.empty_cache(); stamp("part 1 done: start the virtual machines now")

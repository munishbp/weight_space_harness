# A decoding option: the reply starts with the code fence, so the student writes the plan with no prose before it.
# In this mode the plan is decided at the name positions. Base scores, and the name probabilities at step 2 and step 3.
FENCE = tok.encode("```json\n[", add_special_tokens=False)
OPTIONS = {"thinking": False, "max_new_tokens": 512}


def plan_trial(label, items, path=None, reference=None):
    """A trial in the plan-only mode. The rows have the fields of `trial`."""
    t = time.time(); rows = []
    for start in range(0, len(items), 8):
        chunk = items[start:start + 8]; torch.manual_seed(start)
        replies = tools.generate([vp.conversation(i) for i in chunk], append_ids=[FENCE] * len(chunk), **OPTIONS)
        rows += [{"id": i["id"], "dimension": i["dimension"], "task": i["task"], **r, **vp.score(i, r["text"])} for i, r in zip(chunk, replies)]
    if path: json.dump(rows, open(path, "w"))
    tab = table(rows)
    for task, row in tab.items(): print(f"{label} | {task:32s} {row}", flush=True)
    if reference: print(label, "| against reference:", paired_flips([r["correct"] for r in reference], [r["correct"] for r in rows]), flush=True)
    tools.log("trial", trial=label, **tab["all"], seconds=round(time.time() - t))
    return rows


plan_small = plan_trial("plan only, base", SMALL, f"{RUN}/plan_small.json"); stamp("plan-only base on the small set")
print(repr(plan_small[0]["text"][:300]))
for task in sorted(SKELETON):
    for step in (2, 3):
        probs = []
        for r in [r for r in plan_small if r["task"] == task and r["parsed"] and r["steps"] >= step][:16]:
            prefix = name_prefix(r["ids"], step)
            if prefix is None: continue
            out = tools.read([vp.conversation(ITEM[r["id"]])], [LAST], append_ids=[prefix], logprobs_of=NAME_IDS)
            probs.append(out["logprobs"][0].exp().float().cpu())
        if not probs: print(task, "step", step, "no reply reaches this step"); continue
        p = torch.stack(probs); correct = SKELETON[task][step - 1]; col = p[:, NAMES.index(correct)]
        print(f"{task:30s} n={len(probs):2d} step {step} | " + " ".join(f"{n} {float(m):.4f}" for n, m in zip(NAMES, p.mean(0))),
              f"| correct `{correct}`: median {float(col.median()):.5f}, quartiles {[round(float(q), 5) for q in torch.quantile(col, torch.tensor([0.25, 0.75]))]}")
stamp("plan-only step probabilities done")

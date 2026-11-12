# The probability of each skill name at the second step of the plan, in the replies that the student wrote.
# A correct name with a visible probability is a habit fault. A correct name with no probability is a missing ability.
NAMES = ["pick", "place", "push", "pull", "pour"]
NAME_IDS = [SKILL[n][0] for n in NAMES]
LAST = LAY["layers"][-1]
ITEM = {i["id"]: i for i in PRACTICE}


def name_prefix(ids, step):
    """The reply ids up to the position that writes the skill name of one step (1 for the first step), or None."""
    for n in range(1, len(ids) + 1):
        text = tok.decode(ids[:n])
        if "```json" in text and text.endswith('"name": "') and text.split("```json", 1)[1].count('"name": "') == step:
            return ids[:n]
    return None


second = {}
for task in sorted(SKELETON) + ["cook_dishes"]:
    rows = [r for r in base_practice if r["task"] == task and r["parsed"] and r["steps"] >= 2][:16]
    probs = []
    for r in rows:
        prefix = name_prefix(r["ids"], 2)
        if prefix is None: continue
        out = tools.read([vp.conversation(ITEM[r["id"]])], [LAST], append_ids=[prefix], logprobs_of=NAME_IDS)
        probs.append(out["logprobs"][0].exp().float().cpu())
    second[task] = torch.stack(probs)
    mean = second[task].mean(0); correct = (SKELETON.get(task) or ["pick", "place"])[1]
    print(f"{task:30s} n={len(probs):2d} step 2 | " + " ".join(f"{n} {float(m):.4f}" for n, m in zip(NAMES, mean)),
          f"| correct name `{correct}`: median {float(second[task][:, NAMES.index(correct)].median()):.5f}, max {float(second[task][:, NAMES.index(correct)].max()):.4f}")
stamp("step probabilities done")

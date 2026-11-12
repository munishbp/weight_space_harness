# Forced replay for the three tasks with one fixed procedure. The reply gets the skill names of the
# procedure and its length. The student writes the prose before the plan and the objects of each step.
# The scores are the ceiling of an edit that writes the procedure.
SKELETON = {
    "book_rearrange": ["pick", "push", "pull"] * 3,
    "hammer_nail_and_hang_picture": ["pick", "push", "place", "pick", "push"],
    "take_chemistry_experiment": ["pick", "pour", "place"] * 3,
}
FIXED = sorted(SKELETON)
SKILL = {name: tok.encode(name, add_special_tokens=False) for name in {"pick", "push", "pull", "place", "pour"}}
print("skill tokens", SKILL)
assert all(len(ids) == 1 for ids in SKILL.values()), "A skill name has more than one token. The forced replay needs one."
VOCAB = [tok.decode([i]) for i in range(len(tok))]
CLOSES = torch.tensor([i for i, text in enumerate(VOCAB) if "]" in text])
COMMAS = torch.tensor([i for i, text in enumerate(VOCAB) if text.strip().startswith(",")])
print("tokens with ]:", len(CLOSES), "| tokens that start with a comma:", len(COMMAS))


class Skeleton:
    """Force the skill names of each row, and the number of steps, inside the ```json block."""

    def __init__(self, skeletons):
        self.skeletons, self.start = skeletons, None

    def __call__(self, input_ids, scores):
        if self.start is None: self.start = input_ids.shape[1]
        for row, names in enumerate(self.skeletons):
            text = tok.decode(input_ids[row, self.start:])
            if "```json" not in text: continue
            plan = text.split("```json", 1)[1]
            if "```" in plan: continue
            written = plan.count('"name"')
            if plan.endswith('"name": "') and written <= len(names):
                forced = SKILL[names[written - 1]][0]
                value = scores[row, forced].clone(); scores[row] = -float("inf"); scores[row, forced] = value
            elif plan.rstrip().endswith("}") and plan.count("{") == plan.count("}") and written > 0:
                # A step is complete. More steps of the procedure: no end of the list. No more steps: no comma.
                scores[row, CLOSES if written < len(names) else COMMAS] = -float("inf")
        return scores


def forced_trial(label, items, path):
    rows = []
    for start in range(0, len(items), 8):
        chunk = items[start:start + 8]
        torch.manual_seed(start)
        replies = tools.generate([vp.conversation(i) for i in chunk], logits_processor=[Skeleton([SKELETON[i["task"]] for i in chunk])], **vp.GENERATE)
        rows += [{"id": i["id"], "task": i["task"], **r, **vp.score(i, r["text"])} for i, r in zip(chunk, replies)]
    json.dump(rows, open(path, "w"))
    for task, row in table(rows).items(): print(f"{label} | {task:32s} {row}", flush=True)
    return rows


SMALL = [i for i in PRACTICE if i["task"] in FIXED and i["example"] < 40]
base_practice = json.load(open(f"{RUN}/base_practice.json"))
small_ids = {i["id"] for i in SMALL}
for task, row in table([r for r in base_practice if r["id"] in small_ids]).items(): print(f"base, same items | {task:32s} {row}", flush=True)
forced = forced_trial("forced names", SMALL, f"{RUN}/forced_names.json"); stamp("forced replay done")
short = lambda seq: [(s["name"], s["params"].get("target_entity_name"), s["params"].get("target_container_name")) for s in seq]
for task in FIXED:
    for r in [r for r in forced if r["task"] == task][:3]:
        seq = vp.parse(r["text"]); print(task[:12], "given:", short(seq) if seq else repr(r["text"][-200:]))

# The keys for a stacked habit edit in the plan-only mode: one target per step of each fixed procedure.
# A target position is a position that writes a skill name, or a position that ends a step (` },` goes on, ` }` ends the list).
import re

ALL_SKILLS = sorted(set(re.findall(r"\*\*(\w+)\*\*:", vp.prompt())))
SKILL_FIRST = sorted({tok.encode(name, add_special_tokens=False)[0] for name in ALL_SKILLS})
GO, END = tok.encode(" },", add_special_tokens=False), tok.encode(" }", add_special_tokens=False)
assert len(GO) == 1 and len(END) == 1, "The step separators need one token each."
GO, END = GO[0], END[0]
print("skills", ALL_SKILLS, "| first tokens", len(SKILL_FIRST), "| go", GO, "end", END)
ENDS = torch.tensor([i for i, text in enumerate(VOCAB) if text.strip().startswith("}") and "," not in text])
GOES = torch.tensor([i for i, text in enumerate(VOCAB) if text.strip().startswith("},")])


class PlanSkeleton:
    """In the plan-only mode: force the skill names and the number of steps of each row."""

    def __init__(self, skeletons):
        self.skeletons, self.start = skeletons, None

    def __call__(self, input_ids, scores):
        if self.start is None: self.start = input_ids.shape[1]
        for row, names in enumerate(self.skeletons):
            plan = tok.decode(input_ids[row, self.start:])
            if "]" in plan: continue
            written, depth = plan.count('"name"'), plan.count("{") - plan.count("}")
            if plan.endswith('"name": "') and written <= len(names):
                forced = SKILL[names[written - 1]][0]
                value = scores[row, forced].clone(); scores[row] = -float("inf"); scores[row, forced] = value
            elif depth == 1 and plan.rstrip().endswith("}"):
                # The parameters are closed. The next brace ends the step: it goes on, or it ends the list.
                scores[row, ENDS if written < len(names) else GOES] = -float("inf")
            if written < len(names): scores[row, CLOSES] = -float("inf"); scores[row, TICKS] = -float("inf")
        return scores


def forced_plans(items):
    """Plans with the names of the procedure and the objects of the student."""
    rows = []
    for start in range(0, len(items), 8):
        chunk = items[start:start + 8]; torch.manual_seed(start)
        replies = tools.generate([vp.conversation(i) for i in chunk], append_ids=[FENCE] * len(chunk),
                                 logits_processor=[PlanSkeleton([SKELETON[i["task"]] for i in chunk])], **OPTIONS)
        rows += [{"id": i["id"], "dimension": i["dimension"], "task": i["task"], **r, **vp.score(i, r["text"])} for i, r in zip(chunk, replies)]
    return rows


def step_positions(ids):
    """The target positions of one plan: (index of the position that writes the token, kind, step, token written)."""
    found, text = [], ""
    for j, token in enumerate(ids):
        piece = VOCAB[token]
        if text.endswith('"name": "'):
            found.append((j - 1, "name", text.count('"name": "'), token))
        elif "}" in piece and text.count("{") - text.count("}") == 1:
            found.append((j - 1, "close", text.count('"name": "'), token))
        text += piece
    return found


ALL_ITEMS = {i["id"]: i for d in vp.DIMENSIONS for i in vp.load(d)}
TARGET_ITEMS = [i for i in PRACTICE if i["task"] in SKELETON]
FIT_ITEMS = [i for i in TARGET_ITEMS if i["example"] < 50]      # 25 per task: the keys of the fit
READ_ITEMS = [i for i in TARGET_ITEMS if i["example"] >= 50]    # 25 per task: the check of the fit
# Other contexts, from the practice half only: the two tasks with no fixed procedure, and two examples of each short task.
OTHER_ITEMS = [i for i in PRACTICE if i["task"] not in SKELETON and i["example"] < 50] + [
    i for d in vp.DIMENSIONS if d != "Complex" for i in vp.split(vp.load(d))[0] if i["example"] < 4]
print(len(FIT_ITEMS), "fit items,", len(READ_ITEMS), "read items,", len(OTHER_ITEMS), "items of other contexts")

forced_fit = forced_plans(FIT_ITEMS); forced_read = forced_plans(READ_ITEMS); stamp("forced plans done")
for label, rows in [("forced plans, fit items", forced_fit), ("forced plans, read items", forced_read)]:
    for task, row in table(rows).items(): print(f"{label} | {task:32s} {row}", flush=True)
other_base = plan_trial("plan only, other contexts, base", OTHER_ITEMS, f"{RUN}/plan_other_practice.json"); stamp("other contexts done")
json.dump(forced_fit + forced_read, open(f"{RUN}/forced_plans.json", "w"))
r = forced_fit[0]; print([(p, k, s, VOCAB[t]) for p, k, s, t in step_positions(r["ids"])])

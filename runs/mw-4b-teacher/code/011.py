# Fault 2: at the first turn of push and window-open, the first move goes to the goal in place of the object.
PREFIX_MOVE = tok.encode('{"action": "move", "position": [', add_special_tokens=False)
print([tok.decode([t]) for t in PREFIX_MOVE], "|", [tok.decode([t]) for t in tok.encode('[-85, 883, 19]', add_special_tokens=False)])
def first_tokens(numbers):
    return tok.encode(f"[{numbers[0]},", add_special_tokens=False)[1:]
rows = []
for task in ("push-v3", "window-open-v3", "pick-place-v3"):
    convs = CTX1[task]
    for i, (conv, r) in enumerate(zip(convs, by_task[task])):
        text = conv[1]["content"][1]["text"]
        obj = json.loads(text.split("Object position: ")[1].split("\n")[0]); goal = json.loads(text.split("Goal position: ")[1].split("\n")[0])
        rows.append({"task": task, "seed": r["seed"], "obj": obj, "goal": goal, "first": r["primitives"][0].get("position"), "ok": r["correct"], "conv": conv})
for task in ("push-v3", "window-open-v3", "pick-place-v3"):
    sub = [x for x in rows if x["task"] == task]
    to_obj = sum(x["first"] is not None and abs(x["first"][0] - x["obj"][0]) + abs(x["first"][1] - x["obj"][1]) < 10 for x in sub)
    to_goal = sum(x["first"] is not None and abs(x["first"][0] - x["goal"][0]) + abs(x["first"][1] - x["goal"][1]) < 10 for x in sub)
    win_obj = sum(x["ok"] for x in sub if x["first"] is not None and abs(x["first"][0] - x["obj"][0]) + abs(x["first"][1] - x["obj"][1]) < 10)
    print(f"{task:16s} first move to the object: {to_obj} (success {win_obj}) | to the goal: {to_goal} | other: {len(sub) - to_obj - to_goal}")
# Probability of the first token of the object x and of the goal x, where the two differ.
for task in ("push-v3", "window-open-v3"):
    sub = [x for x in rows if x["task"] == task and first_tokens(x["obj"])[0] != first_tokens(x["goal"])[0]]
    lps = []
    for x in sub:
        o, g_ = first_tokens(x["obj"])[0], first_tokens(x["goal"])[0]
        lp = tools.read([x["conv"]], [], append_ids=[PREFIX_MOVE], logprobs_of=[o, g_])["logprobs"][0].exp()
        lps.append((float(lp[0]), float(lp[1]), x["ok"]))
    print(task, "| episodes where the first token differs:", len(sub), "| P(object token), P(goal token), success:", [(round(a, 2), round(b, 2), int(c)) for a, b, c in lps])

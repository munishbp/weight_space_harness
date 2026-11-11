tok = processor.tokenizer
for s in ['{"action": "close"}', '{"action": "move", "position": [1, 2, 3]}', '{"action": "open"}', '{"action": "done"}']:
    ids = tok.encode(s, add_special_tokens=False)
    print(s, ids, [tok.decode([i]) for i in ids])
PREFIX = tok.encode('{"action": "', add_special_tokens=False)
MOVE, CLOSE, OPEN, DONE = [tok.encode(w, add_special_tokens=False)[0] for w in ["move", "close", "open", "done"]]
print("prefix", PREFIX, "move close open done", MOVE, CLOSE, OPEN, DONE)


def step_conversation(task, seed, plan):
    """The prompt after a fixed plan of primitives: the step where the student picks the next action."""
    episode = mw.Episode(task, seed)
    for primitive in plan:
        episode.apply(primitive)
    conv = mw.conversation(episode)
    episode.close()
    return conv


def first_move(r):
    return [r["primitives"][0]]


LAST = parts["layers"][-1]
for task in TASKS:
    rows = [r for r in base if r["task"] == task and r["primitives"] and r["primitives"][0]["action"] == "move"][:10]
    convs = [step_conversation(task, r["seed"], first_move(r)) for r in rows]
    out = tools.read(convs, [LAST], append_ids=[PREFIX] * len(convs), logprobs_of=[MOVE, CLOSE, OPEN, DONE])
    p = out["logprobs"].exp()
    print(task, "step 2 mean p move/close/open/done", [round(float(x), 3) for x in p.mean(0)], "min p move", round(float(p[:, 0].min()), 3))

exec(open("runs/mw-4b-teacher/setup.py").read())
# Where does the model almost succeed? The choice of the action name at the second turn.
base = json.load(open(f"{RUN}/base_practice.json"))
by_task = defaultdict(list)
for r in base: by_task[r["task"]].append(r)

def context(result, steps):
    """The prompt after the first `steps` primitives that the base model wrote in this episode."""
    ep = mw.Episode(result["task"], result["seed"])
    for prim in result["primitives"][:steps]: ep.apply(prim)
    conv = mw.conversation(ep); ep.close()
    return conv

PREFIX = tok.encode('{"action": "', add_special_tokens=False)
print("prefix tokens:", [tok.decode([t]) for t in PREFIX])
for word in ("move", "close", "open", "done"):
    print(word, [tok.decode([t]) for t in tok.encode('{"action": "' + word + '"}', add_special_tokens=False)])
NAMES = {w: tok.encode('{"action": "' + w, add_special_tokens=False)[len(PREFIX)] for w in ("move", "close", "open", "done")}
print(NAMES)
CTX2 = {task: [context(r, 1) for r in rs] for task, rs in by_task.items()}
LP2 = {}
for task, convs in CTX2.items():
    out = tools.read(convs, [], append_ids=[PREFIX] * len(convs), logprobs_of=list(NAMES.values()))
    LP2[task] = out["logprobs"]
    p = out["logprobs"].exp()
    print(f"{task:26s} mean P at turn 2:", {w: round(float(p[:, i].mean()), 3) for i, w in enumerate(NAMES)}, "| min P(move)", round(float(p[:, 0].min()), 4), "| max P(move)", round(float(p[:, 0].max()), 3))

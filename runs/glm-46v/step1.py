# The button fault on GLM: where the model writes `close`, the probabilities there, and a forced replay.
import importlib
from wsp import habit, battery
BUTTON = "button-press-topdown-v3"
r0 = [r for r in base_practice if r["task"] == BUTTON][0]
first = tools.generate([mw.replay(r0, 0)], max_new_tokens=48)[0]
print("first reply:", repr(first["text"]), "| ids:", [tok.decode([t]) for t in first["ids"][:8]])
PREFIX = tok.encode('{"action": "', add_special_tokens=False)
NAMES = {w: tok.encode('{"action": "' + w, add_special_tokens=False)[len(PREFIX)] for w in ("move", "close", "open", "done")}
print("prefix", [tok.decode([t]) for t in PREFIX], "| names", {w: tok.decode([i]) for w, i in NAMES.items()}, "| prefix matches the reply:", first["ids"][:len(PREFIX)] == PREFIX)
def close_step(r):
    """The index of the first `close` in the plan of the model, or None."""
    names = [p.get("action") for p in r["primitives"]]
    return names.index("close") if "close" in names else None
button = [r for r in base_practice if r["task"] == BUTTON]
print("button episodes with a close:", sum(close_step(r) is not None for r in button), "| step of the close:", Counter(close_step(r) for r in button))
def probs(convs, batch=10):
    out = []
    for i in range(0, len(convs), batch):
        part = convs[i:i + batch]
        out.append(tools.read(part, [], append_ids=[PREFIX] * len(part), logprobs_of=list(NAMES.values()))["logprobs"].exp())
    return torch.cat(out)
CTX_B = [(r["seed"], mw.replay(r, close_step(r))) for r in button if close_step(r) is not None]
fmt = lambda q: {w: round(float(q[:, i].mean()), 2) for i, w in enumerate(NAMES)}
p = probs([c for _, c in CTX_B]); print("base, button at the close step:", fmt(p), "| move above 0.05 in", int((p[:, 0] > 0.05).sum()), "of", len(p)); stamp("probabilities done")
def gen_forced(conversations):
    """In the button task, a reply that starts with `close` is written again with `move` forced."""
    out = gen(conversations)
    again = [i for i, (c, r) in enumerate(zip(conversations, out)) if "Press the button" in c[1]["content"][1]["text"] and (mw.parse_primitive(r["text"]) or {}).get("action") == "close"]
    if again:
        start = PREFIX + [NAMES["move"]]
        for i, r in zip(again, tools.generate([conversations[i] for i in again], max_new_tokens=48, append_ids=[start] * len(again))): out[i] = r
    return out
forced = mw.run_episodes([(BUTTON, s) for s in range(20)], gen_forced, batch_size=20)
print("forced replay, button:", sum(r["correct"] for r in forced), "of 20 |", Counter(" ".join(p.get("action", "?") for p in r["primitives"]) for r in forced).most_common(3)); stamp("forced replay done")

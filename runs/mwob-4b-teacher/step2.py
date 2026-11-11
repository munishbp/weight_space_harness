# Replay with a forced action name: does a `click` at the two fault steps make the email episodes succeed?
def table(results):
    rows = {}
    for task in dict.fromkeys(r["task"] for r in results):
        rs = [r for r in results if r["task"] == task]
        rows[task] = {"n": len(rs), "success": sum(r["correct"] for r in rs), "invalid": sum(r["invalid"] for r in rs), "mean_actions": round(sum(len(r["actions"]) for r in rs) / len(rs), 1)}
    n, k = len(results), sum(r["correct"] for r in results)
    rows["all"] = {"n": n, "success": k, "success_pct": round(100 * k / n, 1), "interval_95": [round(v, 1) for v in wilson_interval(k, n)]}
    return rows
PREFIX = tok.encode('{"action": "', add_special_tokens=False)
NAMES = {w: tok.encode('{"action": "' + w, add_special_tokens=False)[len(PREFIX)] for w in ("click", "type", "key", "done")}
print("prefix", [tok.decode([t]) for t in PREFIX], "| names", {w: tok.decode([i]) for w, i in NAMES.items()})
def history_of(conv):
    text = conv[1]["content"][1]["text"]
    lines = text.split("Actions so far:\n")[1].split("\nWrite the next action.")[0].split("\n")
    return text.split("\n")[0], [] if lines == ["none"] else [json.loads(line) for line in lines]
def forced_name(conv):
    """The email rule: after the first click of a reply or forward task comes a click, and after a `type` comes a click."""
    task, history = history_of(conv)
    if not ("reply" in task or "forward" in task) or not history: return None
    if history[-1]["action"] == "type": return "click"
    if len(history) == 1 and history[0]["action"] == "click": return "click"
    return None
def gen_forced(conversations):
    names = [forced_name(c) for c in conversations]; out = [None] * len(conversations)
    free = [i for i, n in enumerate(names) if n is None]
    for i, r in zip(free, gen([conversations[i] for i in free]) if free else []): out[i] = r
    forced = [i for i, n in enumerate(names) if n is not None]
    if forced:
        start = PREFIX + [NAMES["click"]]
        inputs = tools.encode([conversations[i] for i in forced], False, append_ids=[start] * len(forced))
        new = model.generate(**inputs, max_new_tokens=40, do_sample=False)[:, inputs["input_ids"].shape[1]:]
        for i, text in zip(forced, processor.batch_decode(new, skip_special_tokens=True)): out[i] = {"text": '{"action": "click' + text}
    return out
t = time.time(); forced = mb.run_episodes(PRACTICE, gen_forced, batch_size=24)
for task, row in table(forced).items(): print(f"forced click | {task:26s} {row}")
for task in EMAIL:
    rows = [r for r in forced if r["task"] == task]
    print(task, "| ok:", Counter(acts(r) for r in rows if r["correct"]).most_common(3), "| failed:", Counter(acts(r) for r in rows if not r["correct"]).most_common(4))
stamp("forced replay done")

# Forced replay on all tasks: a `close` right after the first move is written again with `move` forced.
# The table shows which tasks have this habit as their fault.
def history_len(conv):
    text = conv[1]["content"][1]["text"]
    return text.count('{"action"')
def gen_forced_all(conversations):
    out = gen(conversations)
    again = [i for i, (c, r) in enumerate(zip(conversations, out)) if history_len(c) == 1 and (mw.parse_primitive(r["text"]) or {}).get("action") == "close"]
    if again:
        start = PREFIX + [NAMES["move"]]
        for i, r in zip(again, tools.generate([conversations[i] for i in again], max_new_tokens=48, append_ids=[start] * len(again))): out[i] = r
    return out
print("history length of a step-1 prompt:", history_len(mw.replay(button[0], 1)), "| text:", mw.replay(button[0], 1)[1]["content"][1]["text"][-160:].replace("\n", " / "))
forced_all = mw.run_episodes(PRACTICE, gen_forced_all, batch_size=16); json.dump(forced_all, open(f"{RUN}/forced_all_practice.json", "w"))
ref = {r["id"]: r["correct"] for r in base_practice}
for task in TASKS:
    rows = [r for r in forced_all if r["task"] == task]
    fixed = sum(r["correct"] and not ref[r["id"]] for r in rows); broken = sum(ref[r["id"]] and not r["correct"] for r in rows)
    print(f"forced | {task:26s} base {sum(ref[r['id']] for r in rows):2d} forced {sum(r['correct'] for r in rows):2d} fixed {fixed} broken {broken} |", Counter(" ".join(p.get("action", "?") for p in r["primitives"]) for r in rows).most_common(2), flush=True)
stamp("forced replay on all tasks done")

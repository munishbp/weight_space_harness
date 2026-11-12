# The prompt option that no OSWorld run tried: thinking on. The student thinks before each action, to a limit of
# 512 new tokens per step. The action is the text after the end of the thinking. A step with no end has no action.
THINK_END = tok.convert_tokens_to_ids("</think>")
THINK_LIMIT = 512
think_log = []


def gen_think(conversations):
    replies = tools.generate(conversations, max_new_tokens=THINK_LIMIT, thinking=True)
    for reply in replies:
        ids = reply["ids"]; closed = THINK_END in ids
        cut = ids.index(THINK_END) if closed else len(ids)
        think_log.append({"closed": closed, "thinking_tokens": cut, "hit_limit": reply["hit_limit"]})
        reply["text"] = tok.decode(ids[cut + 1:], skip_special_tokens=True) if closed else ""
    return replies


vs_practice = [t for t in TASKS.values() if t["domain"] == "vs_code" and t["half"] == "practice"]
think_practice = episodes("thinking vs_code practice", vs_practice, f"{RUN}/thinking_vscode_practice.json", generate=gen_think, sockets=SOCKETS)
compare("thinking on, practice", vs_practice, think_practice)
n = len(think_log); closed = [x for x in think_log if x["closed"]]
print(f"steps {n} | thinking closed {len(closed)} ({100 * len(closed) / n:.0f}%) | at the token limit {sum(x['hit_limit'] for x in think_log)} | median thinking tokens of closed steps {sorted(x['thinking_tokens'] for x in closed)[len(closed) // 2] if closed else None}")
json.dump(think_log, open(f"{RUN}/thinking_log.json", "w")); stamp("thinking on practice done")
for r in think_practice:
    print(r["id"][:8], "base", BASE[r["id"]]["correct"], "thinking", r["correct"], "invalid", r["invalid"], "|", " | ".join((a.get("action") + ":" + str(a.get("keys") or a.get("text") or a.get("position") or ""))[:40] for a in r["actions"][:8]))

# Forced replay, second rule: click the email, click Reply or Forward, click the text field, type, click Send.
def forced_name(conv):
    task, history = history_of(conv)
    if not ("reply" in task or "forward" in task) or not history: return None
    names = [a["action"] for a in history]
    if names[-1] == "type": return "click"
    if names in (["click"], ["click", "click"]): return "click"
    return None
forced2 = mb.run_episodes(PRACTICE, gen_forced, batch_size=24)
for task, row in table(forced2).items(): print(f"forced, rule 2 | {task:26s} {row}")
for task in EMAIL:
    rows = [r for r in forced2 if r["task"] == task]
    print(task, "| ok:", Counter(acts(r) for r in rows if r["correct"]).most_common(3), "| failed:", Counter(acts(r) for r in rows if not r["correct"]).most_common(4))
json.dump(forced2, open(f"{RUN}/forced2_practice.json", "w")); stamp("forced replay 2 done")

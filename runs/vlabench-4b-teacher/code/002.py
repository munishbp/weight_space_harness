# The base model on the practice half of the composite tasks, and what its wrong plans look like.
tools.log("teacher_start", note="VLABench VLM track, composite tasks, base run")
base_practice = trial("base practice", PRACTICE, f"{RUN}/base_practice.json"); stamp("base practice done")
print("row keys:", [k for k in base_practice[0] if k != "item"])
for task in sorted({r["task"] for r in base_practice}):
    rs = [r for r in base_practice if r["task"] == task]
    print("---", task, "| steps written:", Counter(r["steps"] for r in rs).most_common(4))
    for r in rs[:2]: print("    reply:", repr(r.get("text", r.get("reply", ""))[:420]))

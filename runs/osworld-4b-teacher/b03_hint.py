# The cheap option before an edit: when the student writes the action of the step before, tell it and ask again.
# The pass shows if a student that leaves the loop solves more tasks. It is the ceiling of an edit against the loop.
HINT = " The last action changed nothing. Write a different action."
hints = Counter()


def last_action(conversation):
    """The last action of the history in a prompt of `ow.conversation`, or None for an empty history."""
    text = conversation[1]["content"][1]["text"]
    lines = text.split("Actions so far:\n")[1].rsplit("\nWrite the next action.", 1)[0].splitlines()
    return None if lines == ["none"] else json.loads(lines[-1])


def gen_hint(conversations):
    replies = []
    for conversation in conversations:
        reply = gen([conversation])[0]
        if ow.parse_action(reply["text"]) == last_action(conversation):
            image, text = conversation[1]["content"]
            asked = [conversation[0], {"role": "user", "content": [image, {"type": "text", "text": text["text"] + HINT}]}]
            reply = gen([asked])[0]
            hints["asked"] += 1
            hints["still the same"] += ow.parse_action(reply["text"]) == last_action(conversation)
        replies.append(reply)
    return replies


sample = json.load(open(f"{RUN}/sample.json"))
failed_ids = [r["id"] for r in sample if not r["correct"]]
by_id = {t["id"]: t for d in {r["domain"] for r in sample} for t in ow.tasks(d)}
hint = episodes("hint", [by_id[i] for i in failed_ids], f"{RUN}/hint.json", generate=gen_hint, sockets=SOCKETS)
print("hints", dict(hints))
print("failed base tasks that the hint pass solves:", sum(r["correct"] for r in hint), "of", len(hint))
stamp("hint pass done")

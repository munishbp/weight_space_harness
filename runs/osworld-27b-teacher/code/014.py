# The cheap option on the same held-out tasks: the base weights with one sentence in the prompt.
# The sentence is the sentence of the 4B run. Run after b05b_verdict.py, with the virtual machines up.
NOTE = " Some tasks are not possible on this machine or in this application. If this task is not possible, write {\"action\": \"fail\"}."
tools.reset_edit()


def gen_note(conversations):
    """The base student with the sentence after the task text."""
    noted = []
    for conversation in conversations:
        image, text = conversation[1]["content"]
        head, tail = text["text"].split("\nActions so far:\n", 1)
        noted.append([conversation[0], {"role": "user", "content": [image, {"type": "text", "text": head + NOTE + "\nActions so far:\n" + tail}]}])
    return gen(noted)


note_rows = episodes("note heldout", held_hard + held_easy, f"{RUN}/note_heldout.json", generate=gen_note, sockets=SOCKETS)
out["prompt_note/not_possible"] = compare("prompt sentence, not possible", held_hard, note_rows)
out["prompt_note/possible"] = compare("prompt sentence, possible", held_easy, note_rows)
json.dump(out, open(f"{RUN}/verdict.json", "w"), indent=1); stamp("prompt sentence done")

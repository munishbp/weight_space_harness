# Two controls on the held-out tasks: the same fit with a random direction, and the base weights with a note in the prompt.
tools.reset_edit(); torch.manual_seed(0)
g_rand = torch.randn_like(g_fail); g_rand = g_rand * g_fail.norm() / g_rand.norm()
habit.fit(MLP_OUT, [(keys_all, scale_all, g_rand)], keep_all, protect, margin=MARGIN)
rand_rows = episodes("random heldout", held_hard, f"{RUN}/random_heldout.json", sockets=SOCKETS)
out["random_control"] = {"n": len(rand_rows), "solved": sum(r["correct"] for r in rand_rows), "ends_with_fail": sum(r["actions"][-1].get("action") == "fail" for r in rand_rows)}
print("random control:", out["random_control"], flush=True)
tools.reset_edit()
out["reset_restores_weight"] = bool(torch.equal(tools.get_weight(MLP_OUT + ".weight"), ORIGINAL_W)); print("reset restores the weight:", out["reset_restores_weight"])
json.dump(out, open(f"{RUN}/verdict.json", "w"), indent=1); stamp("random control done")


def gen_note(conversations):
    """The base student with the note about tasks that are not possible, after the task text."""
    noted = []
    for conversation in conversations:
        image, text = conversation[1]["content"]
        head, tail = text["text"].split("\nActions so far:\n", 1)
        noted.append([conversation[0], {"role": "user", "content": [image, {"type": "text", "text": head + NOTE + "\nActions so far:\n" + tail}]}])
    return gen(noted)


note_rows = episodes("note heldout", held_hard + held_easy, f"{RUN}/note_heldout.json", generate=gen_note, sockets=SOCKETS)
for name, tasks_ in (("not_possible", held_hard), ("possible", held_easy)):
    ids = [t["id"] for t in tasks_]; rows = {r["id"]: r for r in note_rows}
    before, after = [base_by_id[i]["correct"] for i in ids], [rows[i]["correct"] for i in ids]
    out["prompt_note/" + name] = {"n": len(ids), "base_solved": sum(before), "note_solved": sum(after), "flips": paired_flips(before, after),
                                   "ends_with_fail": sum(rows[i]["actions"][-1].get("action") == "fail" for i in ids)}
    print("prompt note", name, out["prompt_note/" + name], flush=True)
json.dump(out, open(f"{RUN}/verdict.json", "w"), indent=1); stamp("prompt note done")

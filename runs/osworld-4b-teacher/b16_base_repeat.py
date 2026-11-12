# The noise of one run: the base student again on the same 12 VS Code practice tasks, with no change.
repeat = episodes("base repeat vs_code practice", vs_practice, f"{RUN}/base_repeat_vscode_practice.json", sockets=SOCKETS)
compare("base again, practice", vs_practice, repeat)
print("same actions as the first base run:", sum(r["actions"] == BASE[r["id"]]["actions"] for r in repeat), "of", len(repeat)); stamp("base repeat done")

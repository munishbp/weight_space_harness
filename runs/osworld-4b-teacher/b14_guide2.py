# Guide 2. The first guide fixed the one task where it gave an exact key recipe (replace all), and it broke two
# tasks that the base solved: the student pressed ctrl+s and followed lines that did not fit the task.
# Guide 2 holds exact key recipes only, each with the kind of task that it is for. No chord, no visual step.
GUIDE2 = """Guide for Visual Studio Code. Use a line only when the task is of that kind. For every other task, work as usual.
- Task: change a setting. Press ctrl+shift+p, type "Preferences: Open User Settings (JSON)", press enter, press ctrl+a, type the settings as one JSON object, for example {"editor.fontSize": 14}, and press ctrl+s.
- Task: replace text in the open file. Press ctrl+h, type the text to find, press tab, type the new text, press ctrl+alt+enter, and press ctrl+s.
- Task: install an extension. Press ctrl+shift+x, type the name of the extension, and click the Install button of the first result. Then wait. Do not press ctrl+s.
- Task: add a keyboard shortcut. Press ctrl+shift+p, type "Preferences: Open Keyboard Shortcuts (JSON)", press enter, press ctrl+a, type the shortcuts as one JSON list, for example [{"key": "ctrl+j", "command": "workbench.action.files.save", "when": "editorTextFocus"}], and press ctrl+s.
Press ctrl+s only after you change a file. When the task is complete, write done."""
guide2_practice = episodes("guide2 vs_code practice", vs_practice, f"{RUN}/guide2_vscode_practice.json", generate=with_system(GUIDE2), sockets=SOCKETS)
compare("guide 2 in the prompt, practice", vs_practice, guide2_practice); stamp("guide 2 on practice done")
for r in guide2_practice:
    print(r["id"][:8], "base", BASE[r["id"]]["correct"], "guide2", r["correct"], "|", " | ".join((a.get("action") + ":" + str(a.get("keys") or a.get("text") or a.get("position") or ""))[:46] for a in r["actions"][:9]))

# The cheap test for possible tasks: the teacher writes what it knows about the application as a guide in the
# system prompt. Does the base student solve more practice tasks with it? The guide holds general procedures
# of VS Code. The teacher wrote it from its own knowledge and from the practice tasks. It saw no held-out task.
GUIDE_VSCODE = """Guide for Visual Studio Code:
- To change a setting: press ctrl+, to open Settings. Type the name of the setting in the search box. Then use the control of that setting in the list below the search box: click a checkbox to switch it, or click a dropdown and click the option, or click a number or text field, press ctrl+a, type the value, and press tab. A setting saves itself. Do not press ctrl+s for a setting.
- To add an item to a list setting (for example a pattern): click the button "Add Pattern" or "Add Item" of that setting, type the item, and press enter.
- To replace text in the open file: press ctrl+h, type the text to find, press tab, type the new text, press ctrl+alt+enter to replace all, and press ctrl+s.
- To run a command: press ctrl+shift+p, type the name of the command, and press enter.
- To add a keyboard shortcut: press ctrl+k and then ctrl+s to open Keyboard Shortcuts, type the name of the command, double click the command, press the keys of the shortcut, and press enter.
- To add a folder to the workspace: run the command "Workspaces: Add Folder to Workspace", type the full path of the folder, and press enter.
- Type each text one time. When the screen shows that the task is complete, write done."""
vs_practice = [t for t in TASKS.values() if t["domain"] == "vs_code" and t["half"] == "practice"]
guide_practice = episodes("guide vs_code practice", vs_practice, f"{RUN}/guide_vscode_practice.json", generate=with_system(GUIDE_VSCODE), sockets=SOCKETS)
compare("guide in the prompt, practice", vs_practice, guide_practice); stamp("guide on practice done")

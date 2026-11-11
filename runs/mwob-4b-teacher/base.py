# The base model on a scan of all candidate task types, 4 seeds each.
tools.log("teacher_start", note="MiniWoB++ through screenshots: base scan")
scan = episodes("base scan", [(task, seed) for task in TASKS for seed in range(4)], f"{RUN}/base_scan.json")

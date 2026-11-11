# The first two tasks, to check the whole path and to measure the time per task.
small = json.load(open("data/OSWorld/evaluation_examples/test_small.json"))
os_tasks = {t["id"]: t for t in ow.tasks("os")}
first = episodes("first", [os_tasks[i] for i in small["os"]], f"{RUN}/first.json")
for r in first:
    print("---", r["instruction"][:140]); print("    replies:", [x[:90] for x in r["replies"][:6]])
stamp("first two tasks done")

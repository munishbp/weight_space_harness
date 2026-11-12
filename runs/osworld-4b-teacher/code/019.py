# The base student on every task that is not possible by design and that the scan did not run: 19 tasks.
SOCKETS = ("/tmp/wsp-osworld.sock", "/tmp/wsp-osworld2.sock", "/tmp/wsp-osworld3.sock")
seen = {r["id"] for r in scan}
todo = [TASKS[i] for i in sorted(INFEASIBLE, key=lambda i: (TASKS[i]["domain"], i)) if i not in seen]
infeasible_base = episodes("infeasible", todo, f"{RUN}/infeasible_base.json", sockets=SOCKETS)
print(Counter((r["domain"], TASKS[r["id"]]["half"], r["correct"]) for r in infeasible_base))
stamp("base on the other infeasible tasks done")

# The base 27B student on three more sets, in one pass:
# 1. the practice tasks that are not possible and that the scan did not run (their steps are the keys of the fit),
# 2. the held-out tasks that are not possible (11),
# 3. the possible held-out tasks of Thunderbird and VS Code (17), for the verdict.
scanned = {r["id"] for r in json.load(open(f"{RUN}/scan_practice.json"))}
hard_practice = [TASKS[i] for i in sorted(INFEASIBLE) if TASKS[i]["half"] == "practice" and i not in scanned]
held_hard = [TASKS[i] for i in sorted(INFEASIBLE) if TASKS[i]["half"] == "heldout"]
held_easy = [t for t in TASKS.values() if t["domain"] in ("thunderbird", "vs_code") and t["half"] == "heldout" and t["id"] not in INFEASIBLE]
print(len(hard_practice), "practice tasks that are not possible,", len(held_hard), "held-out tasks that are not possible,", len(held_easy), "possible held-out tasks")
more = episodes("more base", hard_practice + held_hard + held_easy, f"{RUN}/more_base.json", sockets=SOCKETS)
by = {r["id"]: r for r in more}
summary("27B base, practice, not in the scan", [by[t["id"]] for t in hard_practice])
summary("27B base, held-out", [by[t["id"]] for t in held_hard + held_easy]); stamp("more base done")

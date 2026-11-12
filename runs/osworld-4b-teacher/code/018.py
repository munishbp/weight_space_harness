# The cheap option for the `fail` fault: a sentence in the prompt. Does the student then give `fail` to the
# tasks that are not possible, and not to the others? Read at the first step and at the last recorded step.
NOTE = " Some tasks are not possible on this machine or in this application. If this task is not possible, write {\"action\": \"fail\"}."
rows = []
for r in [r for r in scan if TASKS[r["id"]]["half"] == "practice" and "error" not in r]:
    for s in (0, len(r["actions"]) - 1):
        conv = ow.conversation(r["instruction"] + NOTE, shot(r["id"], s, "scan"), r["actions"][:s])
        p = tools.read([conv], [LAST], append_ids=[PREFIX], logprobs_of=list(FIRST.values()))["logprobs"][0].exp().float().cpu()
        rows.append((r["id"] in INFEASIBLE, s == 0, float(p[NAMES.index("fail")]), r["id"]))
for kind in (True, False):
    for first in (True, False):
        vals = sorted((v for k, f, v, _ in rows if k == kind and f == first), reverse=True)
        print(f"with the note | {'infeasible' if kind else 'feasible  '} | {'first step' if first else 'last step '} n={len(vals):2d} | p(fail) median {vals[len(vals) // 2]:.4f} | above 0.5: {sum(v > 0.5 for v in vals)} | top {[round(v, 3) for v in vals[:6]]}")
json.dump(rows, open(f"{RUN}/fail_prompt_probs.json", "w")); stamp("fail prompt read done")

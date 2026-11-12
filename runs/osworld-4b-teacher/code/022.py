# The verdict for the `fail` edit (layer 27, margin 16): the held-out tasks in the virtual machine.
MARGIN = 16.0
all_hard, all_easy = prompts(hard), prompts(easy)
keys_all, scale_all = keys_of(all_hard); keep_all, _ = keys_of(all_easy)
print(len(all_hard), "target steps of", len(hard), "tasks;", len(all_easy), "keep steps of", len(easy), "tasks")
tools.reset_edit(); habit.fit(MLP_OUT, [(keys_all, scale_all, g_fail)], keep_all, protect, margin=MARGIN); stamp("final fit done")

base_by_id = {r["id"]: r for r in scan + infeasible_base}
held_hard = [TASKS[i] for i in sorted(INFEASIBLE) if TASKS[i]["half"] == "heldout"]
held_easy = [TASKS[r["id"]] for r in scan if TASKS[r["id"]]["half"] == "heldout" and r["id"] not in INFEASIBLE]
print(len(held_hard), "held-out tasks that are not possible,", len(held_easy), "possible held-out tasks")
edit_rows = episodes("edit heldout", held_hard + held_easy, f"{RUN}/edit_heldout.json", sockets=SOCKETS)
out = {"margin": MARGIN, "matrix": MLP_OUT}
for name, tasks_ in (("not_possible", held_hard), ("possible", held_easy)):
    ids = [t["id"] for t in tasks_]; rows = {r["id"]: r for r in edit_rows}
    before, after = [base_by_id[i]["correct"] for i in ids], [rows[i]["correct"] for i in ids]
    out[name] = {"n": len(ids), "base_solved": sum(before), "edit_solved": sum(after), "flips": paired_flips(before, after),
                 "edit_ends_with_fail": sum(rows[i]["actions"][-1].get("action") == "fail" for i in ids),
                 "bridge_errors": sum("error" in rows[i] for i in ids)}
    print(name, out[name], flush=True)
json.dump(out, open(f"{RUN}/verdict.json", "w"), indent=1); stamp("held-out tasks done")
out["deviation"] = tools.deviation(NEUTRAL); print("deviation:", {k: round(v, 5) for k, v in out["deviation"].items()}, flush=True)
manifest = json.load(open(f"{RUN}/edit/manifest.json")); out["edit"] = {k: v for k, v in manifest.items() if k != "tensors"}
out["edit"]["tensor"] = {k: v for k, v in manifest["tensors"][0].items() if not isinstance(v, (list, dict))}; print("edit:", out["edit"], flush=True)
out["battery"] = battery.compare(json.load(open("runs/ss-pro-4b-teacher/battery_base.json")), battery.run())
for k, v in out["battery"].items(): print("battery", k, v, flush=True)
json.dump(out, open(f"{RUN}/verdict.json", "w"), indent=1); stamp("battery done")

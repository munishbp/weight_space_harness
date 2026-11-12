# Part 2 of the verdict, with the virtual machines up: the held-out tasks with the teacher edit, the deviation,
# the short battery, and the held-out tasks that are not possible with the random control. Run after b05a_prepare.py.
out = {"margin": MARGIN, "matrix": MLP_OUT, "clamped_weights": over_edit}
tools.reset_edit(); entry = tools.save_edit(MLP_OUT + ".weight", W_EDIT)
manifest = json.load(open(f"{RUN}/edit/manifest.json")); out["edit"] = {k: v for k, v in manifest.items() if k != "tensors"}
out["edit"]["tensor"] = {k: v for k, v in manifest["tensors"][0].items() if not isinstance(v, (list, dict))}; print("edit:", out["edit"], flush=True)

base_by_id = {r["id"]: r for r in more_rows}
held_hard = [TASKS[i] for i in sorted(INFEASIBLE) if TASKS[i]["half"] == "heldout"]
held_easy = [t for t in TASKS.values() if t["domain"] in ("thunderbird", "vs_code") and t["half"] == "heldout" and t["id"] not in INFEASIBLE]
edit_rows = episodes("edit heldout", held_hard + held_easy, f"{RUN}/edit_heldout.json", sockets=SOCKETS)
out["not_possible"] = compare("edit, not possible", held_hard, edit_rows); out["possible"] = compare("edit, possible", held_easy, edit_rows)
json.dump(out, open(f"{RUN}/verdict.json", "w"), indent=1); stamp("held-out tasks done")
out["deviation"] = deviation_in_parts(NEUTRAL); print("deviation:", {k: round(v, 5) for k, v in out["deviation"].items()}, flush=True)
battery_edit = short_battery()
out["battery"] = {name: {"before_pct": float(sum(battery_base[name])), "after_pct": float(sum(battery_edit[name])), **paired_flips(battery_base[name], battery_edit[name])} for name in battery_base}
for k, v in out["battery"].items(): print("battery", k, v, flush=True)
json.dump(out, open(f"{RUN}/verdict.json", "w"), indent=1); stamp("battery done")

tools.reset_edit(); control = tools.save_edit(MLP_OUT + ".weight", W_RAND)
rand_rows = episodes("random heldout", held_hard, f"{RUN}/random_heldout.json", sockets=SOCKETS)
out["random_control"] = compare("random control, not possible", held_hard, rand_rows) | {"relative_change": control["relative_change"], "clamped_weights": over_rand}
tools.reset_edit(); state1 = model.get_submodule(MLP_OUT).state()
raw = lambda x: x.contiguous().reshape(-1).view(torch.uint8)
out["reset_restores_stored_bytes"] = all(torch.equal(raw(state0[k]), raw(state1[k])) for k in state0)
print("reset restores the stored bytes:", out["reset_restores_stored_bytes"])
json.dump(out, open(f"{RUN}/verdict.json", "w"), indent=1); stamp("verdict done")

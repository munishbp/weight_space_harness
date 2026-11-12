# The verdict for the stacked procedure edit (layer 27, margin 12) on the held-out half.
MARGIN = 12.0
HELD_TARGET = [i for i in HELDOUT if i["task"] in SKELETON]          # 150 items, 50 per task
HELD_OTHER = [i for i in HELDOUT if i["task"] not in SKELETON]       # 100 items: cook_dishes and texas_holdem
NEAR2 = NEAR[::2]                                                    # 200 items of the five short dimensions
PROSE = [i for i in HELD_TARGET if i["example"] < 50]                # 75 items for the official mode with prose
MEASURES = ["skill_match_score", "entity_match_score", "skill_with_entity_match_score", "exact_match_score", "total_score"]
out = {"margin": MARGIN, "matrix": MLP_OUT, "sets": {"held_target": len(HELD_TARGET), "held_other": len(HELD_OTHER), "near": len(NEAR2), "prose": len(PROSE)}}


def means(rows):
    return {"n": len(rows), "correct": sum(r["correct"] for r in rows), "parsed": sum(r["parsed"] for r in rows),
            "names_all_correct": sum(r["skill_match_score"] == 1.0 for r in rows), **{m: round(sum(r[m] for r in rows) / len(rows), 4) for m in MEASURES}}


def compare(label, before, after):
    row = {"base": means(before), "edit": means(after),
           "names_all_correct_flips": paired_flips([r["skill_match_score"] == 1.0 for r in before], [r["skill_match_score"] == 1.0 for r in after]),
           "correct_flips": paired_flips([r["correct"] for r in before], [r["correct"] for r in after]),
           "total_score_up_down": [sum(a["total_score"] > b["total_score"] + 1e-9 for b, a in zip(before, after)), sum(a["total_score"] < b["total_score"] - 1e-9 for b, a in zip(before, after))],
           "plans_changed": sum(vp.parse(a["text"]) != vp.parse(b["text"]) for b, a in zip(before, after))}
    print(label, json.dumps(row), flush=True); out[label] = row
    for task in sorted({r["task"] for r in before}) if label == "held_target" or label.startswith("prose") or label == "random_control" else []:
        b, a = [r for r in before if r["task"] == task], [r for r in after if r["task"] == task]
        out[f"{label}/{task}"] = {"base": means(b), "edit": means(a)}; print("  ", task, json.dumps(out[f"{label}/{task}"]), flush=True)
    json.dump(out, open(f"{RUN}/verdict.json", "w"), indent=1)


tools.reset_edit()
base_sets = {name: plan_trial(f"verdict base {name}", items, f"{RUN}/verdict_base_{name}.json") for name, items in [("held_target", HELD_TARGET), ("held_other", HELD_OTHER), ("near", NEAR2)]}
base_prose = trial("verdict base prose", PROSE, f"{RUN}/verdict_base_prose.json"); stamp("base passes done")

habit.fit(MLP_OUT, targets, keep, protect, margin=MARGIN)
for name, items in [("held_target", HELD_TARGET), ("held_other", HELD_OTHER), ("near", NEAR2)]:
    compare(name, base_sets[name], plan_trial(f"verdict edit {name}", items, f"{RUN}/verdict_edit_{name}.json"))
compare("prose_mode", base_prose, trial("verdict edit prose", PROSE, f"{RUN}/verdict_edit_prose.json")); stamp("edit passes done")
out["deviation"] = tools.deviation(NEUTRAL); print("deviation:", {k: round(v, 5) for k, v in out["deviation"].items()}, flush=True)
manifest = json.load(open(f"{RUN}/edit/manifest.json")); out["edit"] = {k: v for k, v in manifest.items() if k != "tensors"}
out["edit"]["tensor"] = {k: v for k, v in manifest["tensors"][0].items() if not isinstance(v, (list, dict))}; print("edit:", out["edit"], flush=True)
after = battery.run(); out["battery"] = battery.compare(json.load(open("runs/ss-pro-4b-teacher/battery_base.json")), after)
for k, v in out["battery"].items(): print("battery", k, v, flush=True)
json.dump(out, open(f"{RUN}/verdict.json", "w"), indent=1); stamp("battery done")

# The random control: the same keys and the same fit, with one random direction of the same length per target group.
tools.reset_edit(); torch.manual_seed(0)
random_targets = [(k, s, torch.randn_like(d) * d.norm() / torch.randn_like(d).norm()) for k, s, d in targets]
random_targets = [(k, s, r * d.norm() / r.norm()) for (k, s, d), (_, _, r) in zip(targets, random_targets)]
habit.fit(MLP_OUT, random_targets, keep, protect, margin=MARGIN)
compare("random_control", [r for r in base_sets["held_target"] if ALL_ITEMS[r["id"]]["example"] < 50], plan_trial("verdict random control", PROSE, f"{RUN}/verdict_random.json"))
tools.reset_edit()
out["reset_restores_weight"] = bool(torch.equal(tools.get_weight(MLP_OUT + ".weight"), ORIGINAL_W)); print("reset restores the weight:", out["reset_restores_weight"])
json.dump(out, open(f"{RUN}/verdict.json", "w"), indent=1); stamp("verdict done")

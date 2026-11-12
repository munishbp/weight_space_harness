# The smaller edit (margin 8) on the held-out target tasks, with the battery: the trade between the gain and the general ability.
tools.reset_edit(); habit.fit(MLP_OUT, targets, keep, protect, margin=8.0)
m8 = {"held_target": None}
rows = plan_trial("margin 8 held_target", HELD_TARGET, f"{RUN}/verdict_m8_held_target.json")
m8["held_target"] = {"base": means(base_sets["held_target"]), "edit": means(rows),
                     "names_all_correct_flips": paired_flips([r["skill_match_score"] == 1.0 for r in base_sets["held_target"]], [r["skill_match_score"] == 1.0 for r in rows])}
for task in sorted(SKELETON): m8[task] = means([r for r in rows if r["task"] == task])
m8["deviation"] = tools.deviation(NEUTRAL)
manifest = json.load(open(f"{RUN}/edit/manifest.json")); m8["relative_change"] = manifest["tensors"][0]["relative_change"]; m8["relative_change_of_model"] = manifest["relative_change_of_model"]
m8["battery"] = battery.compare(json.load(open("runs/ss-pro-4b-teacher/battery_base.json")), battery.run())
tools.reset_edit(); json.dump(m8, open(f"{RUN}/verdict_margin8.json", "w"), indent=1)
print(json.dumps(m8, indent=1)); stamp("margin 8 done")

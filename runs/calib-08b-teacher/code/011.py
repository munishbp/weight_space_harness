A23 = L + "23.self_attn.o_proj.weight"
g8 = trial_names("g out of MLP 21, 22, att 18, alpha 2", g, [M21, M22, A18], alpha=2.0)
g9 = trial_names("g out of MLP 21, 22, att 18, att 23, alpha 1", g, [M21, M22, A18, A23], alpha=1.0)
g10 = trial_names("g out of MLP 21, 22, att 18, att 23, alpha 2", g, [M21, M22, A18, A23], alpha=2.0)
json.dump({"g8": g8, "g9": g9, "g10": g10}, open("runs/calib-08b-teacher/trials_g2.json", "w"))
led = [json.loads(l) for l in open("runs/calib-08b-teacher/ledger.jsonl")]
t0 = next(x["t"] for x in led if x["kind"] == "teacher_start")
print("teacher time so far:", round((time.time() - t0) / 60), "min | generate passes:", sum(x["kind"] == "generate" for x in led), "| read passes:", sum(x["kind"] == "read" for x in led), "| code runs:", sum(x["kind"] == "python" for x in led))

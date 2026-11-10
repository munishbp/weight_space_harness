import shutil
from wsp.stats import paired_flips
held = rows[100:228]
old = json.load(open("runs/calib-08b-reproduce/math_gsm8k_128.json"))
assert [r["id"] for r in old["base_rows"]] == [p["id"] for p in held], "the held-out items must be the items of ticket 04"
def measure(label, **over):
    t = time.time()
    opts = {**boxed_math.BLOG_SAMPLING, **over}
    res = evaluate.run(held, boxed_math.conversation, boxed_math.score, batch_size=64, max_new_tokens=4096, **opts)
    s = {"model": label, "seconds": round(time.time() - t), **evaluate.summarize(res)["all"],
         "closed_thinking_pct": round(100 * sum(THINK_END in r["ids"] for r in res) / len(res), 1), "median_think_len": think_len(res)}
    s["vs_base"] = paired_flips([r["correct"] for r in old["base_rows"]], [r["correct"] for r in res])
    s["vs_published"] = paired_flips([r["correct"] for r in old["edited_rows"]], [r["correct"] for r in res])
    print(s, flush=True); return {"summary": s, "rows": res}
results = {}
project_out(g_unit, [M21, M22, A18], 1.75)
shutil.copy("runs/calib-08b-teacher/edit/manifest.json", "runs/calib-08b-teacher/manifest_veto_1.75.json")
shutil.copy("runs/calib-08b-teacher/edit/edit.safetensors", "runs/calib-08b-teacher/edit_veto_1.75.safetensors")
results["veto_1.75_think"] = measure("teacher edit A: veto removal, strength 1.75, thinking on")
results["veto_1.75_nothink"] = measure("teacher edit A, thinking off", thinking=False)
tools.reset_edit()
wire(14)
shutil.copy("runs/calib-08b-teacher/edit/manifest.json", "runs/calib-08b-teacher/manifest_wire_14.json")
shutil.copy("runs/calib-08b-teacher/edit/edit.safetensors", "runs/calib-08b-teacher/edit_wire_14.safetensors")
results["wire_14_think"] = measure("teacher edit B: wiring, +14 nats, thinking on")
tools.reset_edit()
results["base_nothink"] = measure("base, thinking off", thinking=False)
json.dump(results, open("runs/calib-08b-teacher/heldout_128.json", "w"))
tools.log("teacher_end", note="held-out evaluation done")
print("saved")

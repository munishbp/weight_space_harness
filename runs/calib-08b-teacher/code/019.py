gw_unit = gw / gw.norm()
print("cosine between the Wait direction and the closing direction:", round(float(gw_unit @ g_unit), 3))
WAITMODS = [L + "16.mlp.down_proj.weight", L + "23.mlp.down_proj.weight", L + "21.mlp.down_proj.weight",
            L + "21.linear_attn.out_proj.weight", L + "22.linear_attn.out_proj.weight", L + "20.linear_attn.out_proj.weight"]
def run_trial(label):
    t = time.time(); m = json.load(open("runs/calib-08b-teacher/edit/manifest.json"))
    gen = evaluate.run(practice, boxed_math.conversation, boxed_math.score, batch_size=64, max_new_tokens=2048, **boxed_math.BLOG_SAMPLING)
    out = {"trial": label, **behaviour(gen), "median_think_len": think_len(gen), "weights_changed_pct": round(m["changed_weights_pct"], 4), "model_rel_change": round(m["relative_change_of_model"], 5), "seconds": round(time.time() - t)}
    tools.log("trial", **out); print(out, flush=True); return gen
project_out(gw_unit, WAITMODS, 1.0)
x1 = run_trial("remove the Wait direction from six components"); tools.reset_edit()
starts = Counter(w for r in x1 for line in tok.decode(r["ids"]).split("\n") for w in [line.strip(" *-").split(" ")[0][:14]] if w)
print("   line starts now:", starts.most_common(10))
# Combined: wiring at +14 nats first, then the Wait removal on top of it (two of the tensors are shared).
wire(14); project_out(gw_unit, WAITMODS, 1.0)
x2 = run_trial("wire +14 nats, plus remove the Wait direction"); tools.reset_edit()
json.dump({"x1": x1, "x2": x2}, open("runs/calib-08b-teacher/trials_wait.json", "w"))
led = [json.loads(l) for l in open("runs/calib-08b-teacher/ledger.jsonl")]
t0 = next(x["t"] for x in led if x["kind"] == "teacher_start")
print("teacher time so far:", round((time.time() - t0) / 60), "min | generate passes:", sum(x["kind"] == "generate" for x in led), "| read passes:", sum(x["kind"] == "read" for x in led), "| trials:", sum(x["kind"] == "trial" for x in led))

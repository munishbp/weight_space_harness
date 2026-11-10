def trial_names(label, u, names, alpha=1.0, keep=False):
    t = time.time()
    u = u / u.norm()
    project_out(u, names, alpha)
    m = json.load(open("runs/calib-08b-teacher/edit/manifest.json"))
    gen = evaluate.run(practice, boxed_math.conversation, boxed_math.score, batch_size=64, max_new_tokens=2048, **boxed_math.BLOG_SAMPLING)
    out = {"trial": label, **behaviour(gen), "weights_changed_pct": round(m["changed_weights_pct"], 3), "model_rel_change": round(m["relative_change_of_model"], 5), "seconds": round(time.time() - t)}
    tools.log("trial", **out)
    if not keep: tools.reset_edit()
    print(out, flush=True)
    return gen
L = "model.language_model.layers."
M21, M22, A18 = L + "21.mlp.down_proj.weight", L + "22.mlp.down_proj.weight", L + "18.linear_attn.out_proj.weight"
g5 = trial_names("g out of MLP 21 and 22", g, [M21, M22])
g6 = trial_names("g out of MLP 21, 22 and attention 18", g, [M21, M22, A18])
g7 = trial_names("g_abs out of MLP 21, 22 and attention 18", g_abs, [M21, M22, A18])
json.dump({"g5": g5, "g6": g6, "g7": g7}, open("runs/calib-08b-teacher/trials_g.json", "w"))

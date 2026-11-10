import time
print("cosine of the repeat direction between layers 13 and [8, 11, 16, 19, 22]:", [round(float(rdirs[13] @ rdirs[j]), 2) for j in (8, 11, 16, 19, 22)])

def behaviour(gen):
    n = len(gen)
    closed = [THINK_END in r["ids"] for r in gen]
    toks = sum(len(r["ids"]) for r in gen)
    rep = sum(len(repeat_classes(r["ids"], 0)[0]) for r in gen)
    return {"finished_pct": round(100 * sum(c and not r["hit_limit"] for c, r in zip(closed, gen)) / n, 1),
            "correct_pct": round(100 * sum(r["correct"] for r in gen) / n, 1),
            "closed_thinking_pct": round(100 * sum(closed) / n, 1),
            "wait_per_1000": round(1000 * sum(t in WAIT for r in gen for t in r["ids"]) / toks, 2),
            "repeat_share": round(rep / toks, 3), "mean_tokens": round(toks / n)}

def writers(layers, attention):
    names = [f"model.language_model.layers.{i}.mlp.down_proj.weight" for i in layers]
    if attention:
        all_names = {n for n, _ in model.named_parameters()}
        for i in layers:
            names += [n for n in (f"model.language_model.layers.{i}.self_attn.o_proj.weight", f"model.language_model.layers.{i}.linear_attn.out_proj.weight") if n in all_names]
    return names

def project_out(u, names, alpha):
    """W' = W - alpha * u (u^T W): each named module stops writing along u."""
    for name in names:
        W = tools.get_weight(name)
        tools.save_edit(name, W - alpha * torch.outer(u, u @ W))

def trial(label, u, layers, alpha, attention=False, keep=False):
    t = time.time()
    project_out(u, writers(layers, attention), alpha)
    m = json.load(open("runs/calib-08b-teacher/edit/manifest.json"))
    gen = evaluate.run(practice, boxed_math.conversation, boxed_math.score, batch_size=64, max_new_tokens=2048, **boxed_math.BLOG_SAMPLING)
    out = {"trial": label, **behaviour(gen), "weights_changed_pct": round(m["changed_weights_pct"], 3), "model_rel_change": round(m["relative_change_of_model"], 4), "seconds": round(time.time() - t)}
    tools.log("trial", **out)
    if not keep: tools.reset_edit()
    print(out, flush=True)
    return gen

print({"trial": "base", **behaviour(base_gen)})
MID = list(range(8, 23))
g1 = trial("repeat dir (layer 13), MLPs 8-22, alpha 1", rdirs[13], MID, 1.0)
g2 = trial("repeat dir (layer 13), MLPs + attention 8-22, alpha 1.5", rdirs[13], MID, 1.5, attention=True)
g3 = trial("Wait dir (layer 16), MLPs 8-22, alpha 1", dirs[16], MID, 1.0)
torch.manual_seed(7); rand = torch.nn.functional.normalize(torch.randn(1024), dim=0)
g4 = trial("random dir, MLPs 8-22, alpha 1", rand, MID, 1.0)

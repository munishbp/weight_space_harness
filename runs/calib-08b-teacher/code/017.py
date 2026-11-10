g_unit = g / g.norm()
rms_final = float(rms.mean())
WIRE = (21, 22, 23)
med = sum(float((A[l][:, best[l][0]] @ best[l][1])[isCon].median()) for l in WIRE)
ctl = sum(float((A[l][:, best[l][0]] @ best[l][1])[~isCon].mean()) for l in WIRE)
print("summed detector score over layers", WIRE, ": conclusion median", round(med, 1), "| control mean", round(ctl, 1), "| |g| =", round(float(g.norm()), 2), "| final rms =", round(rms_final, 2))

def wire(target_nats):
    """Add a connection: each detector neuron also writes along the closing direction, in proportion to its d'."""
    beta = target_nats * rms_final / (float(g.norm()) * med)
    for l in WIRE:
        name = f"model.language_model.layers.{l}.mlp.down_proj.weight"
        W = tools.get_weight(name); top, dv, _ = best[l]
        W[:, top] += beta * torch.outer(g_unit, dv)
        tools.save_edit(name, W)
    return beta

def trial_wire(label, target, veto_alpha=None):
    t = time.time(); beta = wire(target)
    if veto_alpha: project_out(g_unit, [A18], veto_alpha)       # layers 21 and 22 already hold the wiring edit
    m = json.load(open("runs/calib-08b-teacher/edit/manifest.json"))
    gen = evaluate.run(practice, boxed_math.conversation, boxed_math.score, batch_size=64, max_new_tokens=2048, **boxed_math.BLOG_SAMPLING)
    out = {"trial": label, **behaviour(gen), "median_think_len": think_len(gen), "weights_changed_pct": round(m["changed_weights_pct"], 4), "model_rel_change": round(m["relative_change_of_model"], 5), "seconds": round(time.time() - t)}
    tools.log("trial", **out); tools.reset_edit(); print(out, flush=True)
    return gen
w10 = trial_wire("wire 150 conclusion neurons (layers 21-23) to the closing direction, +10 nats", 10)
w14 = trial_wire("wire, +14 nats", 14)
w20 = trial_wire("wire, +20 nats", 20)
json.dump({"w10": w10, "w14": w14, "w20": w20}, open("runs/calib-08b-teacher/trials_wire.json", "w"))

W_U = tools.get_weight("lm_head.weight"); nw = tools.get_weight(LAY["final_norm"] + ".weight")
G = (W_U[WANT] - W_U[WANT].mean(0)) * nw                      # one direction per candidate token: digits, comma, bracket
def values_for(T, S, margin):
    idx = torch.tensor([WANT.index(t) for t in T.tolist()])
    g = G[idx]
    return margin * S[:, None] * g / g.pow(2).sum(1, keepdim=True)   # raises the target logit by about `margin` nats
def fit_trial(strength, margin, its=dev, label=None):
    t = time.time()
    e = edits.fit_keys_to_values(MLP_OUT + ".weight", K_fit, values_for(T_fit, S_fit, margin), PROTECT, strength)
    print(f"   solve {time.time()-t:.0f} s | relative change of the matrix {e['relative_change']:.4f} | rank {e['delta_rank']} | weights changed {e['changed_weights_pct']:.1f}%", flush=True)
    res = trial(label or f"least-squares fit on L29 MLP, strength {strength}, margin {margin}", its)
    tools.reset_edit(); return res
dev_base = [base_res[i["id"]]["correct"] for i in dev]
f1 = fit_trial(1.0, 3.0)
f2 = fit_trial(0.1, 3.0)
json.dump({"s1.0": f1, "s0.1": f2}, open("runs/ss-pro-4b-teacher/trials_fit.json", "w"))

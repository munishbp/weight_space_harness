# Can a closed-form fit move the first number toward the object position? Fit on seeds 0-9, test on seeds 10-19.
W_U = tools.get_weight("lm_head.weight"); nw = tools.get_weight(LAY["final_norm"] + ".weight")
F1 = [x for x in rows if x["task"] in ("push-v3", "window-open-v3") and first_tokens(x["obj"])[0] != first_tokens(x["goal"])[0]]
for x in F1: x["o"], x["g"] = first_tokens(x["obj"])[0], first_tokens(x["goal"])[0]
def keys_first(convs, mlp_out):
    K = tools.read(convs, [mlp_out], append_ids=[PREFIX_MOVE] * len(convs), inputs=True)[mlp_out]
    S = tools.read(convs, [LAST], append_ids=[PREFIX_MOVE] * len(convs))[LAST].pow(2).mean(1).sqrt()
    return K, S
def p_obj(label, xs):
    out = []
    for x in xs:
        lp = tools.read([x["conv"]], [], append_ids=[PREFIX_MOVE], logprobs_of=[x["o"], x["g"]])["logprobs"][0].exp()
        out.append((float(lp[0]), float(lp[1])))
    wins = sum(a > b for a, b in out)
    print(f"{label}: object token ahead of goal token in {wins} of {len(out)} | mean P(object) {sum(a for a, _ in out) / len(out):.2f} | mean P(goal) {sum(b for _, b in out) / len(out):.2f}", flush=True)
train = [x for x in F1 if x["seed"] < 10]; test = [x for x in F1 if x["seed"] >= 10]
p_obj("base, fit seeds", train); p_obj("base, test seeds", test)
for layer in (28, 20):
    mlp_out = f"model.language_model.layers.{layer}.mlp.down_proj"
    if layer not in PROTECT:
        text = torch.cat(tools.read_tokens(NEUTRAL, [mlp_out], inputs=True)[mlp_out]).double()
        other = torch.cat([keys_first(CTX1[t_][:20], mlp_out)[0] for t_ in TASKS if t_ not in ("push-v3", "window-open-v3")]).double()
        PROTECT[layer] = text.T @ text / len(text) + other.T @ other / len(other)
    K, S = keys_first([x["conv"] for x in train], mlp_out)
    D = torch.stack([(W_U[x["o"]] - W_U[x["g"]]) * nw for x in train])
    for margin in (6.0, 12.0):
        V = margin * S[:, None] * D / D.pow(2).sum(1, keepdim=True)
        e = edits.fit_keys_to_values(mlp_out + ".weight", K, V, PROTECT[layer], 1.0)
        print(f"L{layer} margin {margin} | relative change {e['relative_change']:.4f} | rank {e['delta_rank']}")
        p_obj("   edited, fit seeds", train); p_obj("   edited, test seeds", test)
        tools.reset_edit()

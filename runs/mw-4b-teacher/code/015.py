# Write the object-first state into one MLP, only for the first move of the push task.
def keys_move(convs, mlp_out):
    return torch.cat([tools.read(convs[i:i + 20], [mlp_out], append_ids=[PREFIX_MOVE] * len(convs[i:i + 20]), inputs=True)[mlp_out] for i in range(0, len(convs), 20)])
PX_train = [x for x in PX if x["seed"] < 40]; PX_test = [x for x in PX if x["seed"] >= 40]
PROT_F1, KEEP_F1 = {}, {}
def prepare(layer):
    mlp_out = f"model.language_model.layers.{layer}.mlp.down_proj"
    text = torch.cat(tools.read_tokens(NEUTRAL, [mlp_out], inputs=True)[mlp_out]).double()
    other = torch.cat([keys_move(CTX1[t_], mlp_out) for t_ in TASKS if t_ != "push-v3"] + [keys_at(CTX2[t_], mlp_out)[0] for t_ in TASKS]).double()
    PROT_F1[layer] = text.T @ text / len(text) + other.T @ other / len(other)
    # The third turn of every task writes a move too. The edit must not act there.
    KEEP_F1[layer] = torch.cat([keys_move(CTX3[t_], mlp_out) for t_ in TASKS if CTX3[t_]])
    return mlp_out
def state_fit(layer, alpha):
    mlp_out = f"model.language_model.layers.{layer}.mlp.down_proj"
    K = keys_move([x["conv"] for x in PX_train], mlp_out)
    d = DIRS[LAY["layers"][layer]]
    keys = torch.cat([K, KEEP_F1[layer]]); values = torch.cat([alpha * d[None, :].repeat(len(K), 1), torch.zeros(len(KEEP_F1[layer]), len(d))])
    return edits.fit_keys_to_values(mlp_out + ".weight", keys, values, PROT_F1[layer], 1.0)
def check(label, xs):
    out = []
    for x in xs:
        if x["o"] == x["g"]: continue
        lp = tools.read([x["conv"]], [], append_ids=[PREFIX_MOVE], logprobs_of=[x["o"], x["g"]])["logprobs"][0].exp()
        out.append((float(lp[0]), float(lp[1])))
    print(f"{label}: object token ahead in {sum(a > b for a, b in out)} of {len(out)} | mean P(object) {sum(a for a, _ in out) / len(out):.2f} | mean P(goal) {sum(b for _, b in out) / len(out):.2f}", flush=True)
check("base, seeds 0-39", PX_train); check("base, seeds 40-79", PX_test)
for layer, alphas in ((12, (1.0, 2.0, 4.0)), (20, (1.0, 2.0))):
    prepare(layer)
    for alpha in alphas:
        e = state_fit(layer, alpha)
        print(f"L{layer} alpha {alpha} | relative change of the matrix {e['relative_change']:.4f} | rank {e['delta_rank']}")
        check("   edited, seeds 40-79 (unseen)", PX_test)
        tools.reset_edit()

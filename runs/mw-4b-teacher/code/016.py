# The token-level fit on layer 28 with more push seeds: fit on seeds 0-39, check on seeds 40-79.
prepare(28)
def token_fit(xs, margin, layer=28):
    mlp_out = f"model.language_model.layers.{layer}.mlp.down_proj"
    xs = [x for x in xs if x["o"] != x["g"]]
    K = keys_move([x["conv"] for x in xs], mlp_out)
    S = torch.cat([tools.read([x["conv"] for x in xs[i:i + 20]], [LAST], append_ids=[PREFIX_MOVE] * len(xs[i:i + 20]))[LAST] for i in range(0, len(xs), 20)]).pow(2).mean(1).sqrt()
    D = torch.stack([(W_U[x["o"]] - W_U[x["g"]]) * nw for x in xs])
    keys = torch.cat([K, KEEP_F1[layer]]); values = torch.cat([margin * S[:, None] * D / D.pow(2).sum(1, keepdim=True), torch.zeros(len(KEEP_F1[layer]), D.shape[1])])
    return edits.fit_keys_to_values(mlp_out + ".weight", keys, values, PROT_F1[layer], 1.0)
PUSH_TEST = [("push-v3", s) for s in range(40, 80)]
base_push_test = [r for r in PUSH if r["seed"] >= 40]
for margin in (6.0, 12.0):
    e = token_fit(PX_train, margin)
    print(f"L28 token fit, margin {margin} | relative change of the matrix {e['relative_change']:.4f} | rank {e['delta_rank']}")
    check("   edited, seeds 40-79 (unseen)", PX_test)
    res = mw.run_episodes(PUSH_TEST, gen, batch_size=16)
    print(f"   push seeds 40-79 in the simulator: base {sum(r['correct'] for r in base_push_test)} of 40 -> edited {sum(r['correct'] for r in res)} of 40 |", paired_flips([r["correct"] for r in base_push_test], [r["correct"] for r in res]), flush=True)
    tools.reset_edit()

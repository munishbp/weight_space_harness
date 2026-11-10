ALLDP = [f"model.language_model.layers.{i}.mlp.down_proj" for i in range(24)]
nR = tools.read_tokens(seqs, ALLDP, positions=both2, inputs=True)         # neuron activations at repeat and new positions
mo = tools.read_tokens(seqs, MLP + ATT, positions=both2)                  # module outputs at the same positions
AR = {i: torch.cat(nR[ALLDP[i]]) for i in range(24)}
def layer_table(u):
    rowsu = []
    for i in range(24):
        for kind, names_ in (("mlp", MLP), ("att", ATT)):
            x = torch.cat(mo[names_[i]]) @ u
            rowsu.append((float(x[lab2].mean() - x[~lab2].mean()), float(x[lab2].mean()), float(x[~lab2].mean()), f"{kind}{i}"))
    return sorted(rowsu, reverse=True)
for ref in (16, 20):
    u = rdirs[ref]; tab = layer_table(u)
    total = float((resid2[ref][lab2] @ u).mean() - (resid2[ref][~lab2] @ u).mean())
    print(f"repeat direction of layer {ref}: total gap {total:.3f}. Top writers (gap | at repeat | at new):", [(k, round(d, 3), round(a, 3), round(b, 3)) for d, a, b, k in tab[:6]])

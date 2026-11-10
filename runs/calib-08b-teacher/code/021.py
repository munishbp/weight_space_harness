u = rdirs[20]
def top_neurons(layer, u, n):
    W = tools.get_weight(f"model.language_model.layers.{layer}.mlp.down_proj.weight")      # [1024, 3584]; column j is the output vector of neuron j
    align = u @ W                                                                            # how much neuron j writes along u per unit of activation
    c = AR[layer] * align                                                                    # contribution of each neuron at each position
    gap = c[lab2].mean(0) - c[~lab2].mean(0)
    top = gap.argsort(descending=True)[:n]
    return top, float(gap[top].sum()), float(gap.sum())
for layer in (15, 19):
    top, got, total = top_neurons(layer, u, 64)
    print(f"layer {layer}: the 64 top neurons carry {got:.3f} of the layer's gap of {total:.3f} along the repeat direction")

def edit_neurons(layers, u, n, alpha):
    for layer in layers:
        name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
        W = tools.get_weight(name); top, _, _ = top_neurons(layer, u, n)
        W[:, top] -= alpha * torch.outer(u, u @ W[:, top])
        tools.save_edit(name, W)
edit_neurons((15, 19), u, 64, 1.0); n1 = run_trial("repeat direction out of 64 neurons each in MLP 15 and 19, alpha 1"); tools.reset_edit()
edit_neurons((15, 19), u, 64, 2.0); n2 = run_trial("same, alpha 2"); tools.reset_edit()
project_out(u, [L + "15.mlp.down_proj.weight", L + "19.mlp.down_proj.weight"], 1.0); n3 = run_trial("repeat direction out of all of MLP 15 and 19, alpha 1"); tools.reset_edit()
json.dump({"n1": n1, "n2": n2, "n3": n3}, open("runs/calib-08b-teacher/trials_neurons.json", "w"))

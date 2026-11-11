# Keys for a closed-form fit: the neuron activations of one MLP at the step that writes the action name.
print({k: (v if isinstance(v, str) else (len(v), v[-1])) for k, v in LAY.items()})
LAST = LAY["layers"][-1]
def keys_at(convs, mlp_out):
    """Neuron activations (the input of the MLP output matrix) and the scale of the final residual, at the action-name step."""
    K, S = [], []
    for i in range(0, len(convs), 20):
        out = tools.read(convs[i:i + 20], [mlp_out, LAST], append_ids=[PREFIX] * len(convs[i:i + 20]), inputs=False)
        S.append(out[LAST].pow(2).mean(1).sqrt())
        K.append(tools.read(convs[i:i + 20], [mlp_out], append_ids=[PREFIX] * len(convs[i:i + 20]), inputs=True)[mlp_out])
    return torch.cat(K), torch.cat(S)
# Contexts at the first and third turn, for the protection set.
t = time.time()
CTX1 = {task: [context(r, 0) for r in rs] for task, rs in by_task.items()}
CTX3 = {task: [context(r, 2) for r in rs if len(r["primitives"]) > 2] for task, rs in by_task.items()}
print("contexts built in", round(time.time() - t), "s | turn 3 contexts:", {k.split("-v3")[0]: len(v) for k, v in CTX3.items()})
NEUTRAL = battery.neutral_text(64, 256)

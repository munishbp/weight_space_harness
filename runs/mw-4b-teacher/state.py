# Rebuild the search state after a kernel restart: the contexts of the base practice episodes and the key reader.
# It holds the definitions of step2.py, fit_prep.py, and fit1.py with no trial in between.
import importlib; importlib.reload(edits)
from wsp import battery
t = time.time()
base = json.load(open(f"{RUN}/base_practice.json"))
by_task = defaultdict(list)
for r in base: by_task[r["task"]].append(r)

def context(result, steps):
    """The prompt after the first `steps` primitives that the base model wrote in this episode."""
    ep = mw.Episode(result["task"], result["seed"])
    for prim in result["primitives"][:steps]: ep.apply(prim)
    conv = mw.conversation(ep); ep.close()
    return conv

PREFIX = tok.encode('{"action": "', add_special_tokens=False)
NAMES = {w: tok.encode('{"action": "' + w, add_special_tokens=False)[len(PREFIX)] for w in ("move", "close", "open", "done")}
LAST = LAY["layers"][-1]
def keys_at(convs, mlp_out):
    """Neuron activations (the input of the MLP output matrix) and the scale of the final residual, at the action-name step."""
    K, S = [], []
    for i in range(0, len(convs), 20):
        out = tools.read(convs[i:i + 20], [mlp_out, LAST], append_ids=[PREFIX] * len(convs[i:i + 20]), inputs=False)
        S.append(out[LAST].pow(2).mean(1).sqrt())
        K.append(tools.read(convs[i:i + 20], [mlp_out], append_ids=[PREFIX] * len(convs[i:i + 20]), inputs=True)[mlp_out])
    return torch.cat(K), torch.cat(S)
CTX1 = {task: [context(r, 0) for r in rs] for task, rs in by_task.items()}
CTX2 = {task: [context(r, 1) for r in rs] for task, rs in by_task.items()}
CTX3 = {task: [context(r, 2) for r in rs if len(r["primitives"]) > 2] for task, rs in by_task.items()}
NEUTRAL = battery.neutral_text(64, 256)
MLP_OUT = "model.language_model.layers.28.mlp.down_proj"
print("state rebuilt in", round(time.time() - t), "s | turn 3 contexts:", {k.split("-v3")[0]: len(v) for k, v in CTX3.items()})

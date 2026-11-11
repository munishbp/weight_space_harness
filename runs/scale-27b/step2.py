# Helpers for the habit recipe on the 27B model, and the probabilities of the action names at each turn.
import importlib
from wsp import habit, battery; importlib.reload(habit); importlib.reload(edits)
def context(result, steps):
    """The prompt after the first `steps` primitives that the base model wrote in this episode."""
    ep = mw.Episode(result["task"], result["seed"])
    for prim in result["primitives"][:steps]: ep.apply(prim)
    conv = mw.conversation(ep); ep.close()
    return conv
PREFIX = tok.encode('{"action": "', add_special_tokens=False)
NAMES = {w: tok.encode('{"action": "' + w, add_special_tokens=False)[len(PREFIX)] for w in ("move", "close", "open", "done")}
print("prefix", [tok.decode([t]) for t in PREFIX], "| names", {w: tok.decode([i]) for w, i in NAMES.items()})
def probs(convs, batch=8):
    out = []
    for i in range(0, len(convs), batch):
        out.append(tools.read(convs[i:i + batch], [], append_ids=[PREFIX] * len(convs[i:i + batch]), logprobs_of=list(NAMES.values()))["logprobs"].exp())
    return torch.cat(out)
print("MLP output matrices:", LAY["mlp_outs"][-3:] if "mlp_outs" in LAY else [k for k in LAY])

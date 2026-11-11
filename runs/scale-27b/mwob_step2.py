# Contexts at the fault steps of the email tasks and the probabilities of the action names on the 27B model.
# The page states come from the correct action sequences that the 4B run recorded (the forced replay of the practice seeds).
import importlib
from wsp import habit, battery; importlib.reload(habit); importlib.reload(edits)
forced2 = json.load(open("runs/mwob-4b-teacher/forced2_practice.json")); scan4 = json.load(open("runs/mwob-4b-teacher/base_scan.json"))
PREFIX = tok.encode('{"action": "', add_special_tokens=False)
NAMES = {w: tok.encode('{"action": "' + w, add_special_tokens=False)[len(PREFIX)] for w in ("click", "type", "key", "done")}
print("prefix", [tok.decode([t]) for t in PREFIX], "| names", {w: tok.decode([i]) for w, i in NAMES.items()})
def context(result, steps):
    ep = mb.Episode(result["task"], result["seed"])
    for a in result["actions"][:steps]:
        ep.apply(a)
        if ep.ended: return None
    return mb.conversation(ep)
is_mail = lambda r: ("reply" in r["utterance"] or "forward" in r["utterance"]) and acts(r).startswith("c c c t")
mail = [r for r in forced2 if is_mail(r)]
STEP = {"A (after the email click)": 1, "B (after the Reply click)": 2, "T (after the field click)": 3, "C (after the text)": 4}
CTX = {name: [(r["seed"], context(r, k)) for r in mail] for name, k in STEP.items()}
CTX = {name: [(s, c) for s, c in rows if c is not None] for name, rows in CTX.items()}
def probs(convs, batch=8):
    out = []
    for i in range(0, len(convs), batch):
        out.append(tools.read(convs[i:i + batch], [], append_ids=[PREFIX] * len(convs[i:i + batch]), logprobs_of=list(NAMES.values()))["logprobs"].exp())
    return torch.cat(out)
def p_table(label):
    for name, rows in CTX.items():
        p = probs([c for _, c in rows]); seen = torch.tensor([s < 10 for s, _ in rows])
        fmt = lambda q: {w: round(float(q[:, i].mean()), 2) for i, w in enumerate(NAMES)}
        print(f"{label} | {name:26s} n={len(rows)} | seeds 0-9 {fmt(p[seen])} | seeds 10-19 {fmt(p[~seen])}", flush=True)
p_table("27B base"); stamp("contexts and base probabilities done")
LAST = LAY["layers"][-1]
for L in (52, 56, 60):
    name = f"model.language_model.layers.{L}.mlp.down_proj"; layer = model.get_submodule(name)
    print(name, type(layer).__name__, getattr(layer, "kind", ""), getattr(layer, "shape", ""))

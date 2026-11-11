# Contexts at the three fault steps (from the forced replay) and the probabilities of the action names there.
MLP_OUT = "model.language_model.layers.28.mlp.down_proj"; LAST = LAY["layers"][-1]
def context(result, steps):
    """The prompt after the first `steps` actions of a recorded episode, or None when the episode ends before."""
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
def probs(convs):
    out = []
    for i in range(0, len(convs), 16):
        out.append(tools.read(convs[i:i + 16], [], append_ids=[PREFIX] * len(convs[i:i + 16]), logprobs_of=list(NAMES.values()))["logprobs"].exp())
    return torch.cat(out)
def p_table(label):
    for name, rows in CTX.items():
        p = probs([c for _, c in rows]); seen = torch.tensor([s < 10 for s, _ in rows])
        fmt = lambda q: {w: round(float(q[:, i].mean()), 2) for i, w in enumerate(NAMES)}
        print(f"{label} | {name:26s} n={len(rows)} | seeds 0-9 {fmt(p[seen])} | seeds 10-19 {fmt(p[~seen])}", flush=True)
p_table("base"); stamp("contexts and base probabilities done")

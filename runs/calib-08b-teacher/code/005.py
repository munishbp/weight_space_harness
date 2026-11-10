import random
from collections import Counter
rng = random.Random(1)
# Class W: the position just before a 'Wait' token. Class C: a position with the same current token, where another word follows.
before = Counter(s[i] for s, o in zip(seqs, offs) for i in range(o, len(s) - 1) if s[i + 1] in WAIT)
print("tokens before Wait:", [(repr(tok.decode([t])), n) for t, n in before.most_common(5)])
anchor = {t for t, n in before.most_common(3)}
posW, posC = [], []
for s, o in zip(seqs, offs):
    w = [i for i in range(o, len(s) - 1) if s[i + 1] in WAIT and s[i] in anchor]
    c = [i for i in range(o, len(s) - 1) if s[i + 1] not in WAIT and s[i] in anchor]
    posW.append(w); posC.append(rng.sample(c, min(len(c), 3 * max(len(w), 1))))
print("W positions:", sum(map(len, posW)), "| C positions:", sum(map(len, posC)))
nxt = Counter(tok.decode([s[i + 1]]) for s, c in zip(seqs, posC) for i in c)
print("what follows at C:", nxt.most_common(10))

LAYERS = [f"model.language_model.layers.{i}" for i in range(24)]
both = [w + c for w, c in zip(posW, posC)]
acts = tools.read_tokens(seqs, LAYERS, positions=both)
labels = torch.cat([torch.cat([torch.ones(len(w)), torch.zeros(len(c))]) for w, c in zip(posW, posC)]).bool()
resid = {i: torch.cat(acts[LAYERS[i]]) for i in range(24)}   # [positions, 1024] per layer
print("residual norm by layer:", [round(float(resid[i].norm(dim=1).mean()), 1) for i in range(0, 24, 3)])

def auc(scores, labels):
    order = scores.argsort(); ranks = torch.empty_like(scores); ranks[order] = torch.arange(1, len(scores) + 1, dtype=scores.dtype)
    n1, n0 = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

# Split the sequences in two halves: find the direction on one half, test it on the other.
seq_of = torch.cat([torch.full((len(b),), k) for k, b in enumerate(both)])
train, test = seq_of < 31, seq_of >= 31
dirs = {}
print("layer | rel. size of the difference | AUC on held-out replies")
for i in range(24):
    x = resid[i]
    d = x[train & labels].mean(0) - x[train & ~labels].mean(0)
    dirs[i] = d / d.norm()
    print(f"{i:5d} | {float(d.norm() / x.norm(dim=1).mean()):.3f} | {auc(x[test] @ dirs[i], labels[test]):.3f}")

rng = random.Random(2)
def repeat_classes(s, o, n_rep=8, n_new=4):
    """R: the next token completes an 8-gram that the reply already holds. N: it completes a new 4-gram."""
    seen_rep, seen_new, R, N = set(), set(), [], []
    for i in range(o, len(s) - 1):
        g8 = tuple(s[i - n_rep + 2 : i + 2]); g4 = tuple(s[i - n_new + 2 : i + 2])
        if i - o >= n_rep:
            if g8 in seen_rep: R.append(i)
            elif g4 not in seen_new: N.append(i)
        seen_rep.add(g8); seen_new.add(g4)
    return R, N
allR, allN = zip(*[repeat_classes(s, o) for s, o in zip(seqs, offs)])
blocks = [sum(sum(1 for i in R if (i - o) // 256 == b) for R, o in zip(allR, offs)) / (len(seqs) * 256) for b in range(8)]
print("share of positions inside a repeated 8-gram, by block of 256 tokens:", [round(x, 2) for x in blocks])
posR = [rng.sample(R, min(len(R), 30)) for R in allR]; posN = [rng.sample(N, min(len(N), 30)) for N in allN]
print("R positions:", sum(map(len, posR)), "| N positions:", sum(map(len, posN)))
both2 = [r + n for r, n in zip(posR, posN)]
acts2 = tools.read_tokens(seqs, LAYERS, positions=both2)
lab2 = torch.cat([torch.cat([torch.ones(len(r)), torch.zeros(len(n))]) for r, n in zip(posR, posN)]).bool()
seq2 = torch.cat([torch.full((len(b),), k) for k, b in enumerate(both2)])
tr2, te2 = seq2 < 31, seq2 >= 31
resid2 = {i: torch.cat(acts2[LAYERS[i]]) for i in range(24)}
rdirs = {}
print("layer | rel. size | held-out AUC (repeat vs new) | cosine with the Wait direction | this direction's AUC on Wait-vs-control")
for i in range(24):
    x = resid2[i]
    d = x[tr2 & lab2].mean(0) - x[tr2 & ~lab2].mean(0)
    rdirs[i] = d / d.norm()
    print(f"{i:5d} | {float(d.norm() / x.norm(dim=1).mean()):.3f} | {auc(x[te2] @ rdirs[i], lab2[te2]):.3f} | {float(rdirs[i] @ dirs[i]):+.2f} | {auc(resid[i][test] @ rdirs[i], labels[test]):.3f}")

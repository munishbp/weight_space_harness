LYS = list(range(12, 24))
DP = [f"model.language_model.layers.{i}.mlp.down_proj" for i in LYS]
bothCC = [c + t for c, t in zip(posCon, posCtl)]
neur = tools.read_tokens(seqs, DP, positions=bothCC, inputs=True)       # neuron activations, [positions, 3584] per layer
isCon = torch.cat([torch.cat([torch.ones(len(c)), torch.zeros(len(t))]) for c, t in zip(posCon, posCtl)]).bool()
seqCC = torch.cat([torch.full((len(b),), k) for k, b in enumerate(bothCC)])
trn, tst = seqCC % 2 == 0, seqCC % 2 == 1                               # alternate replies: find on one half, test on the other
print("train conclusions:", int((isCon & trn).sum()), "| test conclusions:", int((isCon & tst).sum()))
A = {l: torch.cat(neur[DP[k]]) for k, l in enumerate(LYS)}
best = {}
print("layer | top neuron: d' on train, AUC on test | number of neurons with test AUC > 0.9 among the 50 best on train")
for l in LYS:
    a = A[l]
    mc, mt = a[isCon & trn].mean(0), a[~isCon & trn].mean(0)
    sd = (a[isCon & trn].var(0) + a[~isCon & trn].var(0)).div(2).sqrt() + 1e-6
    d = (mc - mt) / sd
    top = d.argsort(descending=True)[:50]
    aucs = [auc(a[tst][:, j], isCon[tst]) for j in top.tolist()]
    best[l] = (top, d[top], torch.tensor(aucs))
    print(f"{l:5d} | d'={float(d[top[0]]):.2f}, AUC={aucs[0]:.3f} | {sum(x > 0.9 for x in aucs)}")
# A linear readout over the 50 best neurons of one layer: sum of activations weighted by d'.
for l in (17, 19, 21, 22, 23):
    top, dv, _ = best[l]; score = A[l][:, top] @ dv
    sc, st = score[isCon & tst], score[~isCon & tst]
    print(f"layer {l}: weighted sum of 50 neurons: test AUC {auc(score[tst], isCon[tst]):.3f} | conclusion mean {float(sc.mean()):.2f}, control mean {float(st.mean()):.2f}, control 99th pct {float(st.quantile(0.99)):.2f}, conclusion median {float(sc.median()):.2f}")

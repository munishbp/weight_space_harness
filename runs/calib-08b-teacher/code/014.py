# Apply the alpha 1.75 edit and keep it. Then look at the replies that still never close their thinking.
project_out(g / g.norm(), [M21, M22, A18], 1.75)
gen = sweep[1.75]
seqs2, offs2 = [], []
for p, r in zip(practice, gen):
    if THINK_END in r["ids"]: continue
    pre = tools.prompt_ids(boxed_math.conversation(p), thinking=True); seqs2.append(pre + r["ids"]); offs2.append(len(pre))
lp2 = tools.read_logprobs(seqs2, [THINK_END])
peaks2 = sorted(((float(lp[o:, 0].exp().max()), k, o + int(lp[o:, 0].argmax())) for k, (o, lp) in enumerate(zip(offs2, lp2))), reverse=True)
ps = [x[0] for x in peaks2]
print(len(seqs2), "replies never close. Peak P(</think>) per reply: median", round(ps[len(ps) // 2], 3), "| above 0.5:", sum(p > 0.5 for p in ps), "| above 0.1:", sum(p > 0.1 for p in ps), "| below 0.01:", sum(p < 0.01 for p in ps))
allp2 = torch.cat([lp[o:, 0].exp() for o, lp in zip(offs2, lp2)])
print("positions with P > 0.1:", int((allp2 > 0.1).sum()), "of", len(allp2), "| expected number of closes per reply if sampled independently:", round(float(allp2.sum()) / len(seqs2), 2))
comp = Counter(tok.decode([seqs2[k][i + 1]]) for p, k, i in peaks2 if p > 0.02)
print("token chosen at the peak:", comp.most_common(6))
for p, k, i in peaks2[:2] + peaks2[len(peaks2) // 2: len(peaks2) // 2 + 2]:
    print(f"--- peak P={p:.3f} at reply token {i - offs2[k]}:"); print(tok.decode(seqs2[k][max(offs2[k], i - 50): i + 1])[-220:])

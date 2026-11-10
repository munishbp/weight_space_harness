lp_end = tools.read_logprobs(seqs, [THINK_END])
peaks = []
for k, (s, o, lp) in enumerate(zip(seqs, offs, lp_end)):
    p = lp[o:, 0].exp(); j = int(p.argmax()); peaks.append((float(p[j]), k, o + j))
ps = sorted(x[0] for x in peaks)
print("max P(</think>) per stuck reply: min", f"{ps[0]:.2e}", "| median", f"{ps[len(ps)//2]:.3f}", "| max", f"{ps[-1]:.3f}")
print("replies whose peak is above 0.5:", sum(p > 0.5 for p in ps), "| above 0.1:", sum(p > 0.1 for p in ps), "| above 0.01:", sum(p > 0.01 for p in ps), "of", len(ps))
allp = torch.cat([lp[o:, 0].exp() for o, lp in zip(offs, lp_end)])
print("positions with P > 0.1:", int((allp > 0.1).sum()), "| P > 0.01:", int((allp > 0.01).sum()), "of", len(allp))
# What does the text look like where the model nearly closes its thinking?
for p, k, i in sorted(peaks, reverse=True)[:5]:
    s = seqs[k]
    print(f"--- P={p:.3f} at reply token {i - offs[k]} | chosen next token: {tok.decode([s[i+1]])!r}")
    print(tok.decode(s[max(offs[k], i - 60): i + 1])[-260:])
# Which token precedes the high-probability positions?
high = Counter()
for s, o, lp in zip(seqs, offs, lp_end):
    for j in (lp[o:, 0].exp() > 0.05).nonzero().flatten().tolist(): high[tok.decode([s[o + j]])] += 1
print("tokens at positions with P(</think>) > 0.05:", [(repr(t), n) for t, n in high.most_common(8)])

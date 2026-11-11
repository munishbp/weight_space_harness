# Where does the model almost close its thinking, and which parts push against the closing token there?
rng = random.Random(3)
seqs, offs = [], []
for p, r in zip(practice, base_gen):
    if THINK_END in r["ids"]: continue
    pre = tools.prompt_ids(boxed_math.conversation(p), thinking=True); seqs.append(pre + r["ids"]); offs.append(len(pre))
print(len(seqs), "replies never close their thinking")
lp_end = tools.read_logprobs(seqs, [THINK_END])
peaks = sorted(float(lp[o:, 0].exp().max()) for o, lp in zip(offs, lp_end))
print("peak P(</think>) per stuck reply: min", f"{peaks[0]:.2e}", "| median", round(peaks[len(peaks) // 2], 3), "| max", round(peaks[-1], 3), "| above 0.1:", sum(p > 0.1 for p in peaks), "| above 0.01:", sum(p > 0.01 for p in peaks))
posK, posB = [], []
for s, o, lp in zip(seqs, offs, lp_end):
    p = lp[:, 0].exp()
    posK.append([i for i in range(o + 50, len(s) - 1) if p[i] > 3e-4]); posB.append(rng.sample(range(o + 50, len(s) - 1), 12))
print("K positions (P above 0.0003):", sum(map(len, posK)), "in", sum(1 for k in posK if k), "replies | B positions:", sum(map(len, posB)))
print("tokens at K:", [(repr(tok.decode([t])), n) for t, n in Counter(s[i] for s, k in zip(seqs, posK) for i in k).most_common(6)])
nxt = Counter(s[i + 1] for s, k in zip(seqs, posK) for i in k).most_common(6)
print("next token chosen at K:", [(repr(tok.decode([t])), n) for t, n in nxt])
ALT = nxt[0][0]
g = attribution.token_direction(THINK_END, [ALT])
A = attribution.attribute(seqs, [k + b for k, b in zip(posK, posB)], g)
isK = torch.cat([torch.cat([torch.ones(len(k)), torch.zeros(len(b))]) for k, b in zip(posK, posB)]).bool()
print(f"split error {A['error']:.4f} | logit(</think>) - logit(alt): mean at K {float(A['total'][isK].mean()):+.2f}, at B {float(A['total'][~isK].mean()):+.2f}")
short = lambda name: name.replace("model.language_model.layers.", "L").replace(".linear_attn", " att").replace(".self_attn", " att").replace(".mlp", " mlp").replace("model.language_model.", "")
table_rows = sorted((float(v[isK].mean()), float(v[~isK].mean()), name) for name, v in A["shares"].items())
print("parts that push hardest against the closing token at K (mean at K | mean at B):")
for a, b, name in table_rows[:10]: print(f"  {short(name):12s} {a:+7.2f} | {b:+7.2f}")
print("parts that push hardest for it:")
for a, b, name in table_rows[-6:]: print(f"  {short(name):12s} {a:+7.2f} | {b:+7.2f}")

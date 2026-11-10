rng = random.Random(3)
# K: positions where the model itself gives the closing token a visible chance. B: random other positions.
posK, posB = [], []
for s, o, lp in zip(seqs, offs, lp_end):
    p = lp[:, 0].exp()
    k = [i for i in range(o + 50, len(s) - 1) if p[i] > 3e-4]
    posK.append(k); posB.append(rng.sample(range(o + 50, len(s) - 1), 12))
print("K positions:", sum(map(len, posK)), "in", sum(1 for k in posK if k), "replies | B positions:", sum(map(len, posB)))
print("tokens at K:", [(repr(tok.decode([t])), n) for t, n in Counter(s[i] for s, k in zip(seqs, posK) for i in k).most_common(5)])
print("next token chosen at K:", [(repr(tok.decode([t])), n) for t, n in Counter(s[i + 1] for s, k in zip(seqs, posK) for i in k).most_common(5)])

# Direct logit attribution. The final norm is RMSNorm, so logit(token) = (W_U[token] * norm.weight) . x / rms(x).
W_U = tools.get_weight("lm_head.weight"); nw = tools.get_weight("model.language_model.norm.weight")
alt = Counter(s[i + 1] for s, k in zip(seqs, posK) for i in k).most_common(1)[0][0]
g = (W_U[THINK_END] - W_U[alt]) * nw                      # direction that favours </think> over the usual next token
g_abs = (W_U[THINK_END] - W_U.mean(0)) * nw                # direction that favours </think> over the average token
print("alternative token:", repr(tok.decode([alt])), "| cosine(g, g_abs) =", round(float(torch.nn.functional.cosine_similarity(g, g_abs, dim=0)), 2))

MLP = [f"model.language_model.layers.{i}.mlp" for i in range(24)]
names = {n for n, _ in model.named_modules()}
ATT = [f"model.language_model.layers.{i}.{'self_attn' if f'model.language_model.layers.{i}.self_attn' in names else 'linear_attn'}" for i in range(24)]
bothKB = [k + b for k, b in zip(posK, posB)]
out = tools.read_tokens(seqs, MLP + ATT + ["model.language_model.layers.23", "model.language_model.embed_tokens"], positions=bothKB)
isK = torch.cat([torch.cat([torch.ones(len(k)), torch.zeros(len(b))]) for k, b in zip(posK, posB)]).bool()
final = torch.cat(out["model.language_model.layers.23"]); rms = final.pow(2).mean(1).sqrt()
total = (final @ g) / rms
print(f"logit(</think>) - logit(alt): mean at K {float(total[isK].mean()):+.2f}, at B {float(total[~isK].mean()):+.2f}")
parts = {"emb": torch.cat(out["model.language_model.embed_tokens"])}
for i in range(24): parts[f"mlp{i}"] = torch.cat(out[MLP[i]]); parts[f"att{i}"] = torch.cat(out[ATT[i]])
check = sum(parts.values()); print("sum of parts equals the final residual:", round(float((check - final).norm() / final.norm()), 4), "(relative error)")
contrib = {k: (v @ g) / rms for k, v in parts.items()}
rowsK = sorted(((float(c[isK].mean()), float(c[~isK].mean()), k) for k, c in contrib.items()))
print("most negative contributions to the closing token at K (mean at K | mean at B):")
for cK, cB, k in rowsK[:10]: print(f"  {k:6s} {cK:+7.2f} | {cB:+7.2f}")
print("most positive:")
for cK, cB, k in rowsK[-6:]: print(f"  {k:6s} {cK:+7.2f} | {cB:+7.2f}")

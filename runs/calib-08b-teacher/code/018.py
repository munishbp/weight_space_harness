# Direct attribution for the token 'Wait' against the words that usually start a line.
wait_tok = Counter(s[i + 1] for s, w in zip(seqs, posW) for i in w).most_common(1)[0][0]
posC2 = [[i for i in c if tok.decode([s[i + 1]]).strip().isalpha()] for s, c in zip(seqs, posC)]
alts = [t for t, n in Counter(s[i + 1] for s, c in zip(seqs, posC2) for i in c).most_common(6)]
print("Wait token:", repr(tok.decode([wait_tok])), "| alternatives:", [tok.decode([t]) for t in alts], "| W:", sum(map(len, posW)), "C2:", sum(map(len, posC2)))
gw = (W_U[wait_tok] - W_U[alts].mean(0)) * nw
bothW = [w + c for w, c in zip(posW, posC2)]
outW = tools.read_tokens(seqs, MLP + ATT + ["model.language_model.layers.23", "model.language_model.embed_tokens"], positions=bothW)
isW = torch.cat([torch.cat([torch.ones(len(w)), torch.zeros(len(c))]) for w, c in zip(posW, posC2)]).bool()
finalW = torch.cat(outW["model.language_model.layers.23"]); rmsW = finalW.pow(2).mean(1).sqrt()
tot = (finalW @ gw) / rmsW
print(f"logit(Wait) - logit(alternatives): mean at W {float(tot[isW].mean()):+.2f}, at C2 {float(tot[~isW].mean()):+.2f}")
partsW = {"emb": torch.cat(outW["model.language_model.embed_tokens"])}
for i in range(24): partsW[f"mlp{i}"] = torch.cat(outW[MLP[i]]); partsW[f"att{i}"] = torch.cat(outW[ATT[i]])
cW = {k: (v @ gw) / rmsW for k, v in partsW.items()}
rws = sorted(((float(c[isW].mean() - c[~isW].mean()), float(c[isW].mean()), float(c[~isW].mean()), k) for k, c in cW.items()), reverse=True)
print("components that promote 'Wait' more at W than at C2 (difference | at W | at C2):")
for d, a, b, k in rws[:9]: print(f"  {k:6s} {d:+6.2f} | {a:+6.2f} | {b:+6.2f}")
print("sum of all differences:", round(sum(r[0] for r in rws), 2))

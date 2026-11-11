W_U = tools.get_weight("lm_head.weight"); nw = tools.get_weight(LAY["final_norm"] + ".weight")
rows, dirs = [], []
for r in rec_quick:
    it = by[r["id"]]; (lo, hi), _ = target_ranges(it); V = viable_first_digits(lo, hi)
    c = r["ids"][start] - 15
    if not V or not (0 <= c <= 9): continue
    lp = r["lp"][start][:10]
    if c in V:
        rival = max((d for d in range(10) if d not in V), key=lambda d: float(lp[d]))
        d_vec = W_U[15 + c] - W_U[15 + rival]; kind = "success"; gap = float(lp[c] - lp[rival])
    else:
        d_vec = W_U[[15 + v for v in V]].mean(0) - W_U[15 + c]; kind = "failure"; gap = float(torch.logsumexp(lp[V], 0) - lp[c])
    rows.append((it, kind, gap)); dirs.append(d_vec * nw)
dirs = torch.stack(dirs); print(Counter(k for _, k, _ in rows))
shares = defaultdict(list); totals = []; errs = []
t = time.time()
for s in range(0, len(rows), 4):
    chunk = rows[s:s + 4]
    a = attribution.attribute_last([conv_with()(it) for it, _, _ in chunk], dirs[s:s + 4], append_ids=[PREFIX] * len(chunk))
    for name, v in a["shares"].items(): shares[name].append(v)
    totals.append(a["total"]); errs.append(a["error"])
shares = {k: torch.cat(v) for k, v in shares.items()}; totals = torch.cat(totals)
print(f"attribution pass: {time.time()-t:.0f} s | mean split error {sum(errs)/len(errs):.4f}")
isS = torch.tensor([k == "success" for _, k, _ in rows]); gaps = torch.tensor([g for _, _, g in rows])
print(f"logit gap toward the correct digit, from the attribution: success {float(totals[isS].mean()):+.2f}, failure {float(totals[~isS].mean()):+.2f} | from the recorded log-probs: success {float(gaps[isS].mean()):+.2f}, failure {float(gaps[~isS].mean()):+.2f}")
short = lambda n: n.replace("model.language_model.layers.", "L").replace("model.language_model.", "")
table = sorted(((float(v[~isS].mean()), float(v[isS].mean()), short(k)) for k, v in shares.items()))
print("components that push AWAY from the correct digit in failures (mean share in failures | in successes):")
for f, s_, k in table[:12]: print(f"  {k:22s} {f:+6.2f} | {s_:+6.2f}")
print("components that push TOWARD the correct digit in failures:")
for f, s_, k in table[-10:]: print(f"  {k:22s} {f:+6.2f} | {s_:+6.2f}")
torch.save({"shares": shares, "totals": totals, "isS": isS, "ids": [it["id"] for it, _, _ in rows]}, "runs/ss-pro-4b-teacher/attr_first_digit.pt")

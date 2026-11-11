# Side effect in click-pie: the fit raised `click` at the action step of another task type.
# Add the action steps of the other task types (seeds 0 and 1) to the fit with a zero value.
tools.reset_edit(); keep = []
for i in range(0, len(OTHER), 26):
    for r in [r for r in scan if r["task"] in OTHER[i:i + 26] and r["seed"] < 2]:
        for k in range(min(4, len(r["actions"]) + 1)):
            conv = context(r, k)
            if conv is not None: keep.append(conv)
    mb.close_all()
keep += [context(r, 0) for r in base if r["seed"] < 10]
K_KEEP = keys_at(keep)[0]; stamp(f"{len(keep)} action steps to keep")
def fit2(margin, keep_weight, mask=lambda name: SEEN[name]):
    ks, vs = [], []
    for name, g in (("A", g_type), ("B", g_type), ("C", g_key)):
        full = [n for n in CTX if n.startswith(name)][0]; K, S = KEYS[full]; m = mask(full)
        ks.append(K[m]); vs.append(edits.logit_shift_values(g, S[m], margin))
    full = [n for n in CTX if n.startswith("T")][0]
    zero = torch.cat([KEYS[full][0][mask(full)], K_KEEP * keep_weight**0.5])
    ks.append(zero); vs.append(torch.zeros(len(zero), vs[0].shape[1], dtype=vs[0].dtype, device=vs[0].device))
    return edits.fit_keys_to_values(MLP_OUT + ".weight", torch.cat(ks), torch.cat(vs), P_ALL, 1.0)
pie = [context(r, 1) for r in scan if r["task"] == "click-pie"]
def pie_p(): 
    p = probs(pie); return {w: [round(float(v), 2) for v in p[:, i]] for i, w in enumerate(NAMES) if w in ("click", "key")}
print("base | click-pie after the first click, seeds 0 to 3:", pie_p())
for margin, keep_weight in ((12.0, 1.0), (12.0, 10.0)):
    e = fit2(margin, keep_weight)
    print(f"margin {margin:g}, keep weight {keep_weight:g}, matrix change {e['relative_change']:.3f} | click-pie:", pie_p())
    p_table(f"margin {margin:g}, keep weight {keep_weight:g}")
    tools.reset_edit()
stamp("refit sweep done")

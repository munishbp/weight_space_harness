# What the stuck replies hold: the text near the middle and near the end, and how much of it repeats.
stuck = [r for r in base_gen if THINK_END not in r["ids"]]
def repeat_share(ids, n=20):
    """The share of 20-token windows in the second half that also occur earlier in the reply."""
    seen, hits, total = set(), 0, 0
    for i in range(0, len(ids) - n):
        w = tuple(ids[i:i + n])
        if i > len(ids) // 2:
            total += 1; hits += w in seen
        seen.add(w)
    return hits / max(total, 1)
shares = sorted(repeat_share(r["ids"]) for r in stuck)
print("repeat share in the second half of the stuck replies: min", round(shares[0], 2), "| median", round(shares[len(shares) // 2], 2), "| max", round(shares[-1], 2), "| above 0.5:", sum(s > 0.5 for s in shares), "of", len(shares))
closed = [r for r in base_gen if THINK_END in r["ids"]]
cs = sorted(repeat_share(r["ids"][:r["ids"].index(THINK_END)]) for r in closed)
print("the same for the replies that close: median", round(cs[len(cs) // 2], 2), "| max", round(cs[-1], 2))
for r in stuck[:3]:
    print("--- middle:", repr(tok.decode(r["ids"][900:1010])[:330])); print("    end:   ", repr(tok.decode(r["ids"][-110:])[:330]))

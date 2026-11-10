import re
tools.reset_edit()
NL = {t for t in range(len(tok)) if False}  # placeholder, replaced below
nl_ids = {i for w in ["\n", "\n\n"] for i in tok.encode(w, add_special_tokens=False)}
print("newline ids:", nl_ids)
CONCL = re.compile(r"(?i)(final answer|the answer is|answer:|result is|i will write|i'm confident|write the solution|write down the final|ready to (write|answer)|no ambiguity|logic (stands|holds)|^\W*correct\.?\W*$|calculation is correct|looks correct|seems correct|is correct\.)")
INDENT = [t for t, n in Counter(s[i + 1] for s, o in zip(seqs, offs) for i in range(o, len(s) - 1) if s[i] in nl_ids).most_common(4)]
posCon, posCtl = [], []
rng = random.Random(4)
for s, o in zip(seqs, offs):
    con, ctl = [], []
    for i in range(o + 30, len(s) - 1):
        if s[i] not in nl_ids: continue
        # The last line before this line break.
        j = i - 1
        while j > o and s[j] not in nl_ids: j -= 1
        line = tok.decode(s[j + 1: i])
        (con if CONCL.search(line) else ctl).append(i)
    posCon.append(con); posCtl.append(rng.sample(ctl, min(len(ctl), 40)))
print("conclusion line breaks:", sum(map(len, posCon)), "in", sum(1 for c in posCon if c), "replies | control line breaks:", sum(map(len, posCtl)))
ex = [(s, i) for s, c in zip(seqs, posCon) for i in c]; rng.shuffle(ex)
for s, i in ex[:8]:
    j = i - 1
    while s[j] not in nl_ids: j -= 1
    print("  conclusion line:", repr(tok.decode(s[j + 1: i]))[:110])
# P(</think>) at the two classes in the base model.
pc = torch.cat([lp[torch.tensor(c, dtype=torch.long), 0] for lp, c in zip(lp_end, posCon) if c]).exp()
pt = torch.cat([lp[torch.tensor(c, dtype=torch.long), 0] for lp, c in zip(lp_end, posCtl) if c]).exp()
print(f"base P(</think>): conclusion median {float(pc.median()):.2e}, mean {float(pc.mean()):.2e} | control median {float(pt.median()):.2e}, mean {float(pt.mean()):.2e}")

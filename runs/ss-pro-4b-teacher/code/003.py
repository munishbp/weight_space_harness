import math
DIG = list(range(15, 25)); COMMA = tok.encode(",", add_special_tokens=False)[0]; CLOSE = tok.encode("]", add_special_tokens=False)[0]
WANT = DIG + [COMMA, CLOSE]
class Recorder:
    """Keep the log-probability of each digit, the comma, and the bracket at every generation step."""
    def __init__(self): self.steps = []
    def __call__(self, input_ids, scores):
        self.steps.append(torch.log_softmax(scores.float(), dim=-1)[:, WANT].cpu()); return scores
def generate_recorded(its, build=None, batch=4):
    build = build or conv_with(); out = []
    for s in range(0, len(its), batch):
        chunk = its[s:s + batch]; rec = Recorder()
        replies = tools.generate([build(i) for i in chunk], max_new_tokens=64, thinking=False, logits_processor=[rec])
        lp = torch.stack(rec.steps, dim=1)                          # [batch, steps, 12]
        for k, (it, r) in enumerate(zip(chunk, replies)):
            out.append({"id": it["id"], **r, **sp.score(it, r["text"]), "lp": lp[k, :len(r["ids"])]})
    return out
def decision(it, r):
    """The first digit step where the reply leaves the target range, with the probability of the steps that stay inside."""
    ids = r["ids"]; W, H = it["img_size"]; x1, y1, x2, y2 = it["bbox"]
    try: start = max(i for i, t in enumerate(ids) if tok.decode([t]) == " [") + 1
    except ValueError: return None
    pos = start
    for axis, lo, hi, end in (("x", 1000 * x1 / W, 1000 * x2 / W, COMMA), ("y", 1000 * y1 / H, 1000 * y2 / H, CLOSE)):
        valid = {str(v) for v in range(math.ceil(lo), math.floor(hi) + 1)}
        prefix = ""
        while pos < len(ids):
            t = ids[pos]
            viable = {WANT.index(15 + int(s[len(prefix)])) for s in valid if s.startswith(prefix) and len(s) > len(prefix)}
            if prefix in valid: viable.add(WANT.index(end))
            chosen = WANT.index(t) if t in WANT else None
            if chosen is None or chosen not in viable:
                p_viable = float(r["lp"][pos][sorted(viable)].exp().sum()) if viable else 0.0
                return {"axis": axis, "digit_index": len(prefix), "p_chosen": float(r["lp"][pos][chosen].exp()) if chosen is not None else 0.0, "p_viable": p_viable, "empty_range": not valid}
            if t == end: pos += 1; break
            prefix += str(t - 15); pos += 1
        if axis == "x" and pos < len(ids) and tok.decode([ids[pos]]) == " ": pos += 1
    return None
t = time.time(); rec_quick = generate_recorded(quick); print(f"recorded pass: {time.time()-t:.0f} s | accuracy {100*sum(r['correct'] for r in rec_quick)/len(rec_quick):.1f}% (baseline file: {acc(quick)}%)")
by = {i["id"]: i for i in items}
dec = [(r, decision(by[r["id"]], r)) for r in rec_quick]
print("correct replies flagged as leaving the range (must be 0):", sum(1 for r, d in dec if r["correct"] and d))
fails = [(r, d) for r, d in dec if not r["correct"] and d]
print("failures:", len(fails), "| target range holds no whole number:", sum(d["empty_range"] for _, d in fails))
from collections import Counter
print("where the reply first leaves the range (axis, digit index):", sorted(Counter((d["axis"], d["digit_index"]) for _, d in fails).items()))
bins = Counter("P(in range) >= 0.3" if d["p_viable"] >= 0.3 else "0.1 to 0.3" if d["p_viable"] >= 0.1 else "0.01 to 0.1" if d["p_viable"] >= 0.01 else "below 0.01" for _, d in fails)
print("probability of a digit that stays in range, at the step of the error:", dict(bins))
for name, cond in (("first digit (hundreds)", lambda d: d["digit_index"] == 0), ("second digit (tens)", lambda d: d["digit_index"] == 1), ("third digit (units)", lambda d: d["digit_index"] >= 2)):
    sel = [d for _, d in fails if cond(d)]
    if sel: print(f"  {name}: {len(sel)} errors | median P(in range) {sorted(d['p_viable'] for d in sel)[len(sel)//2]:.3f} | median P(chosen) {sorted(d['p_chosen'] for d in sel)[len(sel)//2]:.3f} | P(in range) >= 0.2 in {sum(d['p_viable'] >= 0.2 for d in sel)}")
conf = [float(r["lp"][i][WANT.index(t)].exp()) for r in rec_quick if r["correct"] for i, t in enumerate(r["ids"]) if t in DIG]
print("correct replies: median probability of each chosen digit", round(sorted(conf)[len(conf)//2], 3))
torch.save({"rec": [{k: v for k, v in r.items()} for r in rec_quick]}, "runs/ss-pro-4b-teacher/recorded_quick.pt")

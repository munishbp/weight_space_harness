import random
rng = random.Random(11)
texts = [i for i in items if i["ui_type"] == "text"]
Tpos = rng.sample([i for i in texts if base_res[i["id"]]["correct"]], 100)          # text targets, clicked correctly
Ipos = [i for i in practice if base_res[i["id"]]["correct"]]                         # icon targets, clicked correctly
Ineg = rng.sample([i for i in practice if not base_res[i["id"]]["correct"]], 100)    # icon targets, missed
print("icon correct:", len(Ipos), "| text correct:", len(Tpos), "| icon missed:", len(Ineg))
LAYERS = [LAY["layers"][i] for i in range(8, 32, 2)]
def read_states(its):
    out = defaultdict(list)
    for s in range(0, len(its), 4):
        chunk = its[s:s + 4]
        a = tools.read([conv_with()(i) for i in chunk], LAYERS, append_ids=[PREFIX] * len(chunk))
        for k, v in a.items(): out[k].append(v)
    return {k: torch.cat(v) for k, v in out.items()}
t = time.time(); SI, ST, SN = read_states(Ipos), read_states(Tpos), read_states(Ineg); print(f"read pass: {time.time()-t:.0f} s")
torch.save({"SI": SI, "ST": ST, "SN": SN, "Ipos": [i["id"] for i in Ipos], "Tpos": [i["id"] for i in Tpos], "Ineg": [i["id"] for i in Ineg]}, "runs/ss-pro-4b-teacher/states_icon_text.pt")
def auc(a, b):
    s = torch.cat([a, b]); lab = torch.cat([torch.ones(len(a)), torch.zeros(len(b))]).bool()
    order = s.argsort(); ranks = torch.empty_like(s); ranks[order] = torch.arange(1, len(s) + 1, dtype=s.dtype)
    return float((ranks[lab].sum() - lab.sum() * (lab.sum() + 1) / 2) / (lab.sum() * (~lab).sum()))
print("layer | size of (icon - text) relative to the state | held-out AUC icon vs text | where the missed icons sit: 0 = like text, 1 = like icon")
VDIR = {}
for name in LAYERS:
    i, tt, n = SI[name], ST[name], SN[name]
    hi, ht = len(i) // 2, len(tt) // 2
    v = i[:hi].mean(0) - tt[:ht].mean(0)                       # direction from the first half
    u = v / v.norm()
    a = auc(i[hi:] @ u, tt[ht:] @ u)                           # tested on the second half
    full = i.mean(0) - tt.mean(0); VDIR[name] = full
    pi, pt, pn = float((i @ full).mean()), float((tt @ full).mean()), float((n @ full).mean())
    print(f"{name.split('.')[-1]:>5s} | {float(full.norm() / i.norm(dim=1).mean()):.3f} | {a:.3f} | {(pn - pt) / (pi - pt):.2f}")
torch.save(VDIR, "runs/ss-pro-4b-teacher/vdir_icon_text.pt")

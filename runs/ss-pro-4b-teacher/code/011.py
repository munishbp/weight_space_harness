# First-digit decisions on unseen items: the model's recorded log-probs plus the correction, at several margins.
first_index, pos = [], 0
for it in dev:
    ids, k, nxt = forced_reply(it); first_index.append(pos); pos += sum(1 for x in nxt if x != SPACE)
assert pos == len(K_dev)
rec_by = {r["id"]: r for r in rec_quick}
lp0 = torch.stack([rec_by[it["id"]]["lp"][start][:10] for it in dev])           # the base model's log-probs of the ten digits
V0 = [viable_first_digits(*target_ranges(it)[0]) for it in dev]
ok0 = lambda scores: torch.tensor([int(scores[i].argmax()) in V0[i] for i in range(len(dev))])
base_ok = ok0(lp0); print("base model, first digit of x in range:", int(base_ok.sum()), "of", len(dev))
D = delta_for(1.0, 3.0)
shift0 = ((K_dev[first_index] @ D.T) @ G.T / S_dev[first_index][:, None])[:, :10]   # logit shift of the ten digits at margin 3
print("correction alone picks an in-range first digit:", int(ok0(shift0).sum()), "of", len(dev), "(chance about", round(sum(len(v) for v in V0) / 10 / len(dev) * len(dev)), ")")
print("on the", int((~base_ok).sum()), "items where the base model is wrong: correction alone is right in", int(ok0(shift0)[~base_ok].sum()))
for scale in (1, 2, 4, 8, 16):
    new_ok = ok0(lp0 + scale * shift0)
    print(f"  margin {3*scale:3d} nats: in range {int(new_ok.sum())} | fixed {int((new_ok & ~base_ok).sum())} | broken {int((~new_ok & base_ok).sum())}")

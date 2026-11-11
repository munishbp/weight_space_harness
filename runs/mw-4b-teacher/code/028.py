# Does the weight change write the direction at the action-name position of the other tasks, or does the shift come from elsewhere?
tools.reset_edit()
W0 = tools.get_weight(MLP_OUT + ".weight").double().clone()
zero_keys = torch.cat([half(k, True) for k in OTHER3.values()])
keys = torch.cat([half(KD, True), zero_keys]); values = torch.cat([half(VD, True), torch.zeros(len(zero_keys), VD.shape[1], dtype=VD.dtype, device=VD.device)])
e = edits.fit_keys_to_values(MLP_OUT + ".weight", keys, values, P_DOOR, 1.0)
D = tools.get_weight(MLP_OUT + ".weight").double() - W0
g = g_door.double().to(D.device); gn = g / g.norm()
def along(K):
    out = K.double().to(D.device) @ D.T          # the vector that the change adds for each key
    return float((out @ gn).mean()), float(out.norm(dim=1).mean())
print("rank of the change (singular values over 1% of the top):", int((torch.linalg.svdvals(D.float()) > 0.01 * torch.linalg.svdvals(D.float())[0]).sum()))
print("door, seeds in the fit      (along g, norm):", [round(v, 2) for v in along(half(KD, True))])
print("door, unseen seeds          (along g, norm):", [round(v, 2) for v in along(half(KD, False))])
for task, K in OTHER3.items():
    a, b = along(half(K, True)), along(half(K, False))
    print(f"{task:26s} in fit {a[0]:6.2f}  unseen {b[0]:6.2f} (norm {b[1]:.2f})")
tools.reset_edit()

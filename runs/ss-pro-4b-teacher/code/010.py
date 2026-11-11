t = time.time(); K_dev, T_dev, S_dev = collect(dev); print(f"dev key pass: {time.time()-t:.0f} s | keys {tuple(K_dev.shape)}")
torch.save({"K": K_dev, "T": T_dev, "S": S_dev}, "runs/ss-pro-4b-teacher/keys_L29_dev.pt")
def delta_for(strength, margin):
    K, V, C = K_fit.double().cuda(), values_for(T_fit, S_fit, margin).double().cuda(), PROTECT.double().cuda()
    return torch.linalg.solve(strength * len(K) * C + K.T @ K, K.T @ V).T.float().cpu()      # [2560, 9216]
Gn = G / 1.0
def correction_quality(D, K, T, S):
    shift = (K @ D.T) @ G.T / S[:, None]                         # logit shift of each of the 12 candidate tokens
    idx = torch.tensor([WANT.index(t) for t in T.tolist()])
    top = shift.argmax(1)
    gain = shift[torch.arange(len(idx)), idx] - shift.mean(1)   # shift of the target token against the mean candidate
    digit = idx < 10
    return {"target is the most-raised token": round(float((top == idx).float().mean()), 3), "digits only": round(float((top == idx)[digit].float().mean()), 3), "mean gain of the target (nats)": round(float(gain.mean()), 2), "digits only gain": round(float(gain[digit].mean()), 2)}
for strength in (1.0, 0.1, 0.001):
    D = delta_for(strength, 3.0)
    print(f"strength {strength}: fit items", correction_quality(D, K_fit, T_fit, S_fit))
    print(f"              unseen items", correction_quality(D, K_dev, T_dev, S_dev), "| change on neutral text, mean norm of the added vector:", round(float((neutral[:2000] @ D.T).norm(dim=1).mean()), 3), "vs on GUI keys:", round(float((K_dev @ D.T).norm(dim=1).mean()), 3))
print("chance level for 'most-raised token' is about", round(1 / 12, 3))

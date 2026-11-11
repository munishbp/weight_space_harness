a = KEYS[name("A")][0][:5]
print("relative difference between two batch sizes:", float((k2 - a).norm() / a.norm()), "| cosine", [round(float(torch.nn.functional.cosine_similarity(k2[i], a[i], dim=0)), 4) for i in range(5)])
k3, _ = habit.step_keys([c for _, c in CTX[name("A")]][:16], MLP_OUT, LAST, PREFIX)
print("same batch composition:", float((k3 - KEYS[name("A")][0][:16]).abs().max()))
mb.close_all()

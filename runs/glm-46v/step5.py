# Attribution with the write norms of GLM in the layout, then the keys and the protection.
importlib.reload(layout); importlib.reload(edits); importlib.reload(attribution); importlib.reload(habit)
LAY = layout.find(model); print("write norm of the last MLP:", LAY["mlp_write_norms"][-1])
a = attribution.attribute_last([c for s, c in pick1 if s < 10], g, append_ids=[PREFIX] * 10)
mlp_share = {name: float(a["shares"][name].mean()) for name in LAY["mlps"]}
print("attribution of move minus close | total", round(float(a["total"].mean()), 3), "| error", round(a["error"], 4))
ranked = sorted(mlp_share.items(), key=lambda kv: kv[1])
print("MLPs, most negative:", [(n.split(".")[-2], round(v, 3)) for n, v in ranked[:6]], "| most positive:", [(n.split(".")[-2], round(v, 3)) for n, v in ranked[-4:]])
print("late MLPs:", [(n.split(".")[-2], round(mlp_share[n], 3)) for n in LAY["mlps"][28:]]); stamp("attribution done")

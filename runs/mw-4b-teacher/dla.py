# Which parts write `close` over `move` at the second turn, and do they differ between the press task and the grasp tasks?
g = attribution.token_direction(NAMES["move"], [NAMES["close"]])
DLA = {}
for task in ("button-press-topdown-v3", "pick-place-v3", "peg-insert-side-v3", "push-v3"):
    DLA[task] = attribution.attribute_last(CTX2[task], g, append_ids=[PREFIX] * len(CTX2[task]))
    print(f"{task:26s} logit(move) - logit(close): mean {float(DLA[task]['total'].mean()):+.2f} | split error {DLA[task]['error']:.4f}")
short = lambda name: name.replace("model.language_model.layers.", "L").replace(".linear_attn", " att").replace(".self_attn", " att").replace(".mlp", " mlp").replace("model.language_model.", "")
b, p = DLA["button-press-topdown-v3"]["shares"], DLA["pick-place-v3"]["shares"]
rows = sorted((float(b[n].mean()), float(p[n].mean()), float(DLA["peg-insert-side-v3"]["shares"][n].mean()), float(DLA["push-v3"]["shares"][n].mean()), n) for n in b)
print("parts that push hardest toward `close` in the button task (button | pick-place | peg | push):")
for vb, vp, vg, vu, n in rows[:10]: print(f"  {short(n):12s} {vb:+6.2f} | {vp:+6.2f} | {vg:+6.2f} | {vu:+6.2f}")
print("parts that push hardest toward `move` in the button task:")
for vb, vp, vg, vu, n in rows[-8:]: print(f"  {short(n):12s} {vb:+6.2f} | {vp:+6.2f} | {vg:+6.2f} | {vu:+6.2f}")
print("largest difference button minus pick-place:")
for vb, vp, vg, vu, n in sorted(rows, key=lambda r: r[0] - r[1])[-6:]: print(f"  {short(n):12s} {vb:+6.2f} | {vp:+6.2f}")

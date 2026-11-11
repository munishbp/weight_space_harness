# The change writes nothing at the action step of the other tasks. Find the prompt positions where it does write.
def all_positions(conv):
    """The input of the MLP output matrix at every position of one prompt with the action prefix appended."""
    inputs = tools.encode([conv], False, append_ids=[PREFIX])
    kept = {}
    hook = model.get_submodule(MLP_OUT).register_forward_pre_hook(lambda module, args: kept.update(k=args[0][0].float()))
    try: model(**inputs)
    finally: hook.remove()
    return inputs["input_ids"][0].tolist(), kept["k"]
for task in ("pick-place-v3", "peg-insert-side-v3", DOOR):
    ids, K = all_positions(CTX3[task][15])
    c = (K.double() @ D.T) @ gn                       # the amount of move-minus-open that the change adds at each position
    top = c.abs().topk(8).indices.tolist()
    print(f"{task}: {len(ids)} positions | sum of |c| {float(c.abs().sum()):.1f} | at the last position {float(c[-1]):.2f}")
    print("   largest:", [(p - len(ids), repr(tok.decode(ids[max(0, p - 2): p + 1])), round(float(c[p]), 2)) for p in sorted(top)])

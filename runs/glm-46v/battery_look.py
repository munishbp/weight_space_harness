# Why ChartQA and RefCOCO score 0 on GLM: the raw replies next to the targets.
for name in ("chartqa", "refcoco"):
    _, conversation, score, options, _ = battery.SETS[name]
    items = battery.items(name)[:3]
    replies = tools.generate([conversation(item) for item in items], **{k: v for k, v in options.items() if k in ("max_new_tokens", "thinking")})
    for item, reply in zip(items, replies):
        print(name, "| reply:", repr(reply["text"][:120]), "| ids start:", [tok.decode([t]) for t in reply["ids"][:6]], "| item keys:", {k: str(v)[:50] for k, v in item.items() if k not in ("image",)})

def dom(ep, tags=("input_text", "textarea", "span", "div")):
    out = []
    for e in ep.obs["dom_elements"]:
        if e["tag"] in ("input_text", "textarea") or (e["classes"] and any(k in e["classes"] for k in ("email-reply", "email-forward", "send", "icon", "email-sender"))):
            cx = round(float(e["left"][0] + e["width"][0] / 2) / ep.width * 1000); cy = round(float(e["top"][0] + e["height"][0] / 2) / ep.height * 1000)
            out.append((e["tag"], e["classes"][:22], (e["text"] or "")[:14], "val=" + repr(e["value"])[:20], "focus" if e["flags"][0] else "", [cx, cy]))
    return out
for r in [x for x in forced if x["task"] in ("email-inbox-reply", "email-inbox-forward")][0:40:20]:
    ep = mb.Episode(r["task"], r["seed"]); print("==", r["id"], ep.utterance)
    print("  start:", [d for d in dom(ep) if "email-sender" in d[1]][:4])
    for a in r["actions"]:
        if ep.ended: break
        ep.apply(a); print("  after", json.dumps(a)[:70], "| ended", ep.ended, ep.success); print("     ", dom(ep)[:8])
mb.close_all()

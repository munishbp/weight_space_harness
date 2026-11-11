def brief(a):
    if a["action"] == "click": return "c%s" % a["position"]
    if a["action"] == "type": return "t(%s)" % a["text"]
    if a["action"] == "key": return "k(%s)" % a["key"]
    return a["action"]
for task in ("enter-date", "enter-time", "form-sequence-2", "form-sequence-3", "email-inbox-forward", "email-inbox-reply", "social-media-some", "use-spinner", "number-checkboxes", "guess-number", "daily-calendar", "order-food", "choose-date", "form-sequence"):
    print("==", task)
    for r in [r for r in scan if r["task"] == task and not r["correct"]][:2]:
        print("  ", r["utterance"][:110].replace("\n", " "), "||", " ".join(brief(a) for a in r["actions"])[:150])
        if r["invalid"]: print("     invalid reply:", repr(r["replies"][0][:120]))

for task in ("order-food", "click-collapsible", "click-collapsible-2", "search-engine"):
    r = [r for r in scan if r["task"] == task and r["invalid"]][0]
    print("==", task, "| invalid replies:", [x[:90] for x, a in zip(r["replies"], r["actions"]) if a["action"] == "invalid"][:2])
for task in ("click-menu", "click-scroll-list", "click-shades", "count-shape", "generate-number", "read-table-2", "use-autocomplete", "phone-book", "search-engine", "email-inbox", "grid-coordinate", "social-media", "click-collapsible", "login-user-popup", "find-word", "email-inbox-noscroll", "email-inbox-star-reply", "social-media-all"):
    print("==", task)
    for r in [r for r in scan if r["task"] == task and not r["correct"]][:2]:
        print("  ", r["utterance"][:100].replace("\n", " "), "||", " ".join(brief(a) for a in r["actions"])[:150])

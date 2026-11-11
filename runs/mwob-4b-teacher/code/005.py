# Target portion: the email task types. The base model presses Enter after it types, where the page needs a click on the send button.
T0 = time.time(); tools.log("search_start", note="email portion of MiniWoB++: Enter after type")
stamp = lambda label: print(f"[{(time.time() - T0) / 60:5.1f} min] {label}", flush=True)
EMAIL = ["email-inbox-reply", "email-inbox-forward", "email-inbox", "email-inbox-noscroll", "email-inbox-star-reply"]
OTHER = [task for task in TASKS if task not in EMAIL]
PRACTICE = [(task, seed) for task in EMAIL for seed in range(20)]
HELDOUT = [(task, seed) for task in EMAIL for seed in range(100, 120)]
def brief(a):
    if a["action"] == "click": return "c"
    return {"type": "t", "key": "k", "done": "d", "invalid": "x"}[a["action"]]
acts = lambda r: " ".join(brief(a) for a in r["actions"])
base = episodes("base email practice", PRACTICE, f"{RUN}/base_practice.json")
for task in EMAIL:
    rows = [r for r in base if r["task"] == task]
    print(task, "| ok:", Counter(acts(r) for r in rows if r["correct"]).most_common(3), "| failed:", Counter(acts(r) for r in rows if not r["correct"]).most_common(4))
stamp("base practice done")

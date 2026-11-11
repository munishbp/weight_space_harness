# The base 27B model on the email practice episodes of MiniWoB++.
T0 = time.time(); stamp = lambda label: print(f"[{(time.time() - T0) / 60:5.1f} min] {label}", flush=True)
tools.log("search_start", note="27B on the MiniWoB++ email tasks")
base = episodes("27B base email practice", PRACTICE, f"{RUN}/mwob_base_practice.json")
for task in EMAIL:
    rows = [r for r in base if r["task"] == task]
    print(task, "| ok:", Counter(acts(r) for r in rows if r["correct"]).most_common(3), "| failed:", Counter(acts(r) for r in rows if not r["correct"]).most_common(4))
print("peak GPU memory: %.1f GB" % (torch.cuda.max_memory_allocated() / 2**30)); stamp("base practice done")

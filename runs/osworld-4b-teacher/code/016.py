# A one-token fault: OSWorld has 26 tasks that are not possible by design, and the correct reply there is `fail`.
# Does the student tell such a task from a possible one? Read the probability of `fail` at every recorded step.
ALL_APPS = ["chrome", "gimp", "libreoffice_calc", "libreoffice_impress", "libreoffice_writer", "multi_apps", "os", "thunderbird", "vlc", "vs_code"]
TASKS = {t["id"]: t | {"half": "practice" if i % 2 == 0 else "heldout"} for app in ALL_APPS for i, t in enumerate(ow.tasks(app))}
INFEASIBLE = {i for i, t in TASKS.items() if t["evaluator"].get("func") == "infeasible"}
print("infeasible tasks:", Counter((TASKS[i]["domain"], TASKS[i]["half"]) for i in INFEASIBLE))


def step_probs(results, label):
    """The probability of each action name at each recorded step of the given results."""
    steps = [(r, s) for r in results if "error" not in r for s in range(len(r["actions"]))]
    probs = []
    for start in range(0, len(steps), 8):
        batch = [ow.conversation(r["instruction"], Image.open(f"{SHOTS}/{label}/{r['id']}-{s}.png").convert("RGB"), r["actions"][:s]) for r, s in steps[start:start + 8]]
        probs.append(tools.read(batch, [LAST], append_ids=[PREFIX] * len(batch), logprobs_of=list(FIRST.values()))["logprobs"].exp().float().cpu())
    return steps, torch.cat(probs)


def fail_table(label, steps, probs, half="practice"):
    by_task = defaultdict(list)
    for (r, s), p in zip(steps, probs): by_task[r["id"]].append(float(p[NAMES.index("fail")]))
    rows = [(i in INFEASIBLE, ps[0], max(ps), sum(ps) / len(ps), i) for i, ps in by_task.items() if TASKS[i]["half"] == half]
    for kind in (True, False):
        part = [r for r in rows if r[0] == kind]
        if part: print(f"{label} {half} {'infeasible' if kind else 'feasible  '} n={len(part):2d} | p(fail) step 0: median {sorted(r[1] for r in part)[len(part) // 2]:.5f} | max over steps: median {sorted(r[2] for r in part)[len(part) // 2]:.5f}, values {[round(r[2], 4) for r in sorted(part, key=lambda r: -r[2])[:8]]}")
    return rows


scan = json.load(open(f"{RUN}/scan.json"))
scan_steps, scan_probs = step_probs([r for r in scan if TASKS[r["id"]]["half"] == "practice"], "scan")
torch.save({"steps": [(r["id"], s) for r, s in scan_steps], "probs": scan_probs, "names": NAMES}, f"{RUN}/scan_step_probs.pt")
scan_rows = fail_table("scan", scan_steps, scan_probs); stamp("fail probabilities on the scan done")

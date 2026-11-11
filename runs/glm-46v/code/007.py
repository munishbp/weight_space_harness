# The pick-place fault on GLM: probabilities at step 1, attribution, keys, protection, and a margin sweep.
importlib.reload(habit)
PICK = "pick-place-v3"; LAST = LAY["layers"][-1]
by_task = defaultdict(list)
for r in base_practice: by_task[r["task"]].append(r)
forced_by = defaultdict(list)
for r in forced_all: forced_by[r["task"]].append(r)
g = attribution.token_direction(NAMES["move"], [NAMES["close"]])
pick1 = [(r["seed"], mw.replay(r, 1)) for r in by_task[PICK]]
wrote_close = torch.tensor([r["primitives"][1].get("action") == "close" for r in by_task[PICK]])
p = probs([c for _, c in pick1])
print("base, pick-place step 1 | episodes that wrote close:", int(wrote_close.sum()), fmt(p[wrote_close]), "| episodes that wrote move:", fmt(p[~wrote_close]))
a = attribution.attribute_last([c for s, c in pick1 if s < 10], g, append_ids=[PREFIX] * 10)
mlp_share = {name: float(a["shares"][name].mean()) for name in LAY["mlps"]}
print("attribution of move minus close | total", round(float(a["total"].mean()), 3), "| error", round(a["error"], 4))
print("MLPs, most negative:", [(n.split(".")[-2], round(v, 3)) for n, v in sorted(mlp_share.items(), key=lambda kv: kv[1])[:6]]); stamp("attribution done")
# Other steps: step 1 of the other tasks, and steps 2 and 3 of all tasks. For pick-place the later steps come from the forced replay.
other1 = [mw.replay(r, 1) for t in TASKS if t != PICK for r in by_task[t] if r["seed"] < 10 and len(r["primitives"]) > 1]
later = [mw.replay(r, k) for t in TASKS for r in (forced_by[t] if t == PICK else by_task[t]) if r["seed"] < 10 for k in (2, 3) if len(r["primitives"]) > k]
print("step 1 of other tasks:", len(other1), "| later steps:", len(later)); stamp("contexts done")
ids_of = lambda prim: tok.encode(json.dumps(prim), add_special_tokens=False)
prompts = []
for t in TASKS:
    for r in (forced_by[t] if t == PICK else by_task[t]):
        if r["seed"] < 5:
            for k in range(min(4, len(r["primitives"]))):
                reply = ids_of(r["primitives"][k]); assert reply[:len(PREFIX)] == PREFIX
                prompts.append((mw.replay(r, k), reply, len(PREFIX) - 1 if (t == PICK and k == 1) else None))
NEUTRAL = battery.neutral_text(64, 256)
stamp(f"{len(prompts)} prompts for the protection")

import time
t0 = time.time()
# The step-2 prompt of every practice episode: the history holds the first move of the student.
STEP2 = {}
for task in TASKS:
    for r in base:
        if r["task"] == task and r["primitives"] and r["primitives"][0]["action"] == "move":
            STEP2[(task, r["seed"])] = (step_conversation(task, r["seed"], first_move(r)), r["replies"][1] if len(r["replies"]) > 1 else None)
print("step-2 prompts", len(STEP2), round(time.time() - t0), "s", flush=True)

g = attribution.token_direction(MOVE, [CLOSE])
FIT_SEEDS, READ_SEEDS = range(0, 10), range(10, 20)
target_fit = [STEP2[(BUTTON, s)][0] for s in FIT_SEEDS]

att = attribution.attribute_last(target_fit, g, append_ids=[PREFIX] * len(target_fit))
mlp_shares = {name: float(v.mean()) for name, v in att["shares"].items() if name.endswith(".mlp")}
print("total move-close", round(float(att["total"].mean()), 2), "error", round(att["error"], 3))
print("most negative MLPs", sorted(mlp_shares.items(), key=lambda kv: kv[1])[:6])
late = {n: v for n, v in mlp_shares.items() if int(n.split(".")[-2]) >= 24}
LAYER = min(late, key=late.get)
MLP_OUT = LAYER + ".down_proj"
print("chosen", MLP_OUT, flush=True)

keys, scale = habit.step_keys(target_fit, MLP_OUT, LAST, PREFIX)
others = [STEP2[(t, s)][0] for t in TASKS if t != BUTTON for s in FIT_SEEDS if (t, s) in STEP2]
keep, _ = habit.step_keys(others, MLP_OUT, LAST, PREFIX)

# Protection: all positions of step-2 prompts with the reply of the student, minus the target step.
prompts = []
for t in TASKS:
    for s in range(5):
        conv, reply = STEP2.get((t, s), (None, None))
        if conv is None or reply is None:
            continue
        ids = tok.encode(reply.strip(), add_special_tokens=False)
        prompts.append((conv, ids, 3 if t == BUTTON else None))
NEUTRAL = battery.neutral_text(64, 256)
protect = habit.protection(prompts, NEUTRAL, MLP_OUT)
ORIGINAL_W = tools.get_weight(MLP_OUT + ".weight")
print("keys", tuple(keys.shape), "keep", tuple(keep.shape), "protect", tuple(protect.shape), round(time.time() - t0), "s", flush=True)

read_target = [STEP2[(BUTTON, s)][0] for s in READ_SEEDS]
read_other = {t: [STEP2[(t, s)][0] for s in READ_SEEDS if (t, s) in STEP2] for t in TASKS if t != BUTTON}


def probe(tag):
    p = tools.read(read_target, [LAST], append_ids=[PREFIX] * len(read_target), logprobs_of=[MOVE, CLOSE])["logprobs"].exp()
    line = {"button p_move": round(float(p[:, 0].mean()), 3), "min": round(float(p[:, 0].min()), 3)}
    for t, convs in read_other.items():
        q = tools.read(convs, [LAST], append_ids=[PREFIX] * len(convs), logprobs_of=[MOVE, CLOSE])["logprobs"].exp()
        line[t.split("-v3")[0]] = round(float(q[:, 0].mean()), 3)
    print(tag, line, flush=True)


probe("base")
stray_conv, stray_reply = STEP2[("pick-place-v3", 15)]
stray_ids = tok.encode(stray_reply.strip(), add_special_tokens=False)
for margin in [4.0, 8.0, 12.0]:
    entry = habit.fit(MLP_OUT, [(keys, scale, g)], keep, protect, margin=margin)
    sw = habit.stray_write(stray_conv, stray_ids, MLP_OUT, ORIGINAL_W, g)
    intended = float(((keys.to(ORIGINAL_W.device).double() @ (tools.get_weight(MLP_OUT + ".weight") - ORIGINAL_W).double().T) @ (g / g.norm()).double()).mean())
    print("margin", margin, "rel_delta", entry.get("relative_delta_norm", entry), "stray", sw, "intended per target", round(intended, 3))
    probe(f"margin {margin}")
    tools.reset_edit()
print(round(time.time() - t0), "s")

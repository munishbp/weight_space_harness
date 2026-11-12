# The probability of each action name at each recorded step, from the saved screenshots (no virtual machine).
# A repeat step is a step where the student writes the same action as in the step before.
from PIL import Image

NAMES = ["click", "double_click", "right_click", "drag", "scroll", "type", "key", "wait", "done", "fail"]
PREFIX = tok.encode('{"action": "', add_special_tokens=False)
FIRST = {n: tok.encode(n, add_special_tokens=False)[0] for n in NAMES}
print("prefix", PREFIX, "first tokens", {n: (i, tok.decode([i])) for n, i in FIRST.items()})
assert len(set(FIRST.values())) == len(NAMES), "Two action names share a first token."
LAST = parts["layers"][-1]


def step_prompt(result, step, label="sample"):
    """The prompt that the student saw at one step of a recorded task."""
    shot = Image.open(f"{SHOTS}/{label}/{result['id']}-{step}.png").convert("RGB")
    return ow.conversation(result["instruction"], shot, result["actions"][:step])


def is_repeat(result, step):
    return step >= 1 and result["actions"][step] == result["actions"][step - 1]


sample = json.load(open(f"{RUN}/sample.json"))
steps = [(r, s) for r in sample if "error" not in r for s in range(len(r["actions"]))]
convs = [step_prompt(r, s) for r, s in steps]
probs = []
for start in range(0, len(convs), 8):
    batch = convs[start:start + 8]
    out = tools.read(batch, [LAST], append_ids=[PREFIX] * len(batch), logprobs_of=list(FIRST.values()))
    probs.append(out["logprobs"].exp().float().cpu())
probs = torch.cat(probs)
torch.save({"steps": [(r["id"], s) for r, s in steps], "probs": probs, "names": NAMES}, f"{RUN}/step_probs.pt")

repeat = torch.tensor([is_repeat(r, s) for r, s in steps])
first_repeat = torch.tensor([is_repeat(r, s) and not is_repeat(r, s - 1) for r, s in steps])
for label, mask in [("steps that are not a repeat", ~repeat), ("first repeat of a run", first_repeat), ("all repeat steps", repeat)]:
    mean = probs[mask].mean(0)
    print(f"{label:28s} n={int(mask.sum()):3d} | " + " ".join(f"{n} {float(m):.3f}" for n, m in zip(NAMES, mean) if m > 0.002))
done_at_repeat = probs[repeat][:, NAMES.index("done")]
print("p(done) at repeat steps: quantiles", [round(float(q), 4) for q in torch.quantile(done_at_repeat, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))])
stamp("step probabilities done")

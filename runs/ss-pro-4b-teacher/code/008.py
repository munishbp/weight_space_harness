exec(open("runs/ss-pro-4b-teacher/setup.py").read())
from wsp import battery
rec_quick = torch.load("runs/ss-pro-4b-teacher/recorded_quick.pt")["rec"]
start = max(i for i, t in enumerate(rec_quick[0]["ids"]) if tok.decode([t]) == " [") + 1
PREFIX = rec_quick[0]["ids"][:start]; SPACE = tok.encode(" ", add_special_tokens=False)[0]
quick_ids = {i["id"] for i in quick}
fit = [i for i in practice if i["id"] not in quick_ids]; dev = quick
print("fit items:", len(fit), "| dev items:", len(dev), "| overlap:", len({i["id"] for i in fit} & quick_ids))
MLP_OUT = LAY["mlp_outputs"][29]; LAST = LAY["layers"][31]

def forced_reply(it):
    """The tokens of a correct reply (a click on the centre of the target) and the next token at each position."""
    (xl, xh), (yl, yh) = target_ranges(it)
    cx, cy = round((xl + xh) / 2), round((yl + yh) / 2)
    seq = [PREFIX[-1]] + [15 + int(d) for d in str(cx)] + [COMMA, SPACE] + [15 + int(d) for d in str(cy)]
    nxt = seq[1:] + [CLOSE]
    return PREFIX[:-1] + seq, len(seq), nxt

def collect(its):
    keys, targets, scales = [], [], []
    for it in its:
        ids, k, nxt = forced_reply(it)
        a = tools.read([conv_with()(it)], [MLP_OUT, LAST], append_ids=[ids], last_positions=k, inputs=True)
        act = a[MLP_OUT][0]; res = a[LAST][0]                     # [k, 9216] neuron activations, [k, 2560] state before the last layer
        for p in range(k):
            if nxt[p] == SPACE: continue                           # the space after the comma is trivial
            keys.append(act[p]); targets.append(nxt[p]); scales.append(float(res[p].pow(2).mean().sqrt()))
    return torch.stack(keys), torch.tensor(targets), torch.tensor(scales)
t = time.time(); K_fit, T_fit, S_fit = collect(fit); print(f"key pass: {time.time()-t:.0f} s | keys {tuple(K_fit.shape)} | targets:", Counter(tok.decode([x]) for x in T_fit.tolist()).most_common(12))
# The protect matrix: the second moment of the same neuron activations on neutral text.
text = battery.neutral_text(64, 256)
neutral = torch.cat(tools.read_tokens(text, [MLP_OUT], inputs=True)[MLP_OUT])
PROTECT = (neutral.double().T @ neutral.double() / len(neutral)).float()
print("neutral activations:", tuple(neutral.shape), "| mean square of a key on GUI digits:", round(float(K_fit.pow(2).mean()), 4), "| on neutral text:", round(float(neutral.pow(2).mean()), 4), "| state rms:", round(float(S_fit.mean()), 3))
torch.save({"K": K_fit, "T": T_fit, "S": S_fit, "PROTECT": PROTECT, "fit": [i["id"] for i in fit]}, "runs/ss-pro-4b-teacher/keys_L29.pt")

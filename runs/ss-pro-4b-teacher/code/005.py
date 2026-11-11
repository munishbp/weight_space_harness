exec(open("runs/ss-pro-4b-teacher/setup.py").read())
rec_quick = torch.load("runs/ss-pro-4b-teacher/recorded_quick.pt")["rec"]
start = max(i for i, t in enumerate(rec_quick[0]["ids"]) if tok.decode([t]) == " [") + 1
PREFIX = rec_quick[0]["ids"][:start]
# Check the forced prefix: the read must reproduce the digit probabilities that generation recorded.
chk = tools.read([conv_with()(by[r["id"]]) for r in rec_quick[:4]], [], append_ids=[PREFIX] * 4, logprobs_of=DIG)["logprobs"]
ref = torch.stack([r["lp"][start][:10] for r in rec_quick[:4]])
print("forced-prefix read against generation: largest probability difference", round(float((chk.exp() - ref.exp()).abs().max()), 4))

import json, random
from wsp import evaluate
from wsp.bench import boxed_math
rows = boxed_math.load_jsonl("data/lm_understand-logs/exp8_math/practice/gsm8k_train.jsonl")
random.Random(0).shuffle(rows)
practice = rows[1000:1064]
base_gen = json.load(open("runs/calib-08b-teacher/base_gen_64.json"))["base_gen"]
tok = processor.tokenizer
THINK_END = tok.convert_tokens_to_ids("</think>")
WAIT = [3655, 11158, 13428, 13784]
# Full token sequences: prompt plus reply, for the replies that never closed their thinking.
seqs, offs = [], []
for p, r in zip(practice, base_gen):
    if THINK_END in r["ids"]: continue
    pre = tools.prompt_ids(boxed_math.conversation(p), thinking=True)
    seqs.append(pre + r["ids"]); offs.append(len(pre))
print(len(seqs), "stuck replies | prompt length example:", offs[0], "| total length:", len(seqs[0]))
print("prompt tail:", repr(tok.decode(seqs[0][offs[0]-12:offs[0]])))

# Sanity test of the new read tools on two short sequences.
a = tools.read_tokens([seqs[0][:50], seqs[1][:80]], ["model.language_model.layers.5", "model.language_model.layers.5.mlp.down_proj"], positions=[[0, 49], [79]])
print({k: [tuple(x.shape) for x in v] for k, v in a.items()})
b = tools.read_tokens([seqs[0][:50]], ["model.language_model.layers.5.mlp.down_proj"], inputs=True)
print("down_proj input:", tuple(b["model.language_model.layers.5.mlp.down_proj"][0].shape))
full = tools.read_tokens([seqs[0][:50]], ["model.language_model.layers.5"])["model.language_model.layers.5"][0]
print("padding does not change the result:", torch.allclose(full[[0, 49]], a["model.language_model.layers.5"][0], atol=2e-2), float((full[[0, 49]] - a["model.language_model.layers.5"][0]).abs().max()))
lp = tools.read_logprobs([seqs[0][:offs[0] + 40]], [THINK_END] + WAIT)
print("logprob shape:", tuple(lp[0].shape), "| P(</think>) over the first 40 reply tokens: max", float(lp[0][offs[0]:, 0].exp().max()))
# The next-token prediction at the last prompt position should favour the first reply token.
nxt = tools.read_logprobs([seqs[0][:offs[0]]], [seqs[0][offs[0]]], positions=[[offs[0] - 1]])
print("P(first reply token | prompt) =", round(float(nxt[0].exp()), 3), repr(tok.decode([seqs[0][offs[0]]])))

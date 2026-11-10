import re
from collections import Counter
tok = processor.tokenizer
THINK_END = tok.convert_tokens_to_ids("</think>")
print("</think> id:", THINK_END, "| appears in", sum(THINK_END in r["ids"] for r in base_gen), "of", len(base_gen), "replies")
wait_ids = sorted({i for w in ["Wait", " Wait", "wait", " wait"] for i in tok.encode(w, add_special_tokens=False)})
print("wait ids:", wait_ids, [tok.decode([i]) for i in wait_ids])
counts = [sum(t in wait_ids for t in r["ids"]) for r in base_gen]
print("Wait tokens per reply: mean", sum(counts)/len(counts), "| median", sorted(counts)[len(counts)//2], "| zero in", sum(c == 0 for c in counts))
# Where is the first statement of an answer, and how much text follows it?
first_answer = []
for r in base_gen:
    text = tok.decode(r["ids"])
    m = re.search(r"(?i)final answer|\\boxed|the answer is|answer:", text)
    first_answer.append(len(tok.encode(text[:m.start()], add_special_tokens=False)) if m else None)
have = [f for f in first_answer if f is not None]
print("replies that state an answer inside the thinking:", len(have), "of", len(base_gen), "| median position of the first statement:", sorted(have)[len(have)//2] if have else None)
# The most common tokens that follow a 'Wait' and the most common line starts.
after = Counter(); starts = Counter()
for r in base_gen:
    ids = r["ids"]
    for i, t in enumerate(ids[:-1]):
        if t in wait_ids: after[tok.decode([ids[i+1]])] += 1
    for line in tok.decode(ids).split("\n"):
        w = line.strip(" *-").split(" ")[0][:14]
        if w: starts[w] += 1
print("after Wait:", after.most_common(6))
print("line starts:", starts.most_common(14))
# Wait rate over the course of a reply, in blocks of 256 tokens.
blocks = Counter(); totals = Counter()
for r in base_gen:
    for i, t in enumerate(r["ids"]):
        totals[i // 256] += 1; blocks[i // 256] += t in wait_ids
print("Wait per 1000 tokens by block of 256:", [round(1000 * blocks[b] / totals[b], 1) for b in sorted(totals)])

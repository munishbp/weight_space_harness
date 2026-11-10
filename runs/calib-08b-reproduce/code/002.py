import json, random, re, time
rows = [json.loads(l) for l in open("data/lm_understand-logs/exp8_math/practice/gsm8k_train.jsonl")]
random.Random(0).shuffle(rows)
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."

class PresencePenalty:
    """The blog's evaluation uses presence penalty 1.5: each token already written loses 1.5 logits."""
    def __init__(self, penalty, prompt_len): self.penalty, self.prompt_len, self.seen = penalty, prompt_len, None
    def __call__(self, input_ids, scores):
        if self.seen is None: self.seen = torch.zeros_like(scores, dtype=torch.bool)
        if input_ids.shape[1] > self.prompt_len: self.seen.scatter_(1, input_ids[:, -1:], True)
        return scores - self.penalty * self.seen

def number(s):
    s = s.replace(",", "").replace("$", "").replace("\\%", "").replace("%", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None

def grade(text, answer):
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    return bool(boxed) and number(boxed[-1]) is not None and number(boxed[-1]) == number(answer)

def run_math(problems, cap, seed=0, batch=64):
    out = []
    for i in range(0, len(problems), batch):
        chunk = problems[i:i + batch]
        convs = [[{"role": "user", "content": [{"type": "text", "text": p["problem"] + SUFFIX}]}] for p in chunk]
        prompt_len = tools.encode(convs, thinking=True)["input_ids"].shape[1]
        torch.manual_seed(seed + i)
        replies = tools.generate(convs, max_new_tokens=cap, thinking=True, do_sample=True, temperature=0.6,
                                 top_p=0.95, top_k=20, logits_processor=[PresencePenalty(1.5, prompt_len)])
        for p, r in zip(chunk, replies):
            out.append({"id": p["id"], "correct": grade(r["text"], p["answer"]), **r})
    return out

t = time.time(); probe = run_math(rows[:16], cap=512, batch=16); dt = time.time() - t
print(f"probe: 16 problems, cap 512, {dt:.1f} s, {sum(r['new_tokens'] for r in probe)/dt:.0f} tokens/s total")
print("hit limit:", sum(r["hit_limit"] for r in probe), "correct:", sum(r["correct"] for r in probe), "(edited model)")
print("--- sample reply tail ---"); print(probe[0]["text"][-400:])

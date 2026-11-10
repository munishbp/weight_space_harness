import hashlib, json, random, time, numpy as np
from wsp import evaluate
from wsp.bench import boxed_math
man = json.load(open("data/lm_understand-logs/exp8_math/runs/p50c/edit/manifest.json"))
def sha(name):
    bits = model.get_parameter(name).detach().cpu().contiguous().view(torch.int16).numpy()
    return hashlib.sha256(np.ascontiguousarray(bits).tobytes()).hexdigest()
entries = tools.load_edit("runs/calib-08b-reproduce/edit/edit.safetensors")
assert all(sha(t["name"]) == t["edited_sha256"] for t in man["tensors"]); print("edit reloaded from the run directory, bit-exact")
rows = boxed_math.load_jsonl("data/lm_understand-logs/exp8_math/practice/gsm8k_train.jsonl")
random.Random(0).shuffle(rows)
t = time.time()
probe = evaluate.run(rows[:16], boxed_math.conversation, boxed_math.score, batch_size=16, max_new_tokens=512, **boxed_math.BLOG_SAMPLING)
dt = time.time() - t
print(f"probe: {dt:.1f} s, {sum(r['new_tokens'] for r in probe)/dt:.0f} tokens/s (before: 14.8 s, 554 tokens/s)")
print(evaluate.summarize(probe))
print(torch.cuda.memory_allocated() / 1e9, "GB allocated")

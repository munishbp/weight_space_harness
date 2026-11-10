import importlib, random, time
importlib.reload(boxed_math)
rows = boxed_math.load_jsonl("data/lm_understand-logs/exp8_math/practice/gsm8k_train.jsonl")
print(len(rows), "practice problems")
random.Random(0).shuffle(rows)
t = time.time()
probe = evaluate.run(rows[:16], boxed_math.conversation, boxed_math.score, batch_size=16, max_new_tokens=512, **boxed_math.BLOG_SAMPLING)
dt = time.time() - t
print(f"probe: {dt:.1f} s, {sum(r['new_tokens'] for r in probe)/dt:.0f} tokens/s (before the fast kernel: 14.8 s, 554 tokens/s)")
print(evaluate.summarize(probe))
print(round(torch.cuda.memory_allocated() / 1e9, 2), "GB allocated")

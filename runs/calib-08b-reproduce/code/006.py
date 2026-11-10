import time
t = time.time()
probe = evaluate.run(rows[:16], boxed_math.conversation, boxed_math.score, batch_size=16, max_new_tokens=512, **boxed_math.BLOG_SAMPLING)
dt = time.time() - t
print(f"second probe: {dt:.1f} s, {sum(r['new_tokens'] for r in probe)/dt:.0f} tokens/s")

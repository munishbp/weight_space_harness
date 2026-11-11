import time, random
from functools import partial
from wsp import evaluate
from wsp.bench import screenspot_pro as sp
torch.cuda.empty_cache()
items, images = sp.load()
conv = partial(sp.conversation, images=images)
probe = random.Random(0).sample(items, 16)
for thinking in (False, True):
    t = time.time()
    res = evaluate.run(probe, conv, sp.score, batch_size=4, max_new_tokens=512 if thinking else 64, thinking=thinking)
    print(f"thinking={thinking}: {time.time()-t:.0f} s", evaluate.summarize(res)["all"], "| no point:", sum(r["point"] is None for r in res))
    print("   sample reply:", repr(res[0]["text"][-200:]))
print("GPU GB allocated:", round(torch.cuda.max_memory_allocated() / 1e9, 1))

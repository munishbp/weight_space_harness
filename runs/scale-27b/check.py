# First check of the packed 27B model: memory, a coherent reply, and the speed of one forward pass.
import json, time
from wsp import layout
print("load report:", model.packed_load_report)
print("GPU memory after load: %.1f GB" % (torch.cuda.memory_allocated() / 2**30))
LAY = layout.find(model)
print("layers:", len(LAY["layers"]) if isinstance(LAY, dict) and "layers" in LAY else LAY)
ask = lambda text: [{"role": "user", "content": [{"type": "text", "text": text}]}]
t = time.time()
out = tools.generate([ask("What is the capital of France? Answer in one word."), ask("What is 17 * 23? Give only the number.")], max_new_tokens=24)
print("2 replies in %.1f s:" % (time.time() - t), [r["text"] for r in out], [r["new_tokens"] for r in out])
t = time.time()
out = tools.generate([ask("Write three sentences about how a bicycle stays upright.")], max_new_tokens=96)
dt = time.time() - t
print("%d tokens in %.1f s = %.2f tokens per second" % (out[0]["new_tokens"], dt, out[0]["new_tokens"] / dt))
print(out[0]["text"])
print("peak GPU memory: %.1f GB" % (torch.cuda.max_memory_allocated() / 2**30))

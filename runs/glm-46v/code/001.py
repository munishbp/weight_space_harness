# Level 1 for a student outside the Qwen family: each harness tool runs on GLM-4.6V-Flash.
import json, time
from wsp import attribution, battery, edits, layout
from wsp.bench import metaworld_text as mw
RUN = "runs/glm-46v"; tok = processor.tokenizer
print("class:", type(model).__name__, "| GPU memory after load: %.1f GB" % (torch.cuda.memory_allocated() / 2**30))
LAY = layout.find(model)
print("layers:", len(LAY["layers"]), "| first MLP output:", LAY["mlp_outputs"][0], "| last layer:", LAY["layers"][-1], "| final norm:", LAY["final_norm"])
ask = lambda text: [{"role": "user", "content": [{"type": "text", "text": text}]}]
t = time.time()
out = tools.generate([ask("What is the capital of France? Answer in one word."), ask("What is 17 * 23? Give only the number.")], max_new_tokens=24)
print("generate, text: %.1f s" % (time.time() - t), [r["text"] for r in out], [r["new_tokens"] for r in out])
out = tools.generate([ask("What is 17 * 23? Give only the number.")], max_new_tokens=200, thinking=True)
print("generate, thinking on:", repr(out[0]["text"][:160]), out[0]["new_tokens"])
ep = mw.Episode("button-press-topdown-v3", 0); conv = mw.conversation(ep); ep.close()
t = time.time(); out = tools.generate([conv], max_new_tokens=48); dt = time.time() - t
print("generate, image: %.1f s, %d tokens:" % (dt, out[0]["new_tokens"]), repr(out[0]["text"]))
mlp_out = LAY["mlp_outputs"][-4]; last = LAY["layers"][-1]
r = tools.read([conv], [mlp_out, last]); print("read:", {k: tuple(v.shape) for k, v in r.items()})
r = tools.read([conv], [mlp_out], inputs=True); print("read inputs:", tuple(r[mlp_out].shape))
ids = tok.encode(" Paris", add_special_tokens=False); ids2 = tok.encode(" London", add_special_tokens=False)
g = attribution.token_direction(ids[0], [ids2[0]]); print("token_direction:", tuple(g.shape), round(float(g.norm()), 3))
lp = tools.read([ask("The capital of France is a city. Name it in one word.")], [], logprobs_of=[ids[0], ids2[0]])["logprobs"]
print("logprobs of ' Paris' and ' London' as the first reply token:", [round(float(x), 2) for x in lp[0]])
name = mlp_out + ".weight"; W0 = tools.get_weight(name).clone()
neutral = battery.neutral_text(16, 256)
e = tools.save_edit(name, W0 * 1.01)
print("edit:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in e.items() if not isinstance(v, (list, dict))})
manifest = json.load(open(f"{RUN}/edit/manifest.json")); print("manifest:", {k: v for k, v in manifest.items() if k != "tensors"})
print("deviation with the edit:", {k: round(v, 4) for k, v in tools.deviation(neutral).items()})
tools.reset_edit(); print("reset is exact:", bool(torch.equal(tools.get_weight(name), W0)))
print("deviation after the reset:", {k: round(v, 4) for k, v in tools.deviation(neutral).items()})
print("peak GPU memory: %.1f GB" % (torch.cuda.max_memory_allocated() / 2**30))

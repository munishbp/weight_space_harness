# One task, to measure the time per step and the memory of the 27B student with a full-size screenshot.
t = time.time(); one = episodes("smoke", practice_of("vs_code")[:1], f"{RUN}/smoke.json")
steps = len(one[0]["actions"]); print("steps", steps, "| seconds", one[0]["seconds"], "| score", one[0].get("score"), "| peak GPU GB", round(torch.cuda.max_memory_allocated() / 2**30, 1))
print([json.dumps(a)[:70] for a in one[0]["actions"]])

# The base 27B model on the Meta-World practice episodes: a short pass first for the speed, then the rest.
t = time.time(); first = mw.run_episodes(PRACTICE[::20], gen, batch_size=10)
print("10 episodes (one per task) in %.0f s:" % (time.time() - t), {r["task"].split("-v3")[0]: (int(r["correct"]), len(r["primitives"])) for r in first}, flush=True)
print("peak GPU memory: %.1f GB" % (torch.cuda.max_memory_allocated() / 2**30), flush=True)
base = episodes("base practice", PRACTICE, f"{RUN}/base_practice.json")

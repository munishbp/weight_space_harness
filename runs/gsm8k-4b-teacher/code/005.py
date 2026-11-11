# After a kernel restart: load the saved baseline in place of a new run, and check the token read on more than eight rows.
saved = json.load(open(f"{RUN}/base_gen_64.json")); base_gen, off_gen = saved["base"], saved["off"]
print("loaded:", behaviour(base_gen))
check = tools.read_tokens([[1, 2, 3, 4]] * 5 + [[5, 6]] * 6, [LAY["layers"][-1]])
print("read_tokens rows:", [tuple(t.shape) for t in check[LAY["layers"][-1]]])
T0 = time.time(); stamp = lambda text: print(f"[{(time.time() - T0) / 60:5.1f} min] {text}", flush=True)

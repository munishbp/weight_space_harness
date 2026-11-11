# Where does the model click on icon items, and where are the targets? First digit of x and of y, on the fit items.
def first_digit(v): return int(str(max(0, min(999, round(v))))[0]) if round(v) >= 100 else -1     # -1 marks a value below 100
def hist(vals): c = Counter(vals); return [c.get(d, 0) for d in [-1] + list(range(1, 10))]
mx = [first_digit(base_res[i["id"]]["point"][0] * 1000) for i in fit]; my = [first_digit(base_res[i["id"]]["point"][1] * 1000) for i in fit]
tx = [first_digit(sum(target_ranges(i)[0]) / 2) for i in fit]; ty = [first_digit(sum(target_ranges(i)[1]) / 2) for i in fit]
print("hundreds band of x (first entry = below 100, then 100s..900s)")
print("  model clicks:", hist(mx)); print("  targets     :", hist(tx))
print("hundreds band of y"); print("  model clicks:", hist(my)); print("  targets     :", hist(ty))
# Simulate a prior shift on the first x digit of unseen items: add gamma * log(target share / model share) for each digit token.
# A value below 100 starts with any digit, so use the first written digit of the coordinate string.
fd = lambda v: int(str(max(0, min(999, round(v))))[0])
mc = Counter(fd(base_res[i["id"]]["point"][0] * 1000) for i in fit); tc = Counter(fd(sum(target_ranges(i)[0]) / 2) for i in fit)
bias = torch.tensor([math.log((tc.get(d, 0) + 2) / (mc.get(d, 0) + 2)) for d in range(10)])
print("first written digit of x, model:", [mc.get(d, 0) for d in range(10)], "| targets:", [tc.get(d, 0) for d in range(10)])
for gamma in (0.5, 1.0, 2.0):
    new_ok = ok0(lp0 + gamma * bias)
    print(f"  prior shift x{gamma}: in range {int(new_ok.sum())} of {len(dev)} (base {int(base_ok.sum())}) | fixed {int((new_ok & ~base_ok).sum())} | broken {int((~new_ok & base_ok).sum())}")
led = [json.loads(l) for l in open("runs/ss-pro-4b-teacher/ledger.jsonl")]
t0 = next(x["t"] for x in led if x["kind"] == "teacher_start")
print("teacher time so far:", round((time.time() - t0) / 60), "min | generate batches:", sum(x["kind"] == "generate" for x in led), "| read passes:", sum(x["kind"] == "read" for x in led), "| trials:", sum(x["kind"] == "trial" for x in led), "| code runs:", sum(x["kind"] == "python" for x in led))

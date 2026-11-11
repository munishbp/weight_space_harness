print("base held-out per task:", {t: table(base_held)[t]["success"] for t in EMAIL}, "| invalid", {t: table(base_held)[t]["invalid"] for t in EMAIL})
print("edit held-out, email-inbox failures:", Counter(acts(r) for r in edit_held if r["task"] == "email-inbox" and not r["correct"]).most_common(4))
# Test of wsp/habit.py: the same fit through the module gives the saved edit.
import importlib
from wsp import habit; importlib.reload(habit)
tools.reset_edit(); W0 = tools.get_weight(MLP_OUT + ".weight").clone()
tools.load_edit(f"{RUN}/edit_email_v3_L28_m12.safetensors"); W_saved = tools.get_weight(MLP_OUT + ".weight").clone(); tools.reset_edit()
name = lambda s: [n for n in CTX if n.startswith(s)][0]
targets = [(KEYS[name("A")][0], KEYS[name("A")][1], g_type), (KEYS[name("B")][0], KEYS[name("B")][1], g_type), (KEYS[name("C")][0], KEYS[name("C")][1], g_key)]
keep_keys = torch.cat([KEYS[name("T")][0], K_KEEP])
e = habit.fit(MLP_OUT, targets, keep_keys, P_ALL, 12.0)
W_new = tools.get_weight(MLP_OUT + ".weight")
print("module fit equals the saved edit:", bool((W_new == W_saved).all()), "| largest difference", float((W_new - W_saved).abs().max()))
sv = torch.linalg.svdvals((W_saved - W0).float().to(model.device)); print("rank of the change:", int((sv > 0.01 * sv[0]).sum()))
k2, s2 = habit.step_keys([c for _, c in CTX[name("A")]][:5], MLP_OUT, LAST, PREFIX)
print("step_keys equals the run keys:", bool(torch.allclose(k2, KEYS[name("A")][0][:5], atol=1e-3)), tuple(k2.shape))
r = [x for x in scan if x["task"] == "click-pie"][0]; ids = tok.encode(r["replies"][1].strip(), add_special_tokens=False)
print("stray write on a click-pie prompt:", {k: round(v, 2) if isinstance(v, float) else v for k, v in habit.stray_write(context(r, 1), ids, MLP_OUT, W0, g_type).items()})
conv = CTX[name("B")][0][1]
print("write at a target step:", {k: round(v, 2) if isinstance(v, float) else v for k, v in habit.stray_write(conv, PREFIX, MLP_OUT, W0, g_type).items()})
P_small = habit.protection([(context(r, 1), ids, None), (conv, PREFIX, len(PREFIX) - 1)], NEUTRAL[:4], MLP_OUT)
print("protection:", tuple(P_small.shape), P_small.dtype, "symmetric", bool(torch.allclose(P_small, P_small.T)))
try: habit.protection([], NEUTRAL[:1], MLP_OUT)
except ValueError as error: print("empty input:", error)
tools.reset_edit(); mb.close_all()

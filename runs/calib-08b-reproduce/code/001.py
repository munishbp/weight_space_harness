import hashlib, json, numpy as np
EDIT = "data/lm_understand-logs/exp8_math/runs/p50c/edit"
man = json.load(open(f"{EDIT}/manifest.json")); z = np.load(f"{EDIT}/edit.npz")
def bits_of(name):
    return model.get_parameter(name).detach().cpu().contiguous().view(torch.int16).numpy().view(np.uint16)
def sha(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
published = {}
for i, t in enumerate(man["tensors"]):
    base = bits_of(t["name"]).copy()
    assert sha(base) == t["base_sha256"], "base tensor differs from the blog's base"
    e = base.reshape(-1).copy(); e[z[f"t{i}_idx"]] = z[f"t{i}_val"]
    published[t["name"]] = torch.from_numpy(e.reshape(base.shape).view(np.int16)).view(torch.bfloat16)
    entry = tools.save_edit(t["name"], published[t["name"]].float())
    assert sha(bits_of(t["name"])) == t["edited_sha256"], "edited tensor differs from the blog's checkpoint"
    print("bit-exact:", t["name"])
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in entry.items() if k != "name"})
    print("  blog manifest: changed_entries", t["changed_entries"], "columns", t["changed_columns"], "rows", t["changed_rows"], "relative_change", round(t["relative_change"], 5))
m = json.load(open("runs/calib-08b-reproduce/edit/manifest.json"))
print({k: v for k, v in m.items() if k != "tensors"})
# Exact undo check, then apply again.
tools.reset_edit()
assert all(sha(bits_of(t["name"])) == t["base_sha256"] for t in man["tensors"]); print("undo restores the original bits")
for name, tensor in published.items(): tools.save_edit(name, tensor.float())
assert all(sha(bits_of(t["name"])) == t["edited_sha256"] for t in man["tensors"]); print("edit applied again")

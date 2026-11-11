import shutil
habit.fit(MLP_OUT, [(keys, scale, g)], keep, protect, margin=12.0)
shutil.copy(f"{RUN}/edit/edit.safetensors", f"{RUN}/edit_final.safetensors")
fitted = tools.get_weight(MLP_OUT + ".weight")
# Check that the saved file alone reproduces the edit.
tools.reset_edit()
tools.load_edit(f"{RUN}/edit_final.safetensors")
print("reload max diff", float((tools.get_weight(MLP_OUT + ".weight") - fitted).abs().max()))
print("deviation", tools.deviation(NEUTRAL[:32]))
print(open(f"{RUN}/edit/manifest.json").read()[:1500])

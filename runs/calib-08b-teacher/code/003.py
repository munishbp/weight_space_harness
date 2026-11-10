import json
json.dump({"practice_ids": [p["id"] for p in practice], "base_gen": base_gen}, open("runs/calib-08b-teacher/base_gen_64.json", "w"))
print("saved", len(base_gen), "| base_model is separate:", model.base_model is not model, type(model.base_model).__name__, "| head:", type(model.get_output_embeddings()).__name__)
print([n for n, _ in model.named_modules() if n.count(".") <= 2 and "layers." not in n and "visual" not in n][:20])
print([n for n, _ in model.named_modules() if n.startswith("model.language_model.layers.3.") or n.startswith("model.language_model.layers.0.")][:30])

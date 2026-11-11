from wsp import battery
NEUTRAL = battery.neutral_text(64, 256)
MLP_OUT = "model.language_model.layers.28.mlp.down_proj"
print("turn 3 contexts:", {k.split("-v3")[0]: len(v) for k, v in CTX3.items()})

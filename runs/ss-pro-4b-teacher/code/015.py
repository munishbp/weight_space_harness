exec(open("runs/ss-pro-4b-teacher/setup.py").read())
EDIT_PATH, LABEL, OUT_PATH = "runs/ss-pro-4b-lora15/lora_edit.safetensors", "LoRA 15 min", "runs/ss-pro-4b-lora15/evaluation.json"
exec(open("runs/ss-pro-4b-teacher/eval_edit.py").read())

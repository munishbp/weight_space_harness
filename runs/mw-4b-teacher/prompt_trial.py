import shutil
shutil.copy(f"{RUN}/edit/edit.safetensors", f"{RUN}/edit_button_L28_m12.safetensors"); shutil.copy(f"{RUN}/edit/manifest.json", f"{RUN}/manifest_button_L28_m12.json")
tools.reset_edit()
# The prompt first: does one more sentence in the system prompt fix the same fault with no edit?
HINT = " Close the gripper only when the task needs a grasp. To press a button, keep the gripper open and move down through the button."
original_prompt = mw.SYSTEM_PROMPT
mw.SYSTEM_PROMPT = original_prompt + HINT
try:
    tp = episodes("prompt hint, no edit", PRACTICE, f"{RUN}/trial_prompt_hint.json", reference=base)
finally:
    mw.SYSTEM_PROMPT = original_prompt

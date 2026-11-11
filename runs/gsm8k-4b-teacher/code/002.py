# The baseline with thinking on, and the cheap prompt option first: thinking off.
tools.log("teacher_start", note="closing-token recipe on the 4B model, GSM8K with thinking")
base_gen = trial("base, thinking on")
off_gen = trial("base, thinking off", thinking=False)
json.dump({"base": base_gen, "off": off_gen}, open(f"{RUN}/base_gen_64.json", "w"))

# The probability of each action name at each recorded step, from the saved screenshots (no virtual machine).
# A repeat step is a step where the student writes the same action as in the step before.
from PIL import Image

NAMES = ["click", "double_click", "right_click", "drag", "scroll", "type", "key", "wait", "done", "fail"]
PREFIX = tok.encode('{"action": "', add_special_tokens=False)
FIRST = {n: tok.encode(n, add_special_tokens=False)[0] for n in NAMES}
print("prefix", PREFIX, "first tokens", {n: (i, tok.decode([i])) for n, i in FIRST.items()})
assert len(set(FIRST.values())) == len(NAMES), "Two action names share a first token."
LAST = parts["layers"][-1]



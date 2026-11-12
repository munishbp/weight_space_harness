import torch
from wsp import attribution, battery, edits, habit, layout
parts = layout.find(model)
print(len(parts["layers"]))

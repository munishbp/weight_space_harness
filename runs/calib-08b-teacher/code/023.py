import json
from wsp import battery
text = battery.neutral_text(64, 256)
print(len(text), "passages,", sum(map(len, text)), "tokens")
out = {}
for label, path in (("teacher edit A (veto 1.75)", "runs/calib-08b-teacher/edit_veto_1.75.safetensors"),
                    ("teacher edit B (wire +14)", "runs/calib-08b-teacher/edit_wire_14.safetensors"),
                    ("published edit", "runs/calib-08b-reproduce/edit/edit.safetensors")):
    tools.load_edit(path)
    d = tools.deviation(text); out[label] = d
    print(label, {k: round(v, 4) for k, v in d.items()})
    tools.reset_edit()
json.dump(out, open("runs/calib-08b-teacher/deviation_wikitext.json", "w"), indent=1)

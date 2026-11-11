# ChartQA and RefCOCO again on GLM, with scorers that read the answer format of GLM.
from wsp.bench import chartqa, refcoco
importlib.reload(chartqa); importlib.reload(refcoco); importlib.reload(battery)
tools.reset_edit(); before2 = battery.run(["chartqa", "refcoco"])
tools.load_edit(f"{RUN}/edit_pick_L34_m12.safetensors"); after2 = battery.run(["chartqa", "refcoco"]); tools.reset_edit()
cmp2 = battery.compare(before2, after2)
for k, v in cmp2.items(): print("battery", k, v, flush=True)
v = json.load(open(f"{RUN}/verdict.json")); v["battery"].update(cmp2); json.dump(v, open(f"{RUN}/verdict.json", "w"))
b = json.load(open(f"{RUN}/battery_base.json")); b.update(before2); json.dump(b, open(f"{RUN}/battery_base.json", "w")); stamp("battery sets done again")

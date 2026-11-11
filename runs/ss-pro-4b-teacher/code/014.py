from wsp import battery
import time, json
base_battery = {}
for name in ("mmlu_pro", "countbench", "chartqa", "refcoco", "gsm8k_thinking"):
    t = time.time()
    base_battery.update(battery.run([name]))
    print(name, base_battery[name]["summary"], f"{time.time()-t:.0f} s", flush=True)
    json.dump(base_battery, open("runs/ss-pro-4b-teacher/battery_base.json", "w"))
print("battery done")

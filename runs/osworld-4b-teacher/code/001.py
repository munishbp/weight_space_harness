# State and helpers of the OSWorld run on the 4B model. Run this file in the kernel after each restart.
import importlib, json, os, time
from collections import Counter, defaultdict
from wsp.bench import osworld_screen as ow
from wsp.stats import paired_flips, wilson_interval

RUN = "runs/osworld-4b-teacher"; tok = processor.tokenizer
CLIENT = ow.Client()
T0 = time.time(); stamp = lambda text: print(f"[{(time.time() - T0) / 60:5.1f} min] {text}", flush=True)

def gen(conversations):
    return tools.generate(conversations, max_new_tokens=96, thinking=False)

def episodes(label, task_list, path, generate=gen):
    """Run the tasks one at a time and save after each task, so that a long pass keeps its results when it stops."""
    done = json.load(open(path)) if os.path.exists(path) else []
    seen = {r["id"] for r in done}
    for task in task_list:
        if task["id"] in seen: continue
        r = ow.run_episodes([task], generate, client=CLIENT)[0]; done.append(r); json.dump(done, open(path, "w"))
        names = " ".join(a.get("action", "?") for a in r["actions"])
        print(f"{label} | {r['domain']:20s} {r['id'][:8]} score {r.get('score')} {r['seconds']:4d} s | {names} | {r.get('error', '')[:80]}", flush=True)
    ok = [r for r in done if "error" not in r]
    print(label, "| solved", sum(r["correct"] for r in ok), "of", len(ok), "| bridge errors", len(done) - len(ok), flush=True)
    return done

print("setup done")

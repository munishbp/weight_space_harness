# State and helpers of the OSWorld run on the quantized 27B model. Run this file in the kernel after each restart.
import importlib, json, os, threading, time
from wsp.bench import osworld_screen as _ow; importlib.reload(_ow)
from collections import Counter, defaultdict
from wsp.bench import osworld_screen as ow
from wsp.stats import paired_flips, wilson_interval

from wsp import attribution, battery, edits, habit, layout

RUN = "runs/osworld-27b-teacher"; tok = processor.tokenizer
parts = layout.find(model)
# The screenshots of each step go to the ignored data directory. They rebuild a prompt with no virtual machine.
SHOTS = "data/osworld_runs_27b"
CLIENT = ow.Client()
T0 = time.time(); stamp = lambda text: print(f"[{(time.time() - T0) / 60:5.1f} min] {text}", flush=True)

def gen(conversations):
    return tools.generate(conversations, max_new_tokens=96, thinking=False)

def episodes(label, task_list, path, generate=gen, sockets=(ow.SOCKET_PATH,)):
    """Run the tasks and save after each task, so that a long pass keeps its results when it stops.

    One virtual machine runs one task at a time. With more than one socket, one thread per machine takes
    tasks from a shared list. The model answers one request at a time, so a lock guards `generate`.
    A machine waits about 80 seconds per task with no model call, so two or three machines share one model well.
    """
    done = json.load(open(path)) if os.path.exists(path) else []
    seen = {r["id"] for r in done}
    todo = [t for t in task_list if t["id"] not in seen]
    model_lock, save_lock = threading.Lock(), threading.Lock()
    def locked(conversations):
        with model_lock: return generate(conversations)
    def worker(socket_path):
        # A thread that fails to connect raises BridgeError here. The other threads take its tasks.
        client = CLIENT if socket_path == ow.SOCKET_PATH else ow.Client(socket_path)
        while True:
            with save_lock:
                if not todo: return
                task = todo.pop(0)
            r = ow.run_episodes([task], locked, client=client, shots_dir=SHOTS + "/" + label.replace(" ", "_"))[0]
            names = " ".join(a.get("action", "?") for a in r["actions"])
            with save_lock:
                done.append(r); json.dump(done, open(path, "w"))
                print(f"{label} | {r['domain']:20s} {r['id'][:8]} score {r.get('score')} {r['seconds']:4d} s | {names} | {r.get('error', '')[:80]}", flush=True)
    threads = [threading.Thread(target=worker, args=(s,)) for s in sockets]
    for t in threads: t.start()
    for t in threads: t.join()
    missing = len({t["id"] for t in task_list} - {r["id"] for r in done})
    if missing: raise ow.BridgeError(f"{missing} tasks did not run: a worker stopped. Read the kernel output, then call `episodes` again; it continues from {path}.")
    order = {t["id"]: i for i, t in enumerate(task_list)}; done.sort(key=lambda r: order.get(r["id"], len(order)))
    ok = [r for r in done if "error" not in r]
    print(label, "| solved", sum(r["correct"] for r in ok), "of", len(ok), "| bridge errors", len(done) - len(ok), flush=True)
    return done

print("setup done")

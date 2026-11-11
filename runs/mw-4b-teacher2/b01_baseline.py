import json, time, torch
from wsp.bench import metaworld_text as mw
from wsp import layout, attribution, edits, habit, battery, stats, evaluate

RUN = "runs/mw-4b-teacher2"
TASKS = list(mw.INSTRUCTIONS)
PRACTICE = [(task, seed) for task in TASKS for seed in range(20)]


def gen(conversations):
    # Greedy decoding with a fixed seed, thinking off (the reference setting).
    torch.manual_seed(0)
    return tools.generate(conversations, max_new_tokens=64, thinking=False)


def run_set(specs, tag):
    started = time.time()
    results = mw.run_episodes(specs, gen, batch_size=20)
    json.dump(results, open(f"{RUN}/{tag}.json", "w"))
    by = {}
    for r in results:
        by.setdefault(r["task"], []).append(r["correct"])
    print(tag, round(time.time() - started), "s", {t: sum(v) for t, v in by.items()}, flush=True)
    return results


print(layout.find(model))
base = run_set(PRACTICE, "base_practice")

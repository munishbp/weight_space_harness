import random, time
from wsp import evaluate
from wsp.bench import boxed_math
tools.log("teacher_start", note="find the repetition fault without the published edit")
rows = boxed_math.load_jsonl("data/lm_understand-logs/exp8_math/practice/gsm8k_train.jsonl")
random.Random(0).shuffle(rows)
# Practice items for the search. Items 100..227 stay untouched as the held-out check of ticket 04.
practice = rows[1000:1064]
t = time.time()
base_gen = evaluate.run(practice, boxed_math.conversation, boxed_math.score, batch_size=64, max_new_tokens=2048, **boxed_math.BLOG_SAMPLING)
print(f"{time.time()-t:.0f} s", evaluate.summarize(base_gen))

def loop_onset(ids, window=64, min_period=1, max_period=200):
    """The first position from which the reply repeats with a fixed period until its end, or None."""
    n = len(ids)
    best = None
    for period in range(min_period, max_period + 1):
        if n < 2 * period + window: break
        # Walk back from the end while the token equals the token one period earlier.
        i = n - 1
        while i - period >= 0 and ids[i] == ids[i - period]: i -= 1
        run = n - 1 - i
        if run >= max(window, 2 * period) and (best is None or i + 1 < best[0]): best = (i + 1, period)
    return best

onsets = [loop_onset(r["ids"]) for r in base_gen]
looping = [(r, o) for r, o in zip(base_gen, onsets) if o]
print("hit limit:", sum(r["hit_limit"] for r in base_gen), "| exact periodic loop at the end:", len(looping), "of", len(base_gen))
print("onset positions:", sorted(o[0] for _, o in looping)[:40])
print("periods:", sorted(o[1] for _, o in looping)[:40])
tok = processor.tokenizer
for r, o in looping[:3]:
    print("=== onset", o, "| before the loop:"); print(tok.decode(r["ids"][max(0, o[0]-120):o[0]]))
    print("--- one period:"); print(repr(tok.decode(r["ids"][o[0]:o[0]+o[1]])))
nl = [r for r, o in zip(base_gen, onsets) if not o and r["hit_limit"]]
print("=== hit the limit with no exact loop:", len(nl))
for r in nl[:2]: print("--- tail:"); print(tok.decode(r["ids"][-200:]))

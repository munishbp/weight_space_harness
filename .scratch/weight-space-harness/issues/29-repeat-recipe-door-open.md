# Repeat the habit recipe on a second Meta-World fault and stack the edits

Type: task
Status: resolved

## Question

The first habit edit took 27 minutes with the fault unknown
([Edit the 4B model for one Meta-World fault](25-teacher-edit-metaworld.md)). What does a repeat of
the same recipe cost on a second fault, and do the two edits work in one model?

The second fault: in door-open the model grasps the handle and then opens the gripper. At that step
`open` has a probability of 0.75 and `move` has 0.22. A replay without the `open` succeeds in 5 of 10
episodes.

## Answer

Run directory: `runs/mw-4b-teacher` (scripts `door.py` to `door6.py`, `verdict2.py`; numbers in
`verdict_stack.json`). The stacked edit is `edit_button_door_L28.safetensors`.

### Time

14.8 minutes from the start of the repeat to the end of the practice trial in the simulator. This
includes one failed attempt and its diagnosis. The repeat used the practice contexts of the first
run. The held-out verdict with the battery took 20 minutes more.

| Minute | Step |
|---|---|
| 0.0 | Read the probabilities at the fault step. |
| 1.4 | First fit. `move` goes from 0.22 to 0.95 on unseen seeds. |
| 5.6 | Practice trial of the stack: door-open 13 of 20, but peg-insert 19 to 8 and pick-place 20 to 15. |
| 7.2 | A zero-value constraint for the other tasks does not help. |
| 8.0 | Diagnosis: the change writes its direction at image and template positions. |
| 10.4 | New protection from all prompt positions. The stray writes fall from about 1,400 to 30 per prompt. |
| 14.8 | Practice trial of the stack: 79.5%, with 34 episodes fixed and 1 broken. |

### The defect in the first recipe

The protection set held the action step of the other contexts and neutral text. It held no image
position and no template position. The fitted change wrote the direction at hundreds of those
positions in every prompt, and layers 29 to 31 read them. The button edit had the same defect and
was lucky. The fix: build the protection from every position of the task prompts, and leave out
only the target step. The matrix change of the door step fell from 0.40 to 0.17 with the fix.

### Held-out result of the stacked edit

| Held-out episodes (20 per task) | Base | Button edit | Button + door edit |
|---|---|---|---|
| button-press-topdown | 0 | 18 | 18 |
| door-open | 0 | 0 | 7 |
| push | 14 | 13 | 15 |
| pick-place | 20 | 20 | 20 |
| peg-insert-side | 17 | 17 | 17 |
| window-open | 15 | 15 | 15 |
| drawer-open | 0 | 0 | 0 |
| reach, drawer-close, window-close | 60 | 60 | 60 |
| All ten tasks | 63.0% | 71.5% | 76.0% |

- Against the base model: 26 episodes fixed, 0 broken (p = 3e-8). Practice seeds: 79.5%.
- Random control (the door fit with a random direction): 0 of 20 on door-open.
- Edit size: 1 tensor, rank 2, 0.50% of the weights. Relative change of the model 0.0054.
- WikiText-2: KL 0.0003, top-token agreement 99.3%, perplexity 13.384 to 13.387.
- Battery: MMLU-Pro 48.0 to 47.0, CountBench 85.0 to 85.0, ChartQA 85.0 to 84.5, RefCOCO 90.0 to
  89.5, GSM8K with thinking 75.0 to 72.0 (4 fixed, 7 broken, p = 0.55). No major drop.

### What this shows

- A repeat of a known recipe fits the 15-minute target, with a failed attempt inside it.
- Two edits for two faults stack in one matrix. The second does not undo the first.
- The door-open gain on held-out seeds (7 of 20) is half of the gain on practice seeds (14 of 20).
  The replay gave 5 of 10, so the edit reaches about what the changed step can give.
- The door fit also lowers `open` after `close` in the other tasks at that step. No task lost
  episodes on held-out seeds.

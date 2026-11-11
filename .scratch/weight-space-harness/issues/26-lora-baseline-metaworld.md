# Run the LoRA baseline on the Meta-World target tasks

Type: task
Status: resolved
Blocked by: 24

## Question

What held-out success and what retention does a LoRA fine-tune of 15 minutes reach on the Meta-World
target tasks (button-press-topdown, push, window-open)?

The training examples are the steps of the reference plans that succeed on the practice seeds 0 to
19 of the three target tasks. The ticket reports the same measures as the teacher edit: held-out
success on the target tasks, success on the other seven tasks, the battery, the edit size, and the
deviation.

## Answer, part 1: LoRA on the three target tasks only

Measured on 2026-11-10. Run directory: `runs/mw-4b-lora15`.

**The LoRA model solves the three target tasks (57 of 60 held-out episodes, base 29) and loses two
tasks that it did not train on (pick-place 20 to 2, peg-insert-side 17 to 0). Over all ten tasks it
scores 60.0%, below the base model (63.0%) and the teacher edit (71.5%).**

### Training data

- One example is one step of a hand-written reference plan (`reference_plan` in
  `wsp/bench/metaworld_text.py`): the prompt that the model sees in the benchmark, and the next
  primitive of the plan as the reply.
- Tasks: button-press-topdown, push, window-open. Seeds 0 to 19. Only plans that succeed in the
  simulator. 122 examples.
- 15 minutes of training: 3,401 examples seen, 27.9 passes over the data. The loss reaches zero
  after about 4 passes. Settings as in the ScreenSpot Pro run (rank 16, learning rate 1e-4).
- The teacher ran read passes on the same GPU during the training. The loss was at zero for the last
  13 minutes, so the shared GPU did not change the result.

### Measures

| Held-out episodes (20 per task) | Base | Teacher edit | LoRA, target tasks |
|---|---|---|---|
| button-press-topdown (target) | 0 | 18 | 20 |
| push (target) | 14 | 13 | 17 |
| window-open (target) | 15 | 15 | 20 |
| pick-place | 20 | 20 | **2** |
| peg-insert-side | 17 | 17 | **0** |
| reach, drawer-close, window-close | 60 | 60 | 60 |
| drawer-open, door-open | 0 | 0 | 1 |
| All ten tasks | 63.0% | 71.5% | 60.0% (31 fixed, 37 broken) |

| Measure | Teacher edit | LoRA, target tasks |
|---|---|---|
| Weights changed | 0.48%, 1 tensor, 1 layer | 60.4%, 176 tensors, 32 layers |
| Relative change of the model | 0.0049 | 0.0053 |
| WikiText-2: KL, top-token agreement | 0.0003, 99.3% | 0.0155, 93.9% |
| MMLU-Pro (base 48.0) | 47.0 | 46.0 |
| CountBench (base 85.0) | 85.0 | 86.0 |
| ChartQA (base 85.0) | 85.0 | 84.5 |
| RefCOCO (base 90.0) | 89.5 | 88.5 |
| GSM8K with thinking (base 75.0) | 75.0 | 85.0 (13 fixed, 3 broken, p = 0.02) |

### Why the two tasks break

No reference plan of the three target tasks holds a `close` step. The LoRA model learned to write
moves only, and the tasks that need a grasp fail. The training set is narrow by construction: it holds
the target portion only, as in the ScreenSpot Pro run.

### A side effect

GSM8K with thinking rises by 10 points. The share of answers at the 4,096 token limit falls from 33%
to 19%. Training on short replies made the model end its thinking sooner. This is the fault of
[Test the closing-token recipe on the 4B model](23-thinking-fault-4b.md), changed by accident.

## Answer, part 2: LoRA on all nine tasks with a reference plan

The narrow training set is a harsh setup for LoRA. A second run trains for 15 minutes on the
reference plans of all nine tasks that have one (502 examples, seeds 0 to 19, 7.4 epochs). Door-open
has no reference plan, so it has no examples. Files: `runs/mw-4b-lora15-all/`.

| Held-out episodes (20 per task) | Base | Teacher edit | LoRA, target tasks | LoRA, nine tasks |
|---|---|---|---|---|
| button-press-topdown | 0 | 18 | 20 | 20 |
| push | 14 | 13 | 17 | 17 |
| window-open | 15 | 15 | 20 | 19 |
| pick-place | 20 | 20 | 2 | 20 |
| peg-insert-side | 17 | 17 | 0 | 19 |
| drawer-open | 0 | 0 | 1 | 20 |
| door-open (no examples) | 0 | 0 | 0 | 0 |
| reach, drawer-close, window-close | 60 | 60 | 60 | 60 |
| All ten tasks | 63.0% | 71.5% | 60.0% | 87.5% |

- Held-out against the base model: 52 episodes fixed, 3 broken (p = 2e-12). Practice seeds: 86.0%.
- Edit size: 63.9% of the weights, 176 tensors, all 32 layers. Relative change of the model 0.0080.
- WikiText-2: KL 0.0205, top-token agreement 92.6%, perplexity 13.38 to 13.42.
- Battery: MMLU-Pro 48.0 to 48.5, CountBench 85.0 to 85.0, ChartQA 85.0 to 82.5 (2 fixed, 7 broken,
  p = 0.18), RefCOCO 90.0 to 88.5, GSM8K with thinking 75.0 to 78.0. No major drop.

### What part 2 shows

- With examples of every task, LoRA beats the teacher edit by 16 points in the same 15 minutes. It
  also fixes drawer-open (0 to 20), which no closed-form edit fixed.
- The cost of part 1 (pick-place and peg-insert lost) came from the narrow training set, not from
  LoRA as a method. Examples of the neighbouring tasks remove it.
- LoRA needs a plan that works for each task. A person wrote those plans. The teacher edit needed
  no plan: it used the action that the model already rated at 12%.
- Door-open stays at 0. A task with no examples gets no gain from the other nine.
- The LoRA model is farther from the original than the teacher edit on every deviation measure
  (top-token agreement 92.6% against 99.3%). The battery does not show a significant drop.

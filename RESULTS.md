# Results

Updated: 2026-11-10. Teacher: Claude (Fable 5.2). Machine: one RTX 5090 (32 GB).

Each block has a ticket with the full numbers in `.scratch/weight-space-harness/issues/`. No block
has a pass or a fail. Each number is a measurement.

## Summary so far

| Block | Student | Teacher edit | LoRA, 15 minutes |
|---|---|---|---|
| Calibration: thinking that never ends, GSM8K | Qwen3.5-0.8B | Held-out 10.2% to 38.3% in 29 minutes. 0.98% of the weights. The published edit of the blog reaches 21.9%. | not run |
| Computer use: ScreenSpot Pro icon targets | Qwen3.5-4B | No edit found in 48 minutes. | Held-out icons 38.2% to 50.2%. Text targets 76.0% to 63.5%. 55% of the weights. |
| Robotics: Meta-World through text waypoints | Qwen3.5-4B | Button-press 0 to 18 of 20 held-out episodes in 27 minutes. One matrix, 0.48% of the weights. Nothing else changes. A second fault did not yield. | Target tasks 29 to 57 of 60. Two other tasks fall from 37 to 2 of 40. A second run on all tasks is pending. |
| Thinking fault on the 4B model, GSM8K | Qwen3.5-4B | planned | |
| Scaling: quantized 27B | Qwen3.8-27B NVFP4 | loader written, run planned | |

## 1. Calibration on Qwen3.5-0.8B

- The harness applies the edit that the blog published bit for bit, and it removes the edit exactly.
  The harness measures the same edit size as the blog (130,188 weights, 0.0153%, layers 15 and 19).
- 128 held-out GSM8K items, thinking on, a limit of 4,096 tokens. Base: 10.2% correct, 93.8% at the
  limit. Published edit: 21.9% correct, 76.6% at the limit.
- The teacher found the fault again with the harness tools in 29 minutes (44 minutes with the
  held-out evaluation; 31 generation passes, 14 read passes, 22 code runs). The fault: the MLPs of
  layers 21 and 22 and the attention of layer 18 suppress the token that closes the thinking.
- Edit A removes that suppression (0.98% of the weights): 38.3% correct, 56.2% at the limit. It beats
  the published edit (p = 0.0003). On WikiText-2 the edited model keeps the top token in 98.1% of
  the positions (KL 0.0011). Edit B wires 150 neurons to the closing token (0.018% of the weights):
  37.5% correct.
- **The base model with thinking off scores 56.2% with no edit.** A prompt switch beats every edit
  here. The edits help the model to stop. They do not stop its second-guessing.

## 2. Computer use: ScreenSpot Pro on Qwen3.5-4B

- Baseline: 59.4% on the full set (published: 60.3). Text targets 74.1%, icon targets 35.6%. One
  full pass takes 36 minutes.
- Target portion: icon targets. 311 practice items and 293 held-out items.
- **Teacher edit: negative after 48 minutes.** In 60 of 99 failures the correct digit has a
  probability of 10% or more at the step where the reply leaves the target. No part of the model
  suppresses the correct answer. The late layers write the choice that the model already made, right
  or wrong. A closed-form fit on the MLP of layer 29 fixes 2 items and breaks 2.
- **LoRA, 15 minutes, same practice items:** held-out icons 50.2% (47 fixed, 12 broken). Text targets
  of the same benchmark drop from 76.0% to 63.5% (6 fixed, 31 broken). The battery shows no major
  drop. The LoRA model differs from the original in 55.3% of its weights, in all 32 layers.

| Battery set | Base | LoRA |
|---|---|---|
| MMLU-Pro | 48.0 | 47.0 |
| CountBench | 85.0 | 82.5 |
| ChartQA | 85.0 | 84.5 |
| RefCOCO | 90.0 | 89.0 |
| GSM8K with thinking | 75.0 | 78.0 |

## 3. Robotics: Meta-World on Qwen3.5-4B

The interface is ours: the model sees a camera image and the positions of the gripper, the object,
and the goal, and it writes one primitive per turn (`move` to a position, `close`, `open`, `done`).
The ten MT10 tasks, 20 practice seeds and 20 held-out seeds per task.

- Base model: 63.0% of 200 episodes, on practice seeds and on held-out seeds.
- Fault 1: the model closes the gripper before a button press. At that step the correct action has
  a probability of 12%.
- **Teacher edit: a least-squares fit on the MLP output matrix of layer 28, rank 1.** It raises
  `move` over `close` at that step of the button task only.

| Held-out episodes (20 per task) | Base | Teacher edit | LoRA, 15 minutes, target tasks only |
|---|---|---|---|
| button-press-topdown (target) | 0 | **18** | 20 |
| push (target) | 14 | 13 | 17 |
| window-open (target) | 15 | 15 | 20 |
| pick-place | 20 | 20 | **2** |
| peg-insert-side | 17 | 17 | **0** |
| the other five tasks | 60 | 60 | 61 |
| All ten tasks | 63.0% | **71.5%** | 60.0% |

| Measure | Teacher edit | LoRA, target tasks only |
|---|---|---|
| Time | 27 minutes to the edit (first time, fault unknown) | 15 minutes of training, plus hand-written reference plans |
| Weights changed | 0.48% (1 tensor, 1 layer) | 60.4% (176 tensors, 32 layers) |
| WikiText-2: top-token agreement with the original | 99.3% | 93.9% |
| Battery, largest drop | 1.0 point (MMLU-Pro) | 2.0 points (MMLU-Pro) |
| Random control (same fit, random direction) | 2 of 20 on button-press | |

- A sentence in the system prompt also fixes the button task (18 of 20 practice episodes), and it
  costs 8 episodes in push and window-open.
- Fault 2 (in push, the first move goes to the goal in place of the object): two closed-form edits
  gave no significant gain on 40 unseen episodes. The LoRA model fixes it.
- The LoRA model saw only the three target tasks. None of their plans holds a `close`, and the tasks
  that need a grasp broke.
- Limits: the tasks are short (2 to 6 primitives), and the prompt holds positions from the simulator
  state. This block does not show a long-horizon gain.

## What the blocks show

- A closed-form edit works when a fault hides an ability that the model has. The 0.8B model knew
  when to close its thinking, and three parts blocked it.
- A closed-form edit also works on the 4B model for a habit that is wrong in one context, when the
  right action already has a visible probability. The edit changed one robot task and nothing else.
- A closed-form edit did not fix a wrong choice between candidates from the input (two screen
  elements, two positions). Training fixes that kind in 15 minutes.
- Training has a price next to the target, in both blocks: 12.5 points on text targets, and two
  robot tasks lost.
- The generalization battery alone does not show the cost of the LoRA model. A near-transfer set
  next to the target does.

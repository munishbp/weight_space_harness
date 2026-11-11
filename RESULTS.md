# Results

Updated: 2026-11-11. Teacher: Claude (Fable 5.2). Machine: one RTX 5090 (32 GB).

Each block has a ticket with the full numbers in `.scratch/weight-space-harness/issues/`. No block
has a pass or a fail. Each number is a measurement.

## Summary so far

| Block | Student | Teacher edit | LoRA, 15 minutes |
|---|---|---|---|
| Calibration: thinking that never ends, GSM8K | Qwen3.5-0.8B | Held-out 10.2% to 38.3% in 29 minutes. 0.98% of the weights. The published edit of the blog reaches 21.9%. | not run |
| Computer use: ScreenSpot Pro icon targets | Qwen3.5-4B | No edit found in 48 minutes. | Held-out icons 38.2% to 50.2%. Text targets 76.0% to 63.5%. 55% of the weights. |
| Robotics: Meta-World through text waypoints | Qwen3.5-4B | First fault: button-press 0 to 18 of 20 held-out episodes in 27 minutes. Second fault, stacked: 15 minutes. Both: 63.0% to 76.0%, 26 episodes fixed, 0 broken, 0.50% of the weights. | Three target tasks only: 60.0% of all episodes, two grasp tasks lost. Nine tasks: 63.0% to 87.5%, no task lost. 64% of the weights. |
| Computer use, multi-step: MiniWoB++ email tasks through screenshots | Qwen3.5-4B | Held-out 29.0% to 57.0% (29 episodes fixed, 1 broken). First working edit at 19 minutes. One matrix, 0.51% of the weights. The other 78 task types: 63.8% to 65.4%. | Held-out 29.0% to 87.0%. The other 78 task types: 63.8% to 61.5%, with 61 of 312 episodes changed. 65% of the weights. |
| Scaling: MiniWoB++ email tasks on a quantized 27B model | Qwen3.8-27B NVFP4 | Held-out 30.0% to 68.0% (39 episodes fixed, 1 broken). The recipe and the contexts of the 4B run, layer 60. One matrix, 0.22% of the weights. The other 78 task types: 73.1% to 73.1%. 56 minutes at 3 tokens per second. | not run: no memory for training |
| Thinking fault on the 4B model, GSM8K | Qwen3.5-4B | not run | |

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

| Held-out episodes (20 per task) | Base | Teacher edit | LoRA, target tasks only | LoRA, nine tasks |
|---|---|---|---|---|
| button-press-topdown (target) | 0 | **18** | 20 | 20 |
| push (target) | 14 | 13 | 17 | 17 |
| window-open (target) | 15 | 15 | 20 | 19 |
| pick-place | 20 | 20 | **2** | 20 |
| peg-insert-side | 17 | 17 | **0** | 19 |
| drawer-open | 0 | 0 | 1 | **20** |
| door-open (no training examples) | 0 | 0 | 0 | 0 |
| reach, drawer-close, window-close | 60 | 60 | 60 | 60 |
| All ten tasks | 63.0% | 71.5% | 60.0% | **87.5%** |

| Measure | Teacher edit | LoRA, target tasks only | LoRA, nine tasks |
|---|---|---|---|
| Time | 27 minutes to the edit (first time, fault unknown) | 15 minutes of training | 15 minutes of training |
| Input from a person | none | reference plans for 3 tasks | reference plans for 9 tasks |
| Weights changed | 0.48% (1 tensor, 1 layer) | 60.4% (176 tensors, 32 layers) | 63.9% (176 tensors, 32 layers) |
| WikiText-2: top-token agreement with the original | 99.3% | 93.9% | 92.6% |
| Battery, largest drop | 1.0 point (MMLU-Pro) | 2.0 points (MMLU-Pro) | 2.5 points (ChartQA) |
| Random control (same fit, random direction) | 2 of 20 on button-press | | |

- **A second edit, stacked.** The same recipe on the door-open fault (the model opens the gripper
  after the grasp) took 14.8 minutes, with one failed attempt and its diagnosis. Both edits in one
  matrix: 76.0% of the held-out episodes, 26 fixed and 0 broken. Door-open goes from 0 to 7 of 20.
  The edit is rank 2, 0.50% of the weights, and the top-token agreement stays at 99.3%.
- The failed attempt showed a defect in the recipe: the protection set must hold every position of
  the prompts, not only the step that the edit targets.
- A sentence in the system prompt also fixes the button task (18 of 20 practice episodes), and it
  costs 8 episodes in push and window-open.
- Fault 2 (in push, the first move goes to the goal in place of the object): two closed-form edits
  gave no significant gain on 40 unseen episodes. The LoRA model fixes it.
- The first LoRA model saw only the three target tasks. None of their plans holds a `close`, and the
  tasks that need a grasp broke. The second LoRA model saw nine tasks and broke none.
- With a working plan for each task, LoRA beats the teacher edit by 16 points in the same time. The
  teacher edit needed no plan from a person.
- Limits: the tasks are short (2 to 6 primitives), and the prompt holds positions from the simulator
  state. This block does not show a long-horizon gain.

## 4. Computer use, multi-step: MiniWoB++ on Qwen3.5-4B

The interface is ours: the model sees one screenshot per turn and writes one action (`click` at a
point, `type`, `key`, `done`). It gets no DOM. 83 task types, at most 10 actions per episode.

- Base model: 59.0% of 332 episodes (4 seeds per task type). One pass takes 6.4 minutes.
- Target portion: five email task types (reply, forward, mixed inbox), 20 practice seeds and 20
  held-out seeds each. Base: reply and forward 0 of 40.
- The fault: three wrong action names in a row. The model types before it clicks Reply, types
  before the text field has the focus, and presses Enter where the page needs a click on Send.
- **Teacher edit: one least-squares fit on the MLP output matrix of layer 28, rank 2.** It raises
  `click` over `type` and over `key` at those three steps of the email tasks only.

| Held-out episodes | Base | Teacher edit | LoRA, 15 minutes |
|---|---|---|---|
| email-inbox-reply (20) | 0 | 11 | **19** |
| email-inbox-forward (20) | 0 | 3 | **17** |
| email-inbox, three mixed types (60) | 29 | 43 | 51 |
| All five email task types | 29.0% | 57.0% | **87.0%** |
| The other 78 task types (312 episodes) | 63.8% | 65.4% | 61.5% |
| Episodes of other task types that change | | **7** (6 fixed, 1 broken) | 61 (27 fixed, 34 broken) |

| Measure | Teacher edit | LoRA, 15 minutes |
|---|---|---|
| Time | 19 minutes to the first edit that works. 58 minutes with the search for side effects. | 15 minutes of training on 140 steps of successful practice episodes |
| Weights changed | 0.51% (1 tensor, 1 layer, rank 2) | 64.6% (176 tensors, 32 layers) |
| WikiText-2: top-token agreement with the original | 99.2% | 94.5% |
| Battery, largest drop | 1.0 point (GSM8K with thinking) | 3.0 points (RefCOCO, p = 0.07) |
| Random control (same fit, random directions) | 29 of 100, no episode changes | |

- A forced replay set the ceiling before the fit: with the three action names forced, the model
  solves 58 of 100 practice episodes. The edit reaches 56.
- Forward stays low. The edit fixes the choice of the action, and the click on the field goes to a
  wrong place in the forward form.
- LoRA gains 30 points more on the target. It learns where to click in the forward form. Its
  examples came from the forced replay, so it used the same diagnosis as the edit.
- Two base runs with different batches differ in 12 of 312 episodes. A side effect needs that
  number next to it.

## 5. Scaling: the quantized Qwen3.8-27B

The checkpoint is a ModelOpt NVFP4 copy (4-bit weights with a scale per block of 16). `wsp/modelopt.py`
keeps the weights in that format on the GPU and decodes them inside each forward pass. An edit goes
back into the same format.

- Load: 40 seconds, 19.6 GB on a 32 GB GPU. Speed: 2.9 tokens per second.
- Meta-World, base model: 77.5% of 200 practice episodes (4B: 63.0%). The two habit faults of the 4B
  model do not exist here (button-press 17 of 20, door-open 14 of 20). No edit was tried.
- MiniWoB++ email tasks, base model: reply and forward 0 of 40, the same fault as on the 4B model.
  At the three fault steps `click` has a probability of 0.31, 0.15, and 0.38.
- **Teacher edit: the recipe of block 4 on the MLP output matrix of layer 60 of 64.** The page
  states at the fault steps came from the 4B run. One fit takes 32 seconds.

| Held-out episodes | Base | Teacher edit |
|---|---|---|
| email-inbox-reply (20) | 0 | **20** |
| email-inbox-forward (20) | 0 | 1 |
| email-inbox, three mixed types (60) | 30 | 47 |
| All five email task types | 30.0% | **68.0%** (39 fixed, 1 broken, p = 7e-11) |
| The other 78 task types (1 seed each) | 73.1% | 73.1% (1 fixed, 1 broken) |

| Measure | Teacher edit |
|---|---|
| Time | 56 minutes from the first base pass to the fit. 20 minutes of that are one base pass. |
| Weights changed | 0.22% (1 tensor of 1,184, 1 layer) |
| WikiText-2: top-token agreement with the original | 98.9% (KL 0.0007) |
| Battery, 100 items per set | MMLU-Pro 60 to 61. CountBench 94, ChartQA 75, RefCOCO 94: no change. |
| Share of the requested change that the 4-bit format keeps | 1.03 |
| `reset_edit` restores the stored bytes | yes |
| Random control (same fit, random directions) | 30 of 100, no episode changes |

- The same closed-form recipe works at 27B and through 4-bit weights. The gain is larger than on the
  4B model (38 points against 28), because the 27B model clicks the right places after the fix.
- The cost of one teacher step grows with the student. One pass of 100 browser episodes takes 20
  minutes, so the teacher reads probabilities on unseen seeds in place of practice trials.
- This run is a transfer of a known fault, not a discovery. The teacher did not search the 27B model
  for a new fault.
- No LoRA comparison ran. Training does not fit next to a 27B model on this GPU.
- The 27B model is a newer generation and a quantized copy. The two sizes differ in more than size.

## What the blocks show

- A closed-form edit works when a fault hides an ability that the model has. The 0.8B model knew
  when to close its thinking, and three parts blocked it.
- A closed-form edit also works on the 4B model for a habit that is wrong in one context, when the
  right action already has a visible probability. This holds in three cases: two robot tasks and the
  email tasks of a web benchmark. Each edit changed its target and nothing else that we measured.
- The time to a working edit fell with each use of the recipe: 27 minutes, 15 minutes, and 19
  minutes on a new benchmark.
- The recipe scales to a 27B student in a 4-bit format. The same fault, the same contexts, and a new
  layer number gave a larger gain there. The time grows with the cost of a forward pass.
- A closed-form edit did not fix a wrong choice between candidates from the input (two screen
  elements, two positions). Training fixes that kind in 15 minutes.
- Training on the target alone has a price next to the target, in both blocks: 12.5 points on text
  targets, and two robot tasks lost. Training data that covers the neighbours removes that price.
- In the two blocks where both methods work (robot tasks and email tasks), LoRA on correct examples
  gains more on the target, and the closed-form edit stays closer to the original model: 0.5% of the
  weights against 60%, 99% top-token agreement against 93 to 95%, and 7 changed episodes of other
  web tasks against 61.
- Training needs correct examples. The closed-form edit needs only a fault that the teacher can
  locate. Where both apply, training on broad examples gave the larger gain.
- The generalization battery alone does not show the cost of the LoRA model. A near-transfer set
  next to the target does.

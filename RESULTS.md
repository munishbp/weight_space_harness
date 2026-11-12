# Results

Updated: 2026-11-11. Teacher: Claude (Fable 5.2). Machine: one RTX 5090 (32 GB).

Each block has a ticket with the full numbers in `.scratch/weight-space-harness/issues/`. No block
has a pass or a fail. Each number is a measurement.

## Answer to the question

The question: can a frontier model edit the weights of a small open model directly, with no training
loop, so that the model gets good at one portion of a realistic task in 10 to 15 minutes and keeps
its general ability? Does this scale to a 4B model, and is it feasible?

- **It is feasible for two kinds of fault.** In the first kind the student has the ability and
  writes a wrong default at one step: `close` before a button press, `type` before a click on
  Reply. One closed-form fit on one MLP matrix fixes that step. It worked in all five cases of this
  kind that we tried (one of them two times, with two teachers), on three students from 4B to 27B
  and from two families. In the second kind a part of the model blocks a token that the model
  needs (the fault of the blog). One removal on one matrix fixes it on the 0.8B and the 4B model.
- **It scales.** The same recipe and the same code ran on Qwen3.5-4B, GLM-4.6V-Flash (10B), and a
  4-bit Qwen3.8-27B. The gain on the 27B model is the largest.
- **The time is 8 to 30 minutes on a 4B model for one step.** The fault of the blog took 8 minutes.
  A habit fault took 15 to 30 minutes. A stacked edit for three procedures took 63 minutes. In OSWorld the virtual machine sets the time: 150 seconds per task. The 27B
  model needs about an hour, because one forward pass is slow.
- **The model keeps its general ability, and a large edit costs more.** Each edit changes 0.2% to
  0.5% of the weights in one matrix. For a one-step edit the top token on neutral text stays the
  same in 99% of the positions, no battery set drops by more than 1 point with significance, and
  tasks next to the target do not change. The stacked edit of 46 targets costs 2.5 points on MMLU-Pro.
- **The gain is local, and the portion does not become "super great".** An edit fixes the steps that
  it targets. The email tasks go from 29% to 57% on the 4B model, because a later click is also
  wrong and no edit of this kind fixes a wrong position.
- **It does not work when the fault is a wrong choice between things in the input** (which icon,
  which position). ScreenSpot Pro and the Meta-World push task gave no edit. Training fixes that kind.
- **Long-horizon tasks: the edits fix a default, not a missing procedure.** In OSWorld (real desktop
  applications) the student solves 16 of 83 scanned tasks, and its failures on possible tasks share
  no step, so no edit applies there. One portion has a habit fault: the tasks that are not possible
  by design, where the correct reply is `fail`. One edit takes them from 0 to 10 of 11 held-out
  tasks, with the battery within 1 point. Its price is a false `fail` on 7 of 38 possible tasks, and
  a prompt sentence with the base weights reaches 7 of 11. On the 27B student the same edit gives
  2 to 8 of 11, but the margin from the sweep costs 5 of the 7 possible tasks that the base solved.
  A weaker edit keeps the gain and loses 2. A prompt sentence gives the same 8 of 11 there and
  loses 1. In
  the VLABench planning tasks (5 to 9 steps) one stacked edit writes the skill names of three
  procedures (all names correct in 91 of 150 held-out plans, base 0). No plan becomes fully
  correct, and that edit costs 2.5 points on one battery set.
- **A LoRA fine-tune of 15 minutes gains more on the target when correct examples exist** (87%
  against 57% on the email tasks). It changes 60% of the weights and moves tasks next to the target.
  The edit needs no examples, only a fault that the teacher can locate.

| Case | Student | Time to a working edit | Target, held-out | All tasks of the benchmark portion |
|---|---|---|---|---|
| Button-press, Meta-World | Qwen3.5-4B | 27 min | 0 to 18 of 20 | 63.0% to 71.5% |
| Door-open, stacked, Meta-World | Qwen3.5-4B | 15 min | 0 to 7 of 20 | 71.5% to 76.0% |
| Button-press, second teacher | Qwen3.5-4B | 30 min | 0 to 16 of 20 | 63.0% to 71.5% |
| Email tasks, MiniWoB++ | Qwen3.5-4B | 19 min | 29 to 57 of 100 | other task types 63.8% to 65.4% |
| Email tasks, MiniWoB++ | Qwen3.8-27B, 4-bit | 56 min | 30 to 68 of 100 | other task types 73.1% to 73.1% |
| Pick-place, Meta-World | GLM-4.6V-Flash | 30 min | 12 to 20 of 20 | 51.5% to 57.0% |
| Thinking that does not end, GSM8K | Qwen3.5-4B | 8 min | 77.3% to 89.1% | battery within 0.5 points |
| Three procedures, VLABench plans | Qwen3.5-4B | 63 min | skill names all correct: 0 to 91 of 150. Fully correct: 0 to 0 | other tasks: total score 0.429 to 0.428 |
| Tasks that are not possible, OSWorld | Qwen3.5-4B | 47 min after a base scan of 63 min | 0 to 10 of 11 | possible tasks 4 to 3 of 38, with 7 false `fail` |
| Possible desktop tasks, OSWorld | Qwen3.5-4B | no edit (2 hours of checks) | base 16 of 83 | |
| Tasks that are not possible, OSWorld | Qwen3.8-27B, 4-bit | 30 min after base runs of 178 min | 2 to 8 of 11 | possible tasks 7 to 2 of 17, with 10 false `fail`. Margin 8, chosen after the verdict: 7 to 5, with 4 |

What this effort does not show:

- A possible long-horizon task that the student completes after an edit and fails before it. The
  edits fix one step, one refusal, or one list of skill names.
- The OSWorld tasks of the other seven applications. The base scan covers 3 of 10 groups and all
  26 tasks that are not possible.
- A clean held-out verdict for the weaker `fail` edit on the 27B student. Its margin came from a
  read of held-out steps.
- VLABench in its simulator. The run here uses the planning track for vision-language models.
- A teacher from another vendor. The second teacher was another Claude model.
- Qwen4. The harness loads a student by name, so a new model needs no code change if its layers
  follow the names in `wsp/layout.py`.

## Summary

| Block | Student | Teacher edit | LoRA, 15 minutes |
|---|---|---|---|
| Calibration: thinking that never ends, GSM8K | Qwen3.5-0.8B | Held-out 10.2% to 38.3% in 29 minutes. 0.98% of the weights. The published edit of the blog reaches 21.9%. | not run |
| Computer use: ScreenSpot Pro icon targets | Qwen3.5-4B | No edit found in 48 minutes. | Held-out icons 38.2% to 50.2%. Text targets 76.0% to 63.5%. 55% of the weights. |
| Robotics: Meta-World through text waypoints | Qwen3.5-4B | First fault: button-press 0 to 18 of 20 held-out episodes in 27 minutes. Second fault, stacked: 15 minutes. Both: 63.0% to 76.0%, 26 episodes fixed, 0 broken, 0.50% of the weights. | Three target tasks only: 60.0% of all episodes, two grasp tasks lost. Nine tasks: 63.0% to 87.5%, no task lost. 64% of the weights. |
| Computer use, multi-step: MiniWoB++ email tasks through screenshots | Qwen3.5-4B | Held-out 29.0% to 57.0% (29 episodes fixed, 1 broken). First working edit at 19 minutes. One matrix, 0.51% of the weights. The other 78 task types: 63.8% to 65.4%. | Held-out 29.0% to 87.0%. The other 78 task types: 63.8% to 61.5%, with 61 of 312 episodes changed. 65% of the weights. |
| Scaling: MiniWoB++ email tasks on a quantized 27B model | Qwen3.8-27B NVFP4 | Held-out 30.0% to 68.0% (39 episodes fixed, 1 broken). The recipe and the contexts of the 4B run, layer 60. One matrix, 0.22% of the weights. The other 78 task types: 73.1% to 73.1%. 56 minutes at 3 tokens per second. | not run: no memory for training |
| A student outside Qwen: Meta-World | GLM-4.6V-Flash (10.3B) | Held-out 51.5% to 57.0% (11 episodes fixed, 0 broken). Pick-place 12 to 20 of 20. Practice trial 30 minutes after the first pass. 0.50% of the weights. | not run |
| Thinking that does not end, GSM8K | Qwen3.5-4B | Held-out 77.3% to 89.1% (17 items fixed, 2 broken). Replies at the token limit: 30.5% to 8.6%. Working edit at 8 minutes. One matrix, 0.44% of the weights. Thinking off scores 93.0% with no edit. | not run |
| Long-horizon computer use: OSWorld | Qwen3.5-4B | Possible tasks: no edit candidate (base 16 of 83, three forced replays solve 0 failed tasks). Tasks that are not possible: held-out 0 to 10 of 11 with one rank-1 edit, 0.51% of the weights, battery within 1 point, 7 of 38 possible tasks end with a false `fail`. | not run (no examples) |
| Long-horizon computer use: OSWorld on the 27B model | Qwen3.8-27B NVFP4 | Base: 24 of 55 possible tasks (4B: 15). Tasks that are not possible: held-out 2 to 8 of 11, 0.20% of the weights, battery unchanged. Possible held-out tasks 7 to 2 of 17 at margin 12 (10 false `fail`), 7 to 5 at margin 8 (after the verdict). A prompt sentence with the base weights: 8 of 11 and 6 of 17. | not run |
| Long-horizon robot plans: VLABench, three procedures | Qwen3.5-4B | Held-out: all skill names correct in 0 to 91 of 150 plans, official total score 0.151 to 0.454, fully correct plans 0 to 0. 0.52% of the weights. MMLU-Pro 48.0 to 45.5. Works in the plan-only mode only. | not run |

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
- **A second teacher.** Claude Opus 5.5, with a fresh context and only `PLAYBOOK.md` and the code,
  repeated the button edit in 30 minutes. It chose layer 27. Held-out: 63.0% to 71.5%, 17 episodes
  fixed and 0 broken, button-press 0 to 16 of 20. The playbook names this fault, so the run shows
  that a second model can operate the harness. It does not show a second discovery.
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

## 6. A student outside the Qwen family: GLM-4.6V-Flash

`zai-org/GLM-4.6V-Flash` has 10.3 billion weights, 40 layers, and takes images. It runs unquantized
on the 32 GB GPU (19.2 GB).

- Every tool ran on it with no code change: load, layout, generation with an image, activation
  reads, an edit, the deviation, and an exact reset.
- Meta-World, base model: 56.5% of 200 practice episodes.
- The fault: in pick-place the model moves above the object and writes `close` before it descends
  (`close` 0.78, `move` 0.22). With `move` forced at that step, practice goes from 11 to 20 of 20.
- The model also has the `close` habit of the 4B model in button-press. A forced replay shows that
  the habit is not the fault there (1 of 20), so no edit was tried on it.
- **Teacher edit: `habit.fit` on the MLP output matrix of layer 34 of 40, rank 1.**

| Held-out episodes (20 per task) | Base | Teacher edit |
|---|---|---|
| pick-place (target) | 12 | **20** |
| peg-insert-side | 5 | 8 |
| The other eight tasks | 86 | 86 |
| All ten tasks | 51.5% | **57.0%** (11 fixed, 0 broken, p = 0.001) |

| Measure | Teacher edit |
|---|---|
| Time | 30 minutes from the first base pass to the end of the practice trial |
| Weights changed | 0.50% (1 tensor of 704, 1 layer) |
| WikiText-2: top-token agreement with the original | 99.0% (KL 0.0004) |
| Battery | MMLU-Pro 45.5 to 45.5. CountBench 71.0 to 71.0. ChartQA 86.0 to 85.5. RefCOCO 46.5 to 47.5. GSM8K with thinking 90.0 to 89.0. |
| Random control (same fit, random direction) | pick-place 9 of 20 |

- Two parts of the harness assumed Qwen. A GLM layer has a norm between the MLP and the residual
  stream, so the attribution was not valid (error 1.06) and the fit wrote its value in the wrong
  place. The layout now reports that norm, and the attribution and the fit use it (error 0.006).
  Two battery scorers read only the Qwen answer format, and they now read both.
- The 30 minutes include 6 minutes on the button-press habit, which a forced replay ruled out, and
  6 minutes on the norm.
- No LoRA comparison and no second benchmark ran on this student.

## 7. The fault of the blog on the 4B model: thinking that does not end

GSM8K with thinking on. 64 practice items at a limit of 2,048 tokens, 128 held-out items at 4,096.

- Base model on the practice items: 48.4% correct, 62.5% of the replies at the limit. The stuck
  replies do not repeat text. The model plans again and questions its answer.
- At the few positions where the closing token of the thinking has a visible probability, the MLP
  of layer 30 pushes against it.
- **Teacher edit: `edits.remove_direction` on the MLP output matrix of layer 30, strength 2.**

| Held-out, 128 items | Base, thinking on | Teacher edit | Base, thinking off |
|---|---|---|---|
| Correct | 77.3% | **89.1%** (17 fixed, 2 broken, p = 0.0007) | 93.0% |
| At the token limit | 30.5% | 8.6% | 0.8% |
| Median thinking length (tokens) | 1,229 | 991 | no thinking |

| Measure | Teacher edit |
|---|---|
| Time | 8 minutes to the working edit. 14 minutes with the strength sweep and the random control. |
| Weights changed | 0.44% (1 tensor, 1 layer) |
| WikiText-2: top-token agreement with the original | 98.8% (KL 0.0006) |
| Battery | MMLU-Pro, CountBench, ChartQA, RefCOCO: within 0.5 points. GSM8K with thinking: 75.0 to 90.0. |
| Random control (random direction, same strength) | 46.9% on the practice items (base 48.4%) |

- A strength of 3 gives the same accuracy with a median thinking of 22 tokens. The model then skips
  its thinking. The thinking length is the second measure that shows this.
- The thinking-off prompt still beats the edit, as on the 0.8B model.

## 8. Long-horizon computer use: OSWorld on Qwen3.5-4B and Qwen3.8-27B

OSWorld runs real applications in an Ubuntu virtual machine and scores the final state with the
official evaluators. The student sees one screenshot and its action history, and writes one JSON
action per step (15 steps at most). Run directory: `runs/osworld-4b-teacher`. Ticket 30 holds the detail.

**For tasks that are possible: no edit candidate.** The base student solves 4 of 22 sampled tasks (18%, interval
7% to 39%): Thunderbird 2 of 2, Chrome 1 of 4, VS Code 1 of 3, and 0 of 13 in the six other applications.

The visible fault is a loop. 157 of 299 steps repeat the action of the step before, and 16 of 18
failed tasks hold a repeat. The loop is not the cause:

| Forced replay | What it changes | Failed tasks solved |
|---|---|---|
| Stop | The recorded actions up to the first repeat, then `done`. | 0 of 16 |
| Hint | The prompt says that the last action changed nothing and asks for a different action. | 0 of 18 |
| No `type` after `type` | The student must click or press a key between two typed fields. | 0 of 4 |

- The task is lost before the loop starts. The student does not know the procedure, and each
  corrected step exposes the next missing step.
- The recipe needs one recurring step with one correct token. The tasks of OSWorld do not share
  such a step, so the recipe stops at the forced replay, before any edit.
- The check that found this took 2 hours, most of it waits of the virtual machine (150 seconds per task).

**A wider scan found one portion with a fault of the habit kind.** A base scan of all 83 tasks of
Thunderbird, VS Code, and Chrome (16 solved) confirms the picture above for tasks that are possible.
It also shows a second group: 26 of the 368 OSWorld tasks are not possible by design, and the
correct reply there is the action `fail`. The base student never writes it and solves 0 of 26. A
prompt sentence makes the ability visible: with it, `fail` has a median probability of 0.41 at a
stuck step of a task that is not possible, and 0.025 in a possible task.

One rank-1 fit on one MLP matrix (`fail` against the other action names, margin 16) uses the steps
of 15 practice tasks that are not possible as targets, and the steps of 38 possible practice tasks as
contexts to keep. Ticket 32 holds the detail.

| Held-out tasks, in the virtual machine | Base | Teacher edit |
|---|---|---|
| Not possible (11, in five applications) | 0 | **10** (10 fixed, 0 broken, p = 0.002) |
| Possible (38) | 4 | 3 (0 fixed, 1 broken, p = 1.0) |
| Possible tasks that end with a false `fail` | 0 | 7 (the base failed all 7) |
| All 49 | 8% | 27% |
| Not possible (11): the same fit with a random direction | 0 | 0 |
| Not possible (11): base weights with a prompt sentence about such tasks | 0 | 7 (and 7 false `fail`) |

| Measure | Teacher edit |
|---|---|
| Time | 47 minutes from the end of the base scan to the choice of the margin. The scan took 63 minutes. |
| Weights changed | 0.51% (1 tensor, 1 layer, rank 1) |
| WikiText-2: top-token agreement with the original | 98.7% |
| Battery, largest drop | 1.0 point (GSM8K with thinking, 10 fixed and 11 broken) |

- **The gain is one behavior: to refuse a task that the machine cannot do.** The student does not
  get better at tasks that are possible.
- **The price is a false `fail`.** The student gives up on 7 of 38 possible tasks. The score does
  not show it, because the base failed those tasks. A user sees it.
- **A prompt sentence does most of the same.** With the base weights and one sentence about tasks
  that are not possible, the student solves 7 of the 11, with the same count of false `fail`. The
  edit gives 10 of 11 with no change to the prompt. With 11 tasks the difference is not significant.
- **The time is the time of the virtual machine.** One task takes 150 seconds, so the steps for the
  keys took 17 minutes and the scan 63 minutes. The fit and its checks took 28 minutes.

**The knowledge of the teacher in the prompt does not lift possible tasks.** A guide for VS Code in
the system prompt (two versions) leaves the 9 possible practice tasks at 2 and 3 solved, against 3
for the base. Both guides fix one task, replace all in a file, whose recipe is five fixed key
presses. Both break tasks that the base solved. The student follows an exact key recipe and fails
at each step that needs a look at the screen. So an edit that writes the guide into the weights has
no ceiling to reach, and none ran. Ticket 33 holds the detail. Thinking on gives the same 3 of 9. A second
base run with no change gives 1 of 9, and only 2 of 12 tasks repeat their actions, so one OSWorld
run has a noise of about 2 tasks in 9. The gain on tasks that are not possible (0 to 10 of 11,
with 0 of 11 for the random control) is far outside that noise. The change on possible tasks (4 to
3 of 38) is inside it.

**The 27B student: better at possible tasks, the same fault, and a margin that was too large.**
The 4-bit Qwen3.8-27B ran the same interface on 81 tasks (178 minutes, 16 seconds per step). It
solves 24 of 55 possible tasks, against 15 for the 4B student on the same tasks. It writes `fail`
in 3 of 26 tasks that are not possible, so the fault is still there. The recipe of the 4B run,
on layer 60, with the margin from a sweep on practice tasks (12), gives this on held-out tasks:

| Run, 27B student | Not possible (11) solved | Possible (17) solved | Possible tasks with a false `fail` |
|---|---|---|---|
| Base | 2 | 7 | 0 |
| Teacher edit, margin 12 | 8 (6 fixed, 0 broken, p = 0.03) | 2 (0 fixed, 5 broken, p = 0.06) | 10 |
| The same fit with a random direction | 1 | not run | |
| Base weights with the prompt sentence | 8 | 6 | 1 |
| Teacher edit, margin 8, chosen after the verdict | 8 | 5 (0 fixed, 2 broken) | 4 |

- **The edit fixes the fault and breaks possible tasks.** At margin 12 the student gives up on 10
  of 17 possible tasks, in most cases at step 4 or 5, in the middle of work that the base finished.
- **The battery does not show it.** MMLU-Pro 60 to 60, CountBench 94 to 94, ChartQA 75 to 75,
  RefCOCO 94 to 94, top-token agreement 98.95%, 0.20% of the weights. The possible tasks of the
  benchmark show the cost.
- **The sweep read too few steps.** It read 5 of 15 steps and showed a false `fail` in 3 of 19
  possible practice tasks. A read of every step shows 8 of 17 at margin 12 and 2 of 17 at margin 8.
  The fit is linear in the margin, so one solve gives every margin.
- **A weaker edit keeps the gain.** At margin 8 the tasks that are not possible stay at 8 of 11 and
  the possible tasks go from 7 to 5. This margin came from held-out steps, so it is a check and
  not a verdict.
- **The prompt sentence is the better tool on this student.** It gives the same 8 of 11 with 1
  false `fail`. On the 4B student the sentence gave 7 of 11 with 7 false `fail`.
- **Two failures of the harness on the way.** One new weight passed the largest value of the 4-bit
  format, and the fit needed more host memory than the machine had with three virtual machines up.
  The fit now clamps to the range, builds its system in place, and runs with no virtual machine up.

Run directory: `runs/osworld-27b-teacher`. Ticket 34 holds the detail.

## 9. Long-horizon robot plans: VLABench on Qwen3.5-4B

VLABench has a track for vision-language models: the model sees a robot scene and writes the plan
as a list of skill calls. The `Complex` dimension holds the long-horizon tasks. Three of its five
tasks have a fixed procedure of 5 to 9 steps (rearrange three books, hammer a nail and hang a
picture, a chemistry experiment). Run directory: `runs/vlabench-4b-teacher`. Ticket 31 holds the detail.

**Partial positive result.** The base student writes a default plan of `pick` and `place`. One fit
on one MLP matrix, with 46 target groups, writes the skill name and the end of each step of the
three procedures.

| Held-out, plan-only mode | Base | Teacher edit |
|---|---|---|
| Three target tasks (150): plans with all skill names correct | 0 | **91** (p = 8e-28) |
| Skill match | 0.200 | 0.864 |
| Entity match | 0.179 | 0.270 |
| Official total score | 0.151 | 0.454 |
| Fully correct plans | 0 | 0 |
| The two other long-horizon tasks (100): official total score | 0.232 | 0.239 |
| Five short dimensions (200): official total score | 0.429 | 0.428 |
| Five short dimensions (200): fully correct plans | 24 | 22 (0 fixed, 2 broken) |
| Random control (75): skill match | 0.207 | 0.217 |

| Measure | Teacher edit |
|---|---|
| Time | 63 minutes to the first trial that works, with one lost pass of 11 minutes |
| Weights changed | 0.52% (1 tensor, 1 layer, rank 6). The tensor changes by 106% of its norm. |
| WikiText-2: top-token agreement with the original | 98.0% |
| Battery | MMLU-Pro 48.0 to 45.5 (0 fixed, 5 broken, p = 0.06). GSM8K with thinking 75.0 to 70.0 (p = 0.33). CountBench 85.0 to 84.5. ChartQA and RefCOCO equal. |

- **The plan-only mode is a condition.** The reply must start with the code fence, so that the
  student writes the plan with no prose before it. With prose, the plan follows the prose: the
  same edit moves skill match from 0.268 to 0.296 there.
- **The edit writes the procedure, not the objects.** The forced replay with perfect names has the
  same ceiling: 0 of 150 plans fully correct. Which book is the oldest and which object is the
  hammer stay with the student.
- **One line in the prompt does the same.** A sentence that states the procedure gives skill match
  0.90 in the official mode. The edit is for a user who cannot or does not want to change the prompt.
- **The price is higher than for one-step edits.** The battery loses 2.5 points on MMLU-Pro, against
  1 point at most before. A margin of 8 costs less and gains less: all names correct in 24 of 150
  plans, total score 0.380, MMLU-Pro 48.0 to 46.5, top-token agreement 98.4%.

## What the blocks show

- A closed-form edit works when a fault hides an ability that the model has. The 0.8B model knew
  when to close its thinking, and three parts blocked it.
- A closed-form edit also works on the 4B model for a habit that is wrong in one context, when the
  right action already has a visible probability. This holds in three cases: two robot tasks and the
  email tasks of a web benchmark. Each edit changed its target and nothing else that we measured.
- The time to a working edit fell with each use of the recipe: 27 minutes, 15 minutes, and 19
  minutes on a new benchmark.
- The recipe is not specific to one teacher or one model family. A second teacher repeated an edit
  from the playbook in 30 minutes, and the recipe gave a held-out gain on a GLM student in 30 minutes.
- The recipe scales to a 27B student in a 4-bit format. The same fault, the same contexts, and a new
  layer number gave a larger gain there. The time grows with the cost of a forward pass.
- A closed-form edit did not fix a wrong choice between candidates from the input (two screen
  elements, two positions). Training fixes that kind in 15 minutes.
- The habit recipe extends to a procedure: 46 targets in one fit gave the skill names of plans with
  5 to 9 steps. The cost on the battery grew with the size of the edit.
- A realistic desktop benchmark gave the recipe one target. In OSWorld the failures on possible
  tasks come from procedures that the student does not know, and no step repeats across tasks. The
  refusal of a task that is not possible is one token with a default against it, and one edit fixed it.
- The refusal fault is also in the 27B student, and the recipe fixes it there. A margin does not
  carry from one student to the next: the margin of the sweep cost 5 of 7 solved possible tasks, and
  a read of every step would have shown that before the run. A battery of general questions did not show it.
- On the stronger student a prompt sentence did the work of the edit with less cost. The edit adds
  the most where the student does not use the sentence well.
- A prompt sentence can be the test for a hidden ability. The correct token had no probability with
  the plain prompt and a clear one with the sentence.
- A prompt sentence or an edit changes a default: one token, one fixed list of names, one fixed list
  of keys. Neither adds an ability. A guide with the procedures of VS Code in the prompt did not
  lift the possible desktop tasks.
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

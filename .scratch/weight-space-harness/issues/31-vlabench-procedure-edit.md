# Run the long-horizon tasks of VLABench on the 4B model and try the recipe on them

Type: task
Status: resolved

## Question

VLABench (Zhang et al., 2024) has a track for vision-language models: the model sees a robot scene
and an instruction, and writes the plan as a list of skill calls. The `Complex` dimension holds the
long-horizon tasks, with plans of 2 to 10 steps. What does the base Qwen3.5-4B score there, which
kind of fault do its plans show, and does a closed-form edit give a held-out gain?

The user allowed the install on 2026-11-11. The run uses only the evaluation dataset from the
Hugging Face Hub (`VLABench/vlm_evaluation_v1.0`, 4,500 examples) and the prompt file of the
repository. It needs no simulator. Module: `wsp/bench/vlabench_plan.py`. Run directory:
`runs/vlabench-4b-teacher`.

## Answer

**A partial positive result. One stacked fit on one MLP matrix teaches the student the procedure of
three long-horizon tasks: the skill names of all steps are correct in 91 of 150 held-out plans
(base: 0), and the official total score goes from 0.151 to 0.454. No plan becomes fully correct,
because the objects of each step stay wrong. The edit works only when the reply starts with the
plan, and it costs more general ability than the earlier edits.**

### The portion and the split

- `Complex` has five tasks with 100 examples each. Even example numbers are practice, odd numbers
  are held-out.
- Three tasks have one fixed procedure: `book_rearrange` (pick, push, pull for each of three books),
  `hammer_nail_and_hang_picture` (pick, push, place, pick, push), and `take_chemistry_experiment`
  (pick, pour, place, three times). They are the target.
- `cook_dishes` and `texas_holdem` have plans of pick and place with a varying length. They are the
  first near-transfer set. The second is 200 examples of the five short dimensions.

### Base and the fault

- Official mode (the reply holds prose, then the plan): 2 of 250 practice plans are fully correct.
  Skill match is 0.19 to 0.28 on the three target tasks.
- The student writes a default plan of `pick` and `place` with 2 to 6 steps. It does not know the
  procedure of the task.
- In the official mode the plan is decided in the prose. At the second step of the plan the correct
  skill name has a median probability of 0.00001. A forced replay of the names against that prose
  gives broken plans.
- **Plan-only mode:** the reply starts with the code fence, so the student writes no prose. This
  is a decoding option, as thinking off is. There the correct name has a visible probability and
  loses to a default: `push` 7%, `pull` 9%, `pour` 13%, `place` 21% to 35%. This is the habit fault.
- Forced replay in the plan-only mode (the names of the procedure, the objects of the student):
  skill match 1.0, total score about 0.5, fully correct plans 0 of 150. This is the ceiling.
- The cheap option: one line in the prompt that states the procedure gives the same picture in the
  official mode (skill match 0.25 to 0.90, 2 of 60 plans fully correct).

### The edit

- One fit (`habit.fit`) on `layers.27.mlp.down_proj`, margin 12, with 46 target groups: for each
  step of each procedure, the position that writes the skill name (correct name against the other
  eight skill names) and the position that ends the step (` },` to go on, ` }` after the last step).
- Keys: 25 practice examples per task, from plans with forced names and the objects of the student.
  1,150 target keys. Keep keys: 695 name and step-end positions of plans of other tasks. Protection:
  all positions of 48 plans of other tasks, the prompt positions of 15 target prompts, neutral text.
- Check on the 75 practice examples that the fit did not use: the probability of the procedure
  token at the target positions rises to 0.76 to 0.98.

### Held-out verdict, plan-only mode

| Set | Measure | Base | Edit |
|---|---|---|---|
| Three target tasks (150) | Plans with all skill names correct | 0 | **91** (p = 8e-28) |
| | Skill match | 0.200 | 0.864 |
| | Entity match | 0.179 | 0.270 |
| | Skill with entity match | 0.113 | 0.303 |
| | Official total score | 0.151 | 0.454 |
| | Fully correct plans | 0 | 0 |
| book_rearrange (50) | Skill match / all names correct | 0.156 / 0 | 0.967 / 45 |
| hammer_nail_and_hang_picture (50) | Skill match / all names correct | 0.232 / 0 | 0.896 / 37 |
| take_chemistry_experiment (50) | Skill match / all names correct | 0.211 / 0 | 0.729 / 9 |
| cook_dishes, texas_holdem (100) | Official total score | 0.232 | 0.239 |
| Five short dimensions (200) | Official total score | 0.429 | 0.428 |
| | Fully correct plans | 24 | 22 (0 fixed, 2 broken, p = 0.5) |

- Random control (the same fit with random directions, 75 examples): skill match 0.207 to 0.217,
  0 plans with all names correct.
- Official mode with prose, the same edit (75 examples): skill match 0.268 to 0.296, total score
  0.205 to 0.197. **The edit has no effect there.**
- The edit changes 33 of 100 and 57 of 200 plans of the other tasks, with no change of the mean scores.

### Size, deviation, and general ability

- One tensor, 0.52% of the weights, rank 6. The change of that tensor is as large as the tensor
  (relative change 1.06, cosine to the original 0.69). The earlier edits changed their tensor less.
- Deviation on neutral text: mean KL 0.0017, top-token agreement 98.0%, perplexity 13.377 to 13.406.
- Battery: MMLU-Pro 48.0 to 45.5 (0 fixed, 5 broken, p = 0.06). GSM8K with thinking 75.0 to 70.0
  (6 fixed, 11 broken, p = 0.33). CountBench 85.0 to 84.5. ChartQA and RefCOCO do not change.
- `tools.reset_edit()` restores the weight exactly.

### The smaller edit: margin 8

| Measure | Margin 8 | Margin 12 |
|---|---|---|
| Plans with all skill names correct (150) | 24 | 91 |
| Skill match | 0.692 | 0.864 |
| Official total score | 0.380 | 0.454 |
| Relative change of the tensor | 0.70 | 1.06 |
| Top-token agreement on neutral text | 98.4% | 98.0% |
| MMLU-Pro (base 48.0) | 46.5 (0 fixed, 3 broken) | 45.5 (0 fixed, 5 broken) |
| GSM8K with thinking (base 75.0) | 71.0 (10 fixed, 14 broken) | 70.0 (6 fixed, 11 broken) |

The gain and the cost both grow with the margin. No margin gives the gain at no cost.

### Time

- 63 minutes from the start of the base run to the first trial that works, with 1,525 read passes
  and 181 generate passes. The time holds one lost pass of 11 minutes (a defect in a table of the
  run script) and the 11-minute base run in the official mode.
- The verdict took 30 minutes more.

### What it means

- The habit recipe extends from one step to a procedure of 5 to 9 steps: 46 targets in one fit on
  one matrix. This is the first edit of the project that writes a multi-step plan.
- The edit writes the procedure, not the perception. The objects of each step (which book is the
  oldest, which object is the hammer) stay with the student, and they stay wrong. This is the same
  limit as before: a wrong choice from the input has no closed-form fix here.
- The price grows with the size of the edit. 46 targets cost 2.5 points on MMLU-Pro, where one
  target cost at most 1 point.
- The edit holds only in the decoding mode of the fit. With prose before the plan, the plan follows
  the prose, and the prose does not change.
- Benchmark note: `take_chemistry_experiment` has one reference plan for all 100 examples, and
  `hammer_nail_and_hang_picture` has two. A method that stores those plans would score 100% there.
  The edit here writes the skill names only, so it does not use this.

### Harness notes

- `vlabench_plan.score` reports the strict `correct` and the official metrics. The official
  `total_score` reaches 0.8 at most, because the official code adds two of the four weights.
- The run script `trial` now saves the rows before it prints the table.

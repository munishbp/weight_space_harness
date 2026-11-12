# Run OSWorld on the quantized 27B student and carry the `fail` edit to it

Type: task
Status: resolved

## Question

The 4B student solves 16 of 83 scanned OSWorld tasks, and one edit makes it refuse the tasks that
are not possible ([ticket 32](32-osworld-fail-edit.md)). Does a stronger student, the 4-bit
Qwen3.8-27B, solve more possible tasks, does it have the same fault, and does the same recipe fix it?

## Answer

**The 27B student is better at possible tasks (24 of 55 against 15 for the 4B student on the same
tasks) and has the same fault: it writes `fail` in 3 of 26 tasks that are not possible. The edit
fixes the fault (held-out 2 to 8 of 11), but at the margin that the sweep chose it also gives up on
10 of 17 possible held-out tasks, and the possible tasks go from 7 to 2 solved. One sentence in the
prompt, with the base weights, gives the same 8 of 11 and leaves the possible tasks at 6 of 17. So
on this student the prompt sentence is the better tool, and the edit at that margin is a net loss.**

**A check after the verdict shows that the margin was the defect, not the recipe. The same edit at
margin 8 keeps the gain (2 to 8 of 11) and costs far less: the possible tasks go from 7 to 5, with
4 false `fail`. The sweep had read 5 of 15 steps and did not show the cost. This check chose its
margin with held-out steps, so it is not a held-out verdict. The prompt sentence stays a little
better at no cost in weights.**

Run directory: `runs/osworld-27b-teacher` (scripts `b01` to `b08`). The interface, the prompt, the
split, and the 15-step limit are the same as in the 4B run.

### Base runs

- 81 tasks in 178 minutes on three virtual machines: 16 seconds per step, about 360 seconds per task.

| Tasks | 27B base solved | 4B base on the same tasks |
|---|---|---|
| Practice, possible: Thunderbird 8, VS Code 9, Chrome 21 | 17 of 38 (4, 4, 9) | 12 |
| Held-out, possible: Thunderbird 7, VS Code 10 | 7 of 17 (2, 5) | 3 |
| Practice, not possible | 1 of 15 | 0 |
| Held-out, not possible: GIMP 4, os 4, VS Code 1, Chrome 1, VLC 1 | 2 of 11 | 0 |

- A forced stop at the first repeated action solves 0 of 7 failed practice tasks, as on the 4B
  student. The failures on possible tasks share no step.
- The two held-out tasks that the base ends with `fail` do not repeat: the random control ends
  neither of them with `fail`. The base rate is about 1 or 2 of 11.

### The fit

- `model.language_model.layers.60.mlp.down_proj` (layer 60 of 64), one direction: `fail` against
  the nine other action names. The recipe is the recipe of ticket 32.
- A read of one prompt with a full-size screenshot takes 2.5 seconds, so the fit uses 7 steps of
  each task (0, 1, 2, 4, 7, 10, 13) and the sweep reads 5 steps (0, 1, 2, 7, 13).
- Sweep, with a fit on one half of the practice tasks and a read on the other half:

| Margin | Not possible: `fail` wins at a read step | Possible: `fail` wins at a read step | Of these, solved by the base |
|---|---|---|---|
| base | 0 of 7 | 0 of 19 | 0 |
| 8 | 4 of 7 | 1 of 19 | 0 |
| 12 | 6 of 7 | 3 of 19 | 0 |

- The rule was fixed before the read: the smallest margin with 5 of 7. That is margin 12.
- Final fit: 100 target steps of 15 practice tasks, 232 keep steps of 38 practice tasks.

### Held-out verdict, in the virtual machine

| Run | Not possible (11) solved | Possible (17) solved | Possible tasks with a false `fail` |
|---|---|---|---|
| Base | 2 | 7 | 0 |
| Teacher edit, margin 12 | **8** (6 fixed, 0 broken, p = 0.03) | **2** (0 fixed, 5 broken, p = 0.06) | 10 |
| The same fit with a random direction | 1 | not run | |
| Base weights with the prompt sentence | 8 (6 fixed, 0 broken, p = 0.03) | 6 (1 fixed, 2 broken) | 1 |

- With the edit the student writes `fail` at step 0 in six tasks that are not possible, and at step
  1 and step 8 in two more. The three misses: the VS Code language task (also a miss on the 4B
  student) and two terminal tasks.
- **The false `fail` is the cost.** 10 of 17 possible tasks end with `fail`. The base solved 4 of
  these 10. The `fail` comes at step 0 or 1 in three tasks and at step 4, 5, or 11 in seven tasks:
  the student stops in the middle of work that the base finished. A fifth broken task does not end
  with `fail`; its actions differ from the base run.
- The random direction gives 1 of 11, so the gain on tasks that are not possible comes from the direction.
- The prompt sentence is the sentence of the 4B run. It gives the same 8 of 11, with `fail` at
  step 0 to 7, and one false `fail`. On the 4B student the sentence gave 7 of 11 with 7 false
  `fail`: the larger student uses the sentence with more care.
- `tools.reset_edit()` restores the stored bytes exactly.

### Why the sweep did not show the cost

The sweep read 5 of 15 steps and showed a false `fail` in 3 of 19 possible tasks. A task ends at the
first step where `fail` wins, so each step that the read leaves out hides a part of the cost. A
read of every recorded step of the held-out base runs, after the verdict:

| Margin | Not possible: `fail` wins at some step | Possible: `fail` wins at some step | Of these, solved by the base |
|---|---|---|---|
| 12 | 7 of 11 (in the machine: 8) | 8 of 17 (in the machine: 10) | 3 (in the machine: 4) |
| 8 | 6 of 11 | 2 of 17 | 1 |
| 6 | 5 of 11 | 2 of 17 | 1 |

- The read of every step agrees with the run in the machine at margin 12. So the defect was the
  read with gaps, not the read as a method.
- The values of the fit are linear in the margin. The weight at margin m is
  `W0 + (m / 12) * (W_edit - W0)`, so one solve gives every margin and the read above needed no new fit.
- This read uses held-out steps. It is a diagnosis, not a verdict.

### Check at margin 8, in the virtual machine

The weight is `W0 + (8 / 12) * (W_edit - W0)`, with no new fit. Same 28 held-out tasks.

| Run | Not possible (11) solved | Possible (17) solved | Possible tasks with a false `fail` | Of these, solved by the base |
|---|---|---|---|---|
| Base | 2 | 7 | 0 | 0 |
| Edit, margin 12 (the verdict) | 8 | 2 | 10 | 4 |
| Edit, margin 8 (after the verdict) | 8 (6 fixed, 0 broken, p = 0.03) | 5 (0 fixed, 2 broken, p = 0.5) | 4 | 2 |
| Base weights with the prompt sentence | 8 | 6 | 1 | 1 |

- At margin 8 the student writes `fail` later on the tasks that are not possible: at step 0, 1, 3,
  3, 3, 6, 7, and 8, against step 0 or 1 at margin 12.
- Both broken tasks are VS Code tasks that end with a false `fail` (at step 4 and step 8).
- The read of every step predicted 2 false `fail` at margin 8, and the machine gave 4. The read
  uses the steps of the base run, and the run with the edit takes other steps. The read is a lower bound.
- Margin 8: the tensor changes by 25% of its norm, mean KL 0.0006, top-token agreement 99.08%,
  and the battery is the same as the base (60, 94, 75, 94). The reset restores the stored bytes.
- The margin came from a read of held-out steps. A clean verdict needs new held-out tasks, for
  example the 21 possible Chrome tasks of the held-out half, which the 27B student did not run.

### Size, deviation, and general ability (margin 12)

- One tensor, 0.20% of the weights of the model. The fit is rank 1; the stored 4-bit format rounds
  it, and the tensor changes by 37% of its norm.
- One of 89 million new weights passed the largest value that the stored format holds (1.172). The
  fit clamps to that range.
- Deviation on neutral text: mean KL 0.0007, top-token agreement 98.95%, perplexity 8.714 to 8.711.
- Battery, 100 items of each set: MMLU-Pro 60 to 60 (1 fixed, 1 broken), CountBench 94 to 94,
  ChartQA 75 to 75, RefCOCO 94 to 94. The battery does not show the cost of the edit. The possible
  tasks of the benchmark show it.

### Time

- Base runs: 178 minutes (scan of 43 practice tasks 97 minutes, 38 more tasks 81 minutes).
- From the end of the base runs to the final weight: 30 minutes of compute (keys of the two halves
  7, protection 1, base read 3, two margins 6, keys of all practice tasks 13, one solve 0.5).
- Held-out run with the edit: 39 minutes for 28 tasks (a task that ends with `fail` is short).
  Battery and deviation 10 minutes, random control 20 minutes, prompt sentence 50 minutes.
- After the verdict: read of every step at three margins 23 minutes, run at margin 8 in the machine 45 minutes.

### Two failures on the way, and the fixes

- **Range of the stored format.** The first sweep stopped: `tools.save_edit` raised, because one new
  weight passed the limit of the NVFP4 format. The run scripts now clamp to the range and print the
  count of clamped weights (1, and 0 for the random direction).
- **Host memory.** The second sweep ended when the host had no memory left, and the system killed the
  kernel. The solve has a system of 21,760 by 21,760 in float64: 3.8 GB per copy, and the code
  held five copies. Three virtual machines held 12 GB at the same time. Two fixes:
  `edits.fit_keys_to_values` now builds the system in place (three copies, the same result to
  rounding), and the fits run with no virtual machine up (`b05a_prepare.py` makes both new weights,
  then `b05b_verdict.py` runs with the machines).

### What it means

- The fault of ticket 32 is not a fault of a small model only. A student that is about seven times larger
  still writes `fail` in about 1 of 10 tasks that are not possible.
- The recipe transfers to the larger student and fixes the fault. The margin that the 4B student
  needed (16) is too large here: margin 12 already costs most of the possible tasks.
- A refusal edit needs a read of every step of possible tasks before a margin is chosen, and the
  count that matters is the count of tasks that the base solved.
- On the stronger student the prompt sentence does the same work with almost no cost. The edit has
  one advantage left: it needs no change to the prompt. At margin 8 the edit is close to the
  sentence (5 against 6 of 17 possible tasks, inside the noise of one run) and writes more false `fail` (4 against 1).
- 11 and 17 held-out tasks are few. The interval of 8 of 11 is 43% to 90%.
- The possible tasks stay without an edit candidate, as in tickets 30 and 33. The larger student
  is the lever there: 44% against 27% of the same 55 tasks.

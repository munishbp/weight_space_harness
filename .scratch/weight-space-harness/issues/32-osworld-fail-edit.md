# Scan more of OSWorld for a fault that tasks share, and edit it

Type: task
Status: resolved

## Question

The first OSWorld sample had 22 tasks, and no fault there had a forced replay with a gain
([ticket 30](30-osworld-portion.md)). A sample of 22 is thin. Does a wider scan find a fault that
several tasks share, and does the habit recipe fix it on held-out tasks?

## Answer

**Yes, for one portion. 26 of the 368 OSWorld tasks are not possible by design, and the correct
reply there is the action `fail`. The base student never writes it. One rank-1 fit on one MLP
matrix makes the student write `fail` on those tasks: 0 to 10 of 11 held-out tasks (a prompt
sentence with the base weights gives 7 of 11). The possible
held-out tasks go from 4 to 3 of 38 solved, and the battery stays within 1 point. The price: the
student also gives up on 7 of 38 possible tasks, all of which the base failed.**

Run directory: `runs/osworld-4b-teacher` (scripts `b05` to `b12`).

### The scan

- Every task of Thunderbird, VS Code, and Chrome (83 tasks), on three virtual machines in 63 minutes.
- The split is fixed before any look: even positions in `test_all.json` are practice, odd positions
  are held-out. The teacher read the actions of practice tasks only.

| Application | Practice solved | Held-out solved |
|---|---|---|
| thunderbird | 3 of 8 | 2 of 7 |
| vs_code | 3 of 12 | 1 of 11 |
| chrome | 6 of 23 | 1 of 22 |
| **All** | **12 of 43** | **4 of 40** |

- The possible tasks show the picture of ticket 30: each failure has its own missing steps.
- The scan showed a second group. The evaluator of 26 tasks is `infeasible` (GIMP 9, os 5, VS Code
  4, Chrome 3, VLC 2, three others 1 each). Such a task scores 1 only when the last action is
  `fail`. Examples: "change the display language of VS Code to Arabic without an extension", a GIMP
  task for a feature that GIMP does not have. The base student solves 0 of 26.

### The fault

- Without help, `fail` has no probability, on tasks that are not possible and on possible tasks
  (median of the largest value over the steps: 0.0006 against 0.0008).
- The cheap option shows that the ability exists. With one sentence in the prompt ("If this task is
  not possible, write `fail`"), the probability of `fail` at the last recorded step has a median of
  0.41 on the 5 practice tasks that are not possible and 0.025 on the 38 possible ones (AUC 0.84).
- So the student can tell the two kinds, and its default never gives up. This is the habit fault.

### The edit

- `habit.fit` on `layers.27.mlp.down_proj`, one direction: `fail` against the nine other action names.
- Target keys: every recorded step of the 15 practice tasks that are not possible (225 steps).
  Keep keys: every recorded step of the 38 possible practice tasks (528 steps).
- Check with a fit on one half of the practice tasks and a read on the other half:

| Margin | Tasks that are not possible: `fail` wins at some step | Possible tasks: `fail` wins at some step | Of these, solved by the base |
|---|---|---|---|
| 8 | 1 of 7 | 0 of 19 | 0 |
| 12 | 3 of 7 | 2 of 19 | not read |
| 16 | 5 of 7 | 2 of 19 | 0 |
| 20 | 5 of 7 | 3 of 19 | 0 |
| 24 | 5 of 7 | 6 of 19 | 1 |

- Margin 16 is the smallest margin with 5 of 7. The final fit uses all practice tasks.

### Held-out verdict, in the virtual machine

| Held-out tasks | Base solved | Edit solved | Paired |
|---|---|---|---|
| Not possible (11): GIMP 4, os 4, VS Code 1, Chrome 1, VLC 1 | 0 | **10** | 10 fixed, 0 broken, p = 0.002 |
| Possible (38): Thunderbird 7, VS Code 10, Chrome 21 | 4 | 3 | 0 fixed, 1 broken, p = 1.0 |
| All 49 | 4 (8%) | 13 (27%) | |

- The student writes `fail` at step 0 in 7 tasks and at step 1 in 3 tasks. The one miss is the
  VS Code language task, where the student waits to the step limit.
- **False `fail`.** The edit ends 7 of the 38 possible tasks with `fail` (at step 0, 1, or 2). The
  base failed all 7, so the score does not show it. A user sees it: the student gives up on about
  one possible task in five.
- The one broken task (Thunderbird, two local folders) does not end with `fail`. Its actions differ
  from the base run after the first steps. One run per task cannot separate this from run-to-run noise.
### Controls on the same held-out tasks

| Run | Not possible (11) solved | Possible (38) solved | Possible tasks with a false `fail` |
|---|---|---|---|
| Base | 0 | 4 | 0 |
| Teacher edit, margin 16 | 10 | 3 | 7 |
| The same fit with a random direction | 0 (no `fail` written) | not run | |
| Base weights with the prompt sentence | 7 | 3 | 7 |

- The random direction has no effect, so the gain comes from the direction.
- The prompt sentence does most of the same: 7 of 11, with the same count of false `fail`. With
  the sentence the student writes `fail` later (step 1 to 7). With the edit it writes `fail` at
  step 0 or 1. With 11 tasks, 10 against 7 is not a significant difference.
- The task that the edit breaks (Thunderbird, two local folders) also fails in the run with the
  prompt sentence, with no `fail`. It looks like a task with an unstable base result.
- `tools.reset_edit()` restores the weight exactly.

### Size, deviation, and general ability

- One tensor, 0.51% of the weights, rank 1. The tensor changes by 50% of its norm.
- Deviation on neutral text: mean KL 0.0009, top-token agreement 98.7%, perplexity 13.377 to 13.390.
- Battery: MMLU-Pro 48.0 to 48.0, CountBench 85.0 to 85.0, ChartQA 85.0 to 84.5, RefCOCO 90.0 to
  90.0, GSM8K with thinking 75.0 to 74.0.

### Time

- 47 minutes from the end of the base scan to the choice of the margin: 17 minutes for the base
  run on the 19 other tasks that are not possible (their steps are the keys), 5 minutes for the two
  probability reads, 15 minutes for keys, protection, and sweep, 8 minutes for the detail read.
  626 read passes and 285 generate passes.
- The base scan before it took 63 minutes. One task takes 150 seconds in the virtual machine, so
  the time is the time of the machine, not of the edit.

### What it means

- This is the first gain of the project on a realistic long-horizon computer use benchmark. The
  portion is one behavior: to refuse a task that the machine cannot do.
- It is a habit fault of the same kind as before: one token, a default that blocks it, and an
  ability that a prompt sentence makes visible. The probability of the correct token was not
  visible with the plain prompt. The prompt sentence was the test that found the ability.
- The edit generalizes to tasks that the fit did not see, in five applications (GIMP, the
  terminal, VS Code, Chrome, VLC). The fit saw other tasks of the same applications.
- The edit does not make the student better at tasks that are possible. Tickets 30 and 31 hold
  that limit.
- 11 held-out tasks are few. The interval of 10 of 11 is 62% to 98%.

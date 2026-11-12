# Run one portion of OSWorld on the 4B model and try the recipe on it

Type: task
Status: resolved

## Question

MiniWoB++ tasks have 2 to 10 steps. OSWorld is the realistic computer use benchmark that the goal
names: real applications in an Ubuntu virtual machine. What does the base Qwen3.5-4B score on one
portion of it through screenshots, which kind of fault do its failures show, and does the habit
recipe give a held-out gain there?

The user allowed the downloads on 2026-11-11: the `desktop-env` package from PyPI, the Ubuntu VM
image from the Hugging Face Hub, the Docker image `happysixd/osworld-docker`, and a clone of the
OSWorld repository for the task files. The user also allowed the VLABench install.

The task, in order:

1. Bring up the environment through `wsp/bench/osworld_server.py` and `wsp/bench/osworld_screen.py`.
   Record the time per task.
2. Run the base model on a sample of every application. Choose one application as the portion: the
   base score must leave room, and the failures must be readable.
3. Split the tasks of that application into practice and held-out before the search.
4. Sort the failures by cause. Run a forced replay where a wrong action name is the cause.
5. If a fault of the habit kind or the blocked-token kind exists, edit and run the verdict. If the
   failures are wrong positions, record a negative result.

The model card reports 35.6 on OSWorld-Verified for Qwen3.5-4B, with an agent setup that it does
not state.

## Answer

**Negative result. The 4B student solves 4 of 22 sampled OSWorld tasks, and no fault of a kind that
a closed-form edit fixes shows in the failures.** Three forced replays, each for one visible fault,
solve 0 failed tasks. No edit ran, because the recipe says to stop where a forced replay finds no
ceiling. Run directory: `runs/osworld-4b-teacher`.

### Setup

- Environment: `desktop-env` 1.0.3 in `.venv-osworld`, the Ubuntu VM in the Docker provider, the
  official evaluators. The student sees one 1920 x 1080 screenshot and the action history, and
  writes one JSON action. Limit: 15 steps. No accessibility tree.
- One task takes about 150 seconds, of which 80 seconds are fixed waits of the reference runner.
  Up to three VMs ran next to one model (`episodes(..., sockets=...)` in `setup.py`), 4 GB of
  memory each.
- The host has no `ip_tables` kernel module, so the VM runs with user-mode networking
  (`--user-network`).

### Base sample: `test_small` without the multi-application tasks

| Application | Solved | Tasks |
|---|---|---|
| chrome | 1 | 4 |
| gimp | 0 | 2 |
| libreoffice_calc | 0 | 3 |
| libreoffice_impress | 0 | 2 |
| libreoffice_writer | 0 | 2 |
| os | 0 | 2 |
| thunderbird | 2 | 2 |
| vlc | 0 | 2 |
| vs_code | 1 | 3 |
| **All** | **4 (18%)** | **22** |

The model card reports 35.6 on OSWorld-Verified with an agent setup that it does not state. Our
setup is simple (one screenshot, 15 steps, no tree), so a lower score is plausible. With 22 tasks
the 95% interval is wide (7% to 39%).

### The visible fault, and three forced replays

- 157 of 299 steps repeat the action of the step before. 16 of 18 failed tasks hold a repeat, and
  14 of 18 run to the step limit. Example: in the Chrome task "delete the Amazon tracking data" the
  student opens the correct dialog, presses "Delete data", and then clicks the same point 10 more times.
- The probability of `done` at a repeat step is low: median 0.9%, mean 3.2%.

| Forced replay | What it changes | Failed tasks solved |
|---|---|---|
| Stop | The recorded actions up to the first repeat, then `done` (no model). | 0 of 16 |
| Hint | When the student repeats an action, the prompt says "The last action changed nothing. Write a different action." and asks again. | 0 of 18 |
| No `type` after `type` | The action name `type` is not possible directly after a `type` (the student wrote two fields with no key or click between them in 4 tasks). | 0 of 4 |

- The hint was needed 86 times. The student wrote the same action again 75 times.
- With `type` forbidden, the student clicks the second field and types there, and then fails at the
  next step (for example it replaces one match, not all).

### What it means

- The loop is a symptom. The task is lost before the loop starts: the student does not know the
  procedure, and each corrected step exposes the next missing step.
- This is the third kind of fault of the playbook (a missing ability, or a wrong choice from the
  input), across many different procedures. The habit recipe needs one recurring step with one
  correct token. OSWorld tasks do not share such a step.
- The positive computer-use result of this project stays the MiniWoB++ email group (2 to 10 steps,
  one shared missing step). On OSWorld a fine-tune with demonstrations is the fitting method.
- Time: 55 minutes for the base sample on one VM, 23 minutes for the hint pass on two VMs, about
  2 hours in total with the bring-up of the bridge.

### Interface notes

- 5 of 299 actions were invalid. All 5 gave the text of a `type` action as a list of one string.
- 3 of the 4 solved tasks never wrote `done`. The evaluator reads the final state, so they count.

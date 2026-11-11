# Install one simulator benchmark and run the base model

Type: task
Status: resolved
Blocked by: 14

## Question

Which simulator benchmark installs and runs on this machine, and what does the base Qwen3.5-4B score
on a small set of episodes through the chosen action interface?

Facts from 2026-11-10:

- Headless MuJoCo 3.2.2 with EGL rendering works on this machine, in a separate Python 3.10
  environment at `data/sim-env` (not in git).
- VLABench needs `rrt-algorithms`, which exists only as a git repository, and assets from two Google
  Drive archives. The permission check of the session blocked an install from third-party git
  repositories. The user must allow that install.
- The model evaluation script of VLABench scores a skill plan against a fixed dataset. It does not
  run the simulator. A run in the simulator needs the skill library of VLABench and extra code.
- `metaworld` and `ai2thor` exist on PyPI.

## Answer

Resolved on 2026-11-10. Run directory: `runs/mw-4b-teacher`. Code: `wsp/bench/metaworld_text.py`.

**Meta-World 3.1.1 installs from PyPI and runs headless. The base Qwen3.5-4B solves 126 of 200
practice episodes (63.0%, interval 56.1 to 69.4) of the MT10 tasks through the text waypoint
interface.** The model wrote valid JSON in every turn. 200 episodes take 4.3 minutes. The simulator
sets the pace, not the model.

| Task | Base model (20 seeds) | Reference plan (20 seeds) |
|---|---|---|
| reach | 20 | 20 |
| pick-place | 20 | 20 |
| drawer-close | 20 | 20 |
| window-close | 20 | 20 |
| peg-insert-side | 19 | 20 |
| window-open | 16 | 20 |
| push | 10 | 14 |
| button-press-topdown | 1 | 20 |
| drawer-open | 0 | 20 |
| door-open | 0 | no plan |

A reference plan is a hand-written sequence of primitives (`reference_plan`). It shows that the
interface can solve a task. Door-open has no plan, because one move of 80 control steps cannot drag
the door to the goal.

**A defect found and fixed.** The observation that `reset` returns holds a stale object position in
`window-close-v3` (10 mm in place of 210 mm). `Episode` now takes one step with no motion first.

### Failure sort

The model writes the same short habit in most tasks: move to the object, close, move to the goal.
The simulator shows which change to that plan makes a failing task succeed (10 seeds each):

| Fault | Tasks | Evidence |
|---|---|---|
| The model closes the gripper before a press. | button-press-topdown (1 of 20) | The plan of the model fails 0 of 10. The same plan with no `close` succeeds 8 of 10. |
| The first move goes to the goal, not to the object. | push (10 of 20), window-open (16 of 20) | "Move to the object, close, move to the goal" succeeds 10 of 10 in both tasks. Every failure starts with a move to the goal. |
| The model opens the gripper after it grasps. | door-open (0 of 20), drawer-open (0 of 20) | Door-open with no `open` succeeds 5 of 10. Drawer-open fails with and without it: it needs the hook motion of the reference plan. |

The first two faults are a wrong choice at one step, where the plan that works differs by one
primitive. They are the target of the teacher edit. Drawer-open needs a motion that the model does
not write at all.

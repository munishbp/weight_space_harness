# Install one simulator benchmark and run the base model

Type: task
Status: open
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

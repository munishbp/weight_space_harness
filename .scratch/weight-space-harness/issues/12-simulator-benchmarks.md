# Survey robot simulator benchmarks for a text-output model

Type: research
Status: resolved

## Question

Which robot simulator benchmarks with actions can run on one RTX 5090 workstation, and which of them
have published results for a general vision-language model that has no action head?

For each candidate (for example LIBERO, SimplerEnv, ManiSkill, CALVIN, RoboCasa, VLABench), report
the install cost, headless rendering support, the licence, the time for one episode, the task
structure that allows a "specific portion", and the published scores of general vision-language
models with the action interface that they used.

Findings: `../research/simulator-benchmarks.md`

## Answer

Resolved on 2026-11-10 by a research subagent. The findings file holds the sources. Nothing was
installed or tested on this machine.

- **Candidates covered:** LIBERO, SimplerEnv, ManiSkill3, CALVIN, RoboCasa, VLABench, RLBench,
  EmbodiedBench EB-Manipulation, Meta-World, BEHAVIOR-1K, and CaP-Bench (2026).
- **Only EB-Manipulation publishes Qwen scores from images with no action head.** Qwen2.5-VL-7B gets
  9.6%, Qwen2-VL-7B gets 0.0%, and GPT-4o gets 28.9%. The action is a discrete 7-D pose in bins, with
  at most 15 steps per episode. The prompt can include detection boxes and object poses.
- **The high no-action-head scores on LIBERO, ManiSkill3, and Meta-World (84.9%, 85.7%, 96.0%) come
  from one paper.** It used Claude Opus 4.5 to write code against privileged simulator state, not
  camera images. No such result was found for SimplerEnv, CALVIN, or RoboCasa.
- **VLABench has a skill library with a DSL output for a vision-language model.** The exact scores
  are in a chart only.
- **Install risk.** RLBench and EmbodiedBench need CoppeliaSim 4.1.0, which targets Ubuntu 20.04, and
  an X server or a virtual display. This machine runs Ubuntu 26.04. BEHAVIOR-1K needs Isaac Sim.
  LIBERO pins Python 3.8 and torch 1.11 with CUDA 11.3, which does not support the RTX 5090.
- **Open items.** Headless rendering is not tested for any candidate. The wall-clock time per episode
  is mostly not found.

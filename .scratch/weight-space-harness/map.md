# Map: weight-space harness

Label: wayfinder:map
Charted: 2026-11-10
Destination redrawn: 2026-11-10, after the user corrected the scope of the harness.

## Destination

A harness that lets a frontier teacher model edit the weights of a small open student model, so that
the student completes one specific long-horizon application well. The harness comes with a measured
verdict on whether the method works. The map ends when all six items exist:

1. **A general harness.** Any frontier model can be the teacher. Any open model can be the student
   (a larger Qwen, a smaller Qwen, GLM, or another family). No part depends on Claude or on Qwen.
2. **Calibration on Qwen3.5-0.8B.** The harness reproduces the edit that the blog published, and the
   teacher finds the same fault again with the harness tools.
3. **A computer use verdict on Qwen3.5-4B.** The first target is one ScreenSpot Pro slice.
4. **A robotics verdict on Qwen3.5-4B.** The target is one robot simulator benchmark with actions.
5. **A scaling run on Qwen3.8-27B.** The user wants it "to show that it does scale". The copies on
   this machine are quantized. The user decided to use the quantized NVFP4 copy and accepts that the
   result is not a direct scale. How to read and edit that copy is an open research ticket.
6. **A teacher playbook in plain text.** A fresh session of any frontier model can repeat a run from it.

Every verdict reports these six measures, plus the set from
[Choose the metric set for the writeup](issues/20-writeup-metrics.md):

- The gain on held-out items of the target.
- The wall-clock time and the number of forward passes.
- **Generalization.** The scores on a fixed battery of general benchmarks before and after the edit.
  The verdict reports each drop, large or small.
- **Edit size.** The share of weights, neurons, tensors, and layers that the edit changes.
- **Deviation from the original.** The distance in weight space and the distance in output space.
- The LoRA comparison at equal wall-clock time.

A negative result is a valid verdict.

## Notes

- **This map carries execution.** The verdict is empirical, so `task` tickets run experiments. This
  overrides the planning-only default of wayfinder.
- **Delegation.** On 2026-11-10 the user said: "I think you can do the rest without me. So, get
  started on it." The teacher makes the open decisions, including the `grilling` tickets, and writes
  each decision with its reason in the ticket. The user can overrule a decision at any time.
- **No pass and no fail.** The harness is the deliverable and 15 minutes per edit is the ideal. Every
  measured result is reported as information.
- **Teacher.** Claude is the teacher for every run of this effort. The harness itself is neutral: the
  teacher interface is a command line plus a plain-text playbook, so another frontier model with a
  shell tool can take the role.
- **Student.** The harness loads a student by its Hugging Face name. Code that knows one model family
  stays out of the core.
- **Agents.** A subagent uses Opus only (user rule). Research tickets go to a `/research` subagent.
- **Edit rules.** The method uses no gradient and no optimizer. Closed-form edits from activation
  statistics are allowed. A LoRA fine-tune with the same wall-clock time is a comparison only.
- **Time.** 10 to 15 minutes per edit is a goal, not a cap. The harness ledger records the wall-clock
  time of every run. Each block has a time box. A block that passes its time box ends with a recorded
  negative result.
- **Order.** Calibration, then computer use, then the simulator, then the 27B scaling run. There is no deadline. Simulator
  research runs in the background from the start.
- **Scope of a target.** The aim is one portion of a benchmark with generality kept. The aim is not
  the highest score on a full benchmark.
- **Blind test.** The teacher read the blog's answer for the 0.8B model in the charting session. The
  0.8B attempt is not blind. The 4B work is blind, because no published answer exists.
- **Repository.** `origin` is https://github.com/munishbp/weight_space_harness.git. Commit or push only
  when the user asks.
- **Tracker.** Local markdown. Research findings go to
  `.scratch/weight-space-harness/research/<name>.md` and the ticket links to that file.
- **Machine.** RTX 5090 with 32 GB VRAM, 45 GB RAM, 24 cores, 692 GB free disk on 2026-11-10.
- **Assets that exist.** `wsp/kernel.py`, `wsp/client.py`, `wsp/tools.py`. The kernel has not run
  against a model yet. `tools.tensor_metrics` passes a check on synthetic tensors. The edit manifest
  records the edit size and the weight-space distance. `tools.deviation` measures the
  output-space distance on text. `wsp/battery.py` holds the generalization battery. The environment is in `.venv` (uv, Python 3.13, torch
  2.15, transformers 5.22, flash-linear-attention 0.6). The Qwen3.5-4B weights are in the Hugging Face cache.
- **Skills.** `/grilling` for decision tickets, `/research` for research tickets, `/prototype` for
  the playbook ticket. The user's `CLAUDE.md` governs all code and prose.

## Decisions so far

- [What the blog method does and what it released](issues/01-blog-method.md): a frontier model edits
  weights through a Python kernel and a small budget of forward passes, with no gradients. The
  harness code is not released. The largest model tested was 0.8B, text only, at 8.5 to 20 hours a run.
- [Qwen3.5-4B facts](issues/02-qwen35-4b-facts.md): the model takes images and video, has 32 layers
  of width 2560, and scores 60.3 on ScreenSpot Pro per its model card.
- [ScreenSpot Pro format and the Qwen3.5 grounding prompt](issues/03-screenspot-pro-format.md): 1581
  samples in 26 annotation files, a click inside the bbox is correct, and the model writes
  coordinates on a 0 to 1000 scale in a `computer_use` tool call.
- [Survey robot simulator benchmarks for a text-output model](issues/12-simulator-benchmarks.md):
  eleven candidates, among them LIBERO, SimplerEnv, ManiSkill3, CALVIN, RoboCasa, VLABench, RLBench,
  and EmbodiedBench EB-Manipulation. Only EB-Manipulation publishes Qwen scores with no action head
  (Qwen2.5-VL-7B 9.6%, GPT-4o 28.9%).
- [Survey Qwen3.5 spatial outputs and action interfaces](issues/13-qwen-spatial-outputs.md): point
  and box formats exist only in the Qwen3-VL cookbooks. Small open models score low zero-shot in every
  published interface. Qwen-VLA reaches 97.9 on LIBERO with Qwen3.5-4B only through a trained action decoder.
- [Set the pass criteria and the time boxes](issues/05-pass-criteria-and-time-boxes.md): no pass and
  no fail. Time boxes of 2 to 4 hours per block only stop an open-ended search.
- [Choose the generalization battery](issues/16-retention-battery.md): six fixed sets (MMLU-Pro,
  GSM8K, CountBench, ChartQA, RefCOCO, WikiText-2). A drop of 5 points with p below 0.05 is "major".
- [Choose the long-horizon computer use target](issues/17-long-horizon-computer-use-target.md):
  ScreenSpot Pro first, then one portion of OSWorld-Verified or AndroidWorld.
- [Find a way to run and edit the 27B model](issues/21-27b-run-and-edit-paths.md): no library loads
  the NVFP4 copy into PyTorch. The path is a custom loader that turns stored values into real weights
  inside each forward pass (about 20 GB on the GPU). Small edits can vanish in 4-bit rounding.
- [Reproduce the published 0.8B edit](issues/04-reproduce-published-edit.md): the harness applies
  the blog's edit bit for bit. Answers at the token limit drop from 93.8% to 76.6% at a 4,096 limit
  (p = 0.00006). Six harness defects found and fixed.
- [Find the 0.8B repetition fault again as the teacher](issues/06-teacher-refinds-fault.md): in 29
  minutes the teacher found that three components suppress the closing token of the thinking. Two
  edits raise held-out accuracy from 10.2% to about 38%, above the published edit (21.9%). The base
  model with thinking off scores 56.2% with no edit.
- [How the blog's teacher found the direction](issues/07-blog-direction-method.md): it wrote four
  text pairs by hand, took the activation difference after layers 15 and 19, and reversed that
  direction at strength 6 in 64 neurons per layer. Its first prompt named the fault.

## Not yet specified

- **The robotics block after the simulator choice.** A baseline, a target portion, an edit attempt,
  and a LoRA comparison. These become tickets when
  [Choose the simulator and the action interface](issues/14-choose-simulator-and-interface.md) resolves.
- **The long-horizon computer use block.** If
  [Choose the long-horizon computer use target](issues/17-long-horizon-computer-use-target.md) adds a
  multi-step benchmark, that block needs an environment setup, a baseline, and an edit attempt.
- **The 27B block after the path choice.** A target, a baseline, an edit attempt, and the measures.
  These become tickets when [Find a way to run and edit the 27B model](issues/21-27b-run-and-edit-paths.md)
  resolves and the user picks a path.
- **Tool gaps.** `tools.read` returns only the last prompt position of a named module. The teacher
  may need attention patterns, values at other positions, or a logit lens. The calibration block
  shows which.
- **The fallback when no hidden fault exists.** The blog's edit removed a fault that hid an ability.
  If the 4B model has no such fault on the target, the next edit family is not clear. The line
  between a closed-form fit and training also needs a rule at that point.
- **The release form for other users.** The licence, the README, the install path, and how a user
  adds a new target. This becomes sharp after two students have passed through the harness.
- **The form of the final verdict.** A report, a table, or an artifact, and who reads it.

## Out of scope

- **Qwen4.** The release is not out. The harness must be able to load it later, because the student
  is interchangeable.
- **A full-benchmark score.** The user does not want to maximize a full benchmark.

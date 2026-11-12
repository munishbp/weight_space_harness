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
- [Measure the 4B baseline on ScreenSpot Pro](issues/08-baseline-4b-screenspot-pro.md): 59.4% in
  our harness against a published 60.3. Text targets 74.1%, icon targets 35.6%. Thinking off is the
  reference setting.
- [Choose the ScreenSpot Pro slice and the split](issues/09-choose-screenspot-slice.md): icon
  targets, 311 practice items and 293 held-out items, split inside each application.
- [Edit the 4B model for the ScreenSpot Pro slice](issues/10-teacher-edit-4b-slice.md): negative
  after 48 minutes. The model almost succeeds in 60% of failures, but no part suppresses the right
  answer, and a closed-form fit only copies the belief of the model.
- [Run the LoRA baseline on the same slice](issues/11-lora-baseline.md): 15 minutes of LoRA raise
  held-out icons from 38.2% to 50.2% and drop text targets from 76.0% to 63.5%. The battery shows no
  major drop. The change touches 55% of the weights in all 32 layers.
- [Choose the simulator and the action interface](issues/14-choose-simulator-and-interface.md):
  Meta-World from PyPI, with text waypoints and the scene state in the prompt. VLABench fits better
  but its install is blocked.
- [Install one simulator benchmark and run the base model](issues/24-simulator-install.md): the base
  4B model solves 63.0% of 200 Meta-World practice episodes. Two faults are a wrong choice at one
  step: a `close` before a button press, and a first move to the goal in place of the object.
- [Check which multi-step computer use benchmark runs here](issues/22-multi-step-benchmark-setup.md):
  MiniWoB++ first, because one pass takes minutes. A portion of OSWorld-Verified follows. KVM and
  Docker work here with no sudo. Both installs reach outside PyPI.
- [Edit the 4B model for one Meta-World fault](issues/25-teacher-edit-metaworld.md): a rank-1
  least-squares fit on the MLP output of layer 28 raises held-out button-press from 0 to 18 of 20 in
  27 minutes. It changes 0.48% of the weights and no other task. A second fault did not yield.
- [Run the LoRA baseline on the Meta-World target tasks](issues/26-lora-baseline-metaworld.md): LoRA
  on the three target tasks reaches 57 of 60 there and breaks two grasp tasks (60.0% overall). LoRA
  on nine tasks reaches 87.5% with no broken task. Both change about 60% of the weights.
- [Repeat the habit recipe on a second Meta-World fault and stack the edits](issues/29-repeat-recipe-door-open.md):
  the repeat took 14.8 minutes with one failed attempt. The stacked edit reaches 76.0% held-out with
  26 episodes fixed and 0 broken. The protection must hold every prompt position.
- [Install MiniWoB++, run the base model, and choose a portion](issues/27-miniwob-base-and-portion.md):
  the base model solves 59.0% of 83 task types through screenshots. A rank-2 edit raises the five
  email task types from 29.0% to 57.0% held-out, with no loss on the other 78 task types. The first
  working edit came at 19 minutes.
- [Draft the teacher playbook](issues/15-teacher-playbook.md): `PLAYBOOK.md`, from five teacher
  runs. The recipe that worked three times is code in `wsp/habit.py`.
- [Choose the metric set for the writeup](issues/20-writeup-metrics.md): the six README measures in
  every block, plus a random control, a base repeat for noise, stacking, and quantization.
- [Run the harness on the quantized 27B model](issues/28-27b-scaling-run.md): the 27B model loads in
  40 seconds at 19.6 GB and runs at 3 tokens per second. The email recipe of the 4B model raises
  held-out email tasks from 30.0% to 68.0% on it, with 0.22% of the weights changed. The 4-bit
  format keeps the full change, and the reset is exact.
- [Show that the teacher is interchangeable](issues/18-teacher-interchange.md): a second teacher
  (Opus 5.5, the playbook and the code only) reached the button edit in 30 minutes. Held-out: 63.0%
  to 71.5%, 17 fixed and 0 broken. It reported four gaps, and all four are closed. A teacher from
  another vendor needs an account from the user.
- [Choose a student outside the Qwen family](issues/19-student-outside-qwen.md): GLM-4.6V-Flash
  (10.3B). Every tool ran with no change. The habit recipe raises held-out Meta-World from 51.5% to
  57.0% (pick-place 12 to 20 of 20) in 30 minutes. The layout now handles a norm between the MLP
  and the residual stream.
- [Test the closing-token recipe on the 4B model](issues/23-thinking-fault-4b.md): one removal on
  the MLP of layer 30 raises held-out GSM8K with thinking from 77.3% to 89.1% and cuts the replies
  at the token limit from 30.5% to 8.6%. The working edit came at 8 minutes. Thinking off scores 93.0%.
- [Run one portion of OSWorld on the 4B model and try the recipe on it](issues/30-osworld-portion.md):
  negative. The base student solves 4 of 22 sampled tasks. Three forced replays (stop at the loop, a
  hint, no `type` after `type`) solve 0 failed tasks, so no edit candidate exists.
- [Run the long-horizon tasks of VLABench on the 4B model and try the recipe on them](issues/31-vlabench-procedure-edit.md):
  partial. One stacked fit of 46 targets gives all skill names correct in 91 of 150 held-out plans
  (base 0). No plan is fully correct, the edit needs the plan-only mode, and MMLU-Pro loses 2.5 points.
- [Scan more of OSWorld for a fault that tasks share, and edit it](issues/32-osworld-fail-edit.md):
  positive for one portion. 26 OSWorld tasks are not possible by design and need the action `fail`.
  One rank-1 edit takes them from 0 to 10 of 11 held-out tasks. Possible tasks go from 4 to 3 of 38,
  and 7 of them end with a false `fail`. The battery stays within 1 point.
- [Test if the knowledge of the teacher, as a guide in the prompt, lifts possible OSWorld tasks](issues/33-osworld-guide-in-prompt.md):
  negative. Two guides for VS Code leave 9 possible practice tasks at 2 and 3 solved (base 3). The
  student follows an exact key recipe and fails at each step that needs a look at the screen.

## Not yet specified

All tickets of the map are resolved. The destination has its six items: the general harness, the
0.8B calibration, the computer use verdict, the robotics verdict, the 27B run, and the playbook.
`RESULTS.md` holds the verdict, with the answer to the question at its top. What stays open needs a
decision of the user:

- **The removal of the benchmark installs.** The user asked to keep the OSWorld and VLABench
  downloads inside the project directory, git-ignored, and to remove them later. The list is in
  `README.md` under "Temporary installs".
- **The rest of OSWorld.** The base scan covers Thunderbird, VS Code, Chrome (83 tasks), and all 26
  tasks that are not possible. The seven other groups (259 tasks) take about 4 hours on three
  virtual machines. A scan there shows if the false `fail` costs solved tasks in other applications.
- **VLABench in its simulator.** The run used the planning track. The simulator track needs the
  repository install and its asset archives.
- **A teacher from another vendor.** It needs an account from the user.
- **The release form for other users.** The licence of the repository and how a user adds a new
  target. The repository is public and has no licence file.
- **The user name in the git history.** Some run outputs in earlier commits hold the home path. The
  newest commit does not. A history rewrite needs the request of the user.
- **The next edit family.** No closed-form edit fixed a wrong choice between candidates from the
  input. That kind is the largest share of the failures on ScreenSpot Pro and in the push task, and
  it is the reason that no VLABench plan became fully correct.

## Out of scope

- **Qwen4.** The release is not out. The harness must be able to load it later, because the student
  is interchangeable.
- **A full-benchmark score.** The user does not want to maximize a full benchmark.

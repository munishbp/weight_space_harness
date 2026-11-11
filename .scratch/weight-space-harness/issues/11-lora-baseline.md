# Run the LoRA baseline on the same slice

Type: task
Status: resolved
Blocked by: 09, 10

## Question

What held-out score and what retention does a LoRA fine-tune reach on the same slice, with the same
practice items and the same wall-clock time as the teacher edit?

This ticket waits for the teacher edit, because the wall-clock budget comes from that run. The
ticket reports the same generalization battery and deviation measures for the LoRA model, and it
states the LoRA settings (rank, target modules, learning rate, steps) and why.

## Answer

Resolved on 2026-11-10. Run directory: `runs/ss-pro-4b-lora15`. Code: `wsp/lora_baseline.py`.

**A LoRA fine-tune of 15 minutes raises the held-out icon score from 38.2% to 50.2%. The teacher found
no edit in 48 minutes. The LoRA model loses 12.5 points on text targets, which the training did not
show to it.**

### Settings and why

- Budget: 15 minutes of training, the ideal time of the goal. The teacher search took 48 minutes, so
  the LoRA model had less time than the teacher.
- Data: the 311 practice icons, the same items that the teacher used. The reply to learn is a click
  on the centre of the target box, in the model's own tool-call format.
- Rank 16, alpha 32, no dropout, learning rate 1e-4, AdamW, 4 examples per optimizer step. These are
  common defaults. No setting was tuned, because a search over settings costs time that the budget
  does not hold.
- Adapted: every linear matrix of the attention blocks and the MLPs of the language model (176
  matrices, 27.9 million trainable weights). The vision tower keeps its weights.
- The run saw 384 examples (1.23 passes over the practice set). The peak GPU memory was 14.0 GB.
- Not counted in the 15 minutes: the model load, the write of the merged weights (6.6 GB), and the
  evaluation.

### The six measures

| Measure | Teacher edit | LoRA, 15 minutes |
|---|---|---|
| Held-out icons (293) | no edit to test; base 38.2% | **50.2%** (interval 44.5 to 55.9). 47 fixed, 12 broken, p = 0.000005 |
| Practice icons (311, seen in training) | base 33.1% | 70.7%. 121 fixed, 4 broken |
| Text targets (200, not in training) | base 76.0% | **63.5%**. 6 fixed, 31 broken, p = 0.00004 |
| Time | 48 minutes of search | 15 minutes of training |
| Weights changed | 0 | 55.3% of all weights, 176 of 723 tensors, all 32 layers |
| Relative change of the model | 0 | 0.0034 (per matrix 0.004 to 0.012) |
| Deviation on WikiText-2 | 0 | KL 0.0056, top-token agreement 96.3%, perplexity 13.38 to 13.28 |

Battery, base to LoRA (a major drop is 5 points or more with p below 0.05):

| Set | Base | LoRA | Fixed | Broken | p |
|---|---|---|---|---|---|
| MMLU-Pro (200) | 48.0 | 47.0 | 5 | 7 | 0.77 |
| CountBench (200) | 85.0 | 82.5 | 2 | 7 | 0.18 |
| ChartQA (200) | 85.0 | 84.5 | 0 | 1 | 1.0 |
| RefCOCO (200) | 90.0 | 89.0 | 1 | 3 | 0.63 |
| GSM8K with thinking (100) | 75.0 | 78.0 | 10 | 7 | 0.63 |

No battery set has a major drop. Four of the five sets move down by 0.5 to 2.5 points, inside the noise.

### What this means

- The icon ability is learnable from 311 examples. The teacher could not reach it with a closed-form
  edit, because no part of the model hides it. Training adds an ability. The edit method does not.
- The LoRA gain has a price that the battery does not show. The battery holds general sets. The
  loss sits in the nearest neighbour of the target: text targets of the same benchmark. A verdict
  needs a near-transfer set as well as a general battery.
- The LoRA change is diffuse. Each layer holds 2.6% to 3.6% of the squared change. The 0.8B teacher
  edits changed 0.02% to 1% of the weights in two or three layers.
- The LoRA model generalizes inside the target: 50.2% on held-out icons against 70.7% on the icons
  that it saw.

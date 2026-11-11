# Edit the 4B model for the ScreenSpot Pro slice

Type: task
Status: resolved
Blocked by: 05, 06, 09, 16

## Question

Does a closed-form edit by the teacher raise the held-out score of Qwen3.5-4B on the chosen slice,
and what does it cost in wall-clock time and in retention?

The teacher first sorts the practice failures by cause. The question behind that step: does a fault
hide an ability that the model has? The teacher then edits within the time box from
[Set the pass criteria and the time boxes](05-pass-criteria-and-time-boxes.md).

Record the six measures that the map requires of every verdict: the held-out gain, the ledger totals
(time and forward passes), the generalization battery before and after, the edit size from the
manifest, the weight-space and output-space deviation, and a time-versus-gain table for the LoRA
comparison. A failure to improve is a valid answer.

## Answer

Resolved on 2026-11-10. Run directory: `runs/ss-pro-4b-teacher`. The teacher was Claude (Fable 5.2).

**No. The teacher found no closed-form edit that raises the icon score. This block is a negative
result after 48 minutes of teacher time.** No edit went to the held-out items, because no trial beat
the base model on the unseen practice items.

### Cost

48 minutes of teacher time, 11 code runs, 195 generation batches, 428 read passes, 4 generation
trials. One trial on 156 items takes 3.7 minutes.

### What the teacher tried, in the order of the playbook

All scores are on the 156 unseen practice items. The base score is 35.9%.

| Step | Result |
|---|---|
| Prompt hint "the target is an icon or a small symbol, not a text label" | 34.6%. 6 fixed, 8 broken. No change. |
| Thinking on | 27.6%. 2 fixed, 15 broken (p = 0.002). Worse. |
| Where does the reply leave the target range? | 51 of 99 failures leave at the first digit of x. In 60 of 99, a digit that stays in range has a probability of 10% or more at that step. In 1 it is below 1%. The model almost succeeds. |
| Direct logit attribution at the first digit (split error 0.3%) | No part pushes against the correct digit selectively. The MLPs of layers 27 to 30 and the attention of layer 23 write the digit that the model chooses, right or wrong (layer 29: +0.88 in successes, -0.72 in failures). The attention of layer 31 pushes against the chosen digit in both cases. |
| A direction from "about to click an icon" minus "about to click text" | It separates the two states (held-out AUC up to 0.86). The missed icons already sit on the icon side (0.87 to 1.04 on a scale from text to icon). A push along it has nothing to move. No trial run. |
| Closed-form least-squares fit on the MLP of layer 29 (1,166 keys from 155 items, protected by neutral text) | 35.9% at both strengths. 2 fixed, 2 broken. |
| Prior shift on the first digit, from the click and target distributions | Simulated on recorded probabilities: 2 to 9 fixed, 4 to 21 broken. No gain. |

### Why the fit fails

The fitted correction is not noise. On unseen items it raises the correct digit most in 41% of the
digit steps (chance: 10%). Taken alone, it picks an in-range first digit on 92 of 156 items, and the
base model on 105. On the 51 items where the base model is wrong, the correction is right 7 times.
The correction is a copy of the belief that the model already has. At the late layers, no
information about the right element exists beyond the choice of the model.

### What this means

The 0.8B fault was a suppressed token: the model had the answer and a few parts blocked it. The icon
failures of the 4B model are a wrong choice between elements, made by the layers that read the image.
The blog states the limit: an edit "could remove an obstacle that hid an ability, but it could not
add an ability the model lacked". This block agrees with that limit.

Three images of failures show the kind of error: for "Hide the icing layer" the model clicks the row
named "icing" and not the eye icon; for "click language template" it clicks the text entry "Language
Templates" and not the lightbulb icon.

### Not tried

- An edit of the layers that read the image (the attention layers 19 to 27 or the vision tower). The
  teacher has no closed-form target for them.
- A higher image resolution or a zoom step. Those change the pipeline and not the weights.
- A hand-written contrast pair, as in the blog. A pair needs a behavior to contrast, and the failures
  here show no single behavior.

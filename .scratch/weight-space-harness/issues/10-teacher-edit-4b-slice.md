# Edit the 4B model for the ScreenSpot Pro slice

Type: task
Status: open
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

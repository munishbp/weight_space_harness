# Test the closing-token recipe on the 4B model

Type: task
Status: resolved

## Question

The base Qwen3.5-4B reaches the 4,096 token limit in 33% of GSM8K answers with thinking on (battery
run of 2026-11-10: 75.0% correct). This is the fault that the teacher edited in the 0.8B model. Does
the same recipe (find the parts that suppress the closing token of the thinking, then remove that
suppression) work on the 4B model, and how long does it take?

This ticket answers the scaling question on a task where a fault exists. It is not a computer use
task, so it does not replace the computer use verdict.

The task: follow the playbook on GSM8K practice items with thinking on. Measure the thinking-off
prompt first, because it beat every edit on the 0.8B model. Report the six measures of the map.

## Answer

Run directory: `runs/gsm8k-4b-teacher` (scripts `setup.py`, `step1.py` to `step5.py`, `state.py`,
`verdict.py`; numbers in `verdict.json`). The edit is `edit_close_L30_s2.safetensors`.

Yes. The recipe of the 0.8B model works on the 4B model, with one matrix, in 8 minutes.

### Time

| Minute | Step |
|---|---|
| 0.0 | Start. 64 practice items, thinking on, a limit of 2,048 tokens. |
| 1.8 | Base, thinking on: 48.4% correct, 62.5% at the limit. |
| 3.0 | Base, thinking off: 92.2% correct. |
| 3.9 | Kernel restart for a defect in `tools.read_tokens` (fixed). The attribution follows. |
| 6.5 | Trial, strength 1: 51.6%. No effect. |
| 8.3 | Trial, strength 2: 82.8% correct, 17.2% at the limit, 23 items fixed and 1 broken. |
| 12.2 | Strength 3 and 4: about 80% correct, but the median thinking is 22 tokens. The thinking collapses. |
| 14.0 | Random control at strength 2: 46.9%. |

14 generation passes and 4 read passes.

### The fault

- 37 of 64 replies never close their thinking. They do not repeat text (no 20-token window occurs
  two times). The model plans again and questions its own answer.
- In 27 of the 37, the closing token never has a probability above 0.0003. In 10 it does, always
  after a line break, and the model writes an indent in place of it.
- At those positions the MLP of layer 30 pushes against the closing token (share -5.71, against
  -1.10 at other positions). No other part comes near it. The split error is 0.0055.

### The edit

`edits.remove_direction` on `layers.30.mlp.down_proj.weight`, direction "closing token minus the
indent token", strength 2. A strength of 1 sets the push to zero and is not enough. A strength of 2
reverses it. One tensor, 0.44% of the weights, relative change of the model 0.0013.

### Held-out verdict: 128 items, a limit of 4,096 tokens

| Measure | Base, thinking on | Edit | Base, thinking off |
|---|---|---|---|
| Correct | 77.3% | 89.1% | 93.0% |
| At the token limit | 30.5% | 8.6% | 0.8% |
| Thinking closed | 75.8% | 92.2% | no thinking |
| Median thinking length (tokens) | 1,229 | 991 | |
| Mean tokens per reply | 2,503 | 1,601 | 446 |

- Edit against base: 17 items fixed, 2 broken, p = 0.0007.
- Deviation on neutral text: mean KL 0.0006, top-token agreement 98.8%, perplexity 13.377 to 13.404.
- Battery: MMLU-Pro 48.0 to 48.0, CountBench 85.0 to 85.0, ChartQA 85.0 to 84.5, RefCOCO 90.0 to
  90.5. The GSM8K set of the battery (thinking on) goes from 75.0 to 90.0 (16 fixed, 1 broken).

### What it means

- This is the fault of the blog on a model five times larger. One removal on one matrix fixes most
  of it in 8 minutes, inside the 10 to 15 minute aim.
- The thinking-off prompt still scores higher (93.0% against 89.1%), as on the 0.8B model. The edit
  is useful for a user who wants the thinking and wants it to end.
- The strength needs a second measure. Strength 3 has the same accuracy and no thinking.

### Harness defect found

`tools.read_tokens` failed on more than 8 sequences. The change that runs 8 rows per pass (for the
27B model) left the hook with the row index of one pass. The hook now counts rows across passes.

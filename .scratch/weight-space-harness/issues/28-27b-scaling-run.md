# Run the harness on the quantized 27B model

Type: task
Status: resolved

## Question

Does the harness work on the Qwen3.8-27B NVFP4 checkpoint on one 32 GB GPU, and what does one step of
the teacher cost there?

The user chose the path on 2026-11-10: "If they're quanted, just use them". The loader of
[Find a way to run and edit the 27B model](21-27b-run-and-edit-paths.md) keeps each weight in its
4-bit or 8-bit format and decodes it inside the forward pass.

The task, in order:

1. Load the model in the kernel. Record the load time, the GPU memory, and the tokens per second.
   A coherent reply shows that the decoder reads the stored format correctly.
2. Run one benchmark that an earlier block has on the 4B model, so that the two sizes compare.
   Meta-World is the choice: one pass is short, and the 4B edit recipe exists.
3. If the base model shows a fault at one step, repeat the recipe of
   [Edit the 4B model for one Meta-World fault](25-teacher-edit-metaworld.md). Record the time.
4. Record how much of the requested edit survives the 4-bit format (`kept_share_of_request`), and
   whether `reset_edit` restores the stored bytes exactly.

A negative result at step 3 (no fault of that kind in the larger model) is a valid result. Steps 1,
2, and 4 answer the scaling question alone.

## Answer

Run directory: `runs/scale-27b` (scripts `check.py`, `setup.py`, `base.py`, `mwob_setup.py`,
`mwob_base.py`, `mwob_step2.py`, `mwob_step3.py`, `mwob_verdict.py`, `mwob_control.py`; numbers in
`mwob_verdict.json`). The edit is `edit_email_27b_L60.safetensors` (not in the repository, 357 MB).

Yes. The harness loads, reads, edits, and resets the quantized 27B model on one 32 GB GPU. The habit
recipe of the 4B model works on it with a new layer number.

### Step 1: load and speed

- Load time: about 40 seconds. GPU memory after the load: 19.6 GB. With a batch of 14 Meta-World
  prompts or 8 browser prompts the peak is 28 to 31 GB.
- Speed: 2.9 tokens per second for one sequence. Each forward pass decodes every 4-bit matrix.
- The replies are coherent ("Paris", "391" for 17 times 23).
- Three defects of the harness showed only at this size, and all three are fixed: the decode of the
  output head needed 10 GB in one piece, `tools._model_squared_norm` made a float copy of each
  matrix, and the protection matrix (17,408 by 17,408 in float64, 2.4 GB) had two copies.

### Step 2: Meta-World

Base model, 200 practice episodes, 19.3 minutes: 77.5% (the 4B model: 63.0%). Push 6, door-open 14,
drawer-open 0, button-press 17, peg-insert 18, the other five tasks 20 of 20. The 27B model does not
have the two habit faults of the 4B model (button-press 17 against 0, door-open 14 against 0). Its
failures in push and drawer-open are choices of positions, the kind that no closed-form edit fixed
on the 4B model. No edit was tried here.

### Step 3: the email tasks of MiniWoB++

The 27B model has the same fault as the 4B model. Base, 100 practice episodes: 36 solved, reply and
forward 0 of 40. At the three fault steps on unseen seeds the correct action name `click` has a
probability of 0.31, 0.15, and 0.38.

The fit: `habit.fit` on the MLP output matrix of layer 60 of 64 (`mlp.down_proj`, 5,120 by 17,408,
stored as NVFP4), three target steps, the typing step and 403 action steps of other task types as
zero-value keys, protection over all positions of 487 prompts. One fit takes 32 seconds.

| Minute | Step |
|---|---|
| 0.0 | Start of the base practice pass. |
| 20.1 | Base practice pass done: 36 of 100. |
| 23.1 | Contexts at the fault steps and the base probabilities. |
| 45.8 | Protection matrix done (one failed run for memory before it). |
| 52.6 | Keys done. |
| 56.1 | Fit sweep done: margin 8 gives `click` 0.87, 0.69, 0.92; margin 16 gives 1.0, 1.0, 0.98. |

The search took 56 minutes. 20 of them are one base pass, and about 11 are a rerun after a memory
failure. The teacher did not search for the fault again: the page states at the fault steps came
from the forced replay of the 4B run. The 27B number is the cost of a transfer, not of a discovery.

Held-out verdict, margin 12 (the middle of the sweep, chosen before the held-out pass):

| Held-out episodes | Base | Edit |
|---|---|---|
| email-inbox-reply (20) | 0 | 20 |
| email-inbox-forward (20) | 0 | 1 |
| email-inbox (20) | 6 | 9 |
| email-inbox-noscroll (20) | 12 | 19 |
| email-inbox-star-reply (20) | 12 | 19 |
| All five email task types | 30.0% | 68.0% |
| The other 78 task types (1 seed each) | 73.1% | 73.1% |

- Email: 39 episodes fixed, 1 broken, p = 7e-11. Other task types: 1 fixed, 1 broken.
- Forward stays at 1 of 20, as on the 4B model (3 of 20). The action names are correct after the
  edit, and the click on the field goes to a wrong place in the forward form.
- Edit size: 1 tensor of 1,184, 1 layer, 0.22% of the weights. Relative change of the matrix 0.32,
  of the model 0.0153.
- Deviation on neutral text (13,730 positions): mean KL 0.0007, top-token agreement 98.9%,
  perplexity 8.714 to 8.711.
- Battery, 100 items per set: MMLU-Pro 60 to 61, CountBench 94 to 94, ChartQA 75 to 75, RefCOCO 94
  to 94. GSM8K with thinking did not run: at 3 tokens per second it needs hours.
- Passes in the run directory: 760 generation passes, 1,252 read passes.
- The verdict took 89 minutes: two email passes of 19 minutes, two passes of the other task types of
  12 minutes, and the battery two times.

- Random control: the same fit with random directions of the same length (relative change of the
  matrix 0.33 against 0.32). Email held-out 30 of 100, and no episode changes.

### Step 4: the 4-bit format

- `kept_share_of_request` is 1.03: the stored change has the full size of the requested change. The
  rounding to the 4-bit grid adds noise to it and removes none of it.
- The requested change has rank 2 (two token directions). The stored change has rank 7, and it differs from the original in
  67.9% of the weights of the matrix, because a block of 16 weights shares one scale and a changed
  scale moves the whole block.
- `tools.reset_edit` restores the stored bytes of the matrix exactly.
- The format has a largest value per tensor (1.172 for this matrix). A new weight above it is an
  error, and the model stays unchanged. The fit with the token directions stays in the range. The
  random control passed it in 1 weight.

### What this does not show

- No LoRA comparison ran. Training needs gradients through the packed weights and more memory than
  the GPU has next to a 27B model.
- The user accepted that the quantized copy is not a direct scale of the 4B model. The 27B model is
  a newer generation (Qwen3.8), so the two sizes differ in more than size.
- One benchmark portion, one fault. The Meta-World faults of the 4B model do not exist in the 27B
  model, so the larger model leaves fewer faults of this kind to fix.

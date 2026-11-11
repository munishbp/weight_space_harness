# Choose a student outside the Qwen family

Type: grilling
Status: resolved
Blocked by: 06

## Question

Both planned students are Qwen3.5 models. Does the map need a run on a student from another family
(for example GLM) to show that the harness is not specific to Qwen, and which model?

Decide with the user after the calibration, when it is clear which parts of the tools know the
Qwen3.5 architecture. A fact for the decision: three of every four Qwen3.5 layers are Gated DeltaNet
layers and not standard attention, so module names and edit recipes differ between families.

## Decision

Decided by the teacher on 2026-11-11 under the delegation of the user ("I think you can do the rest
without me"). The user can overrule it.

- **A run on a student from another family is needed.** The user named the aim: "bigger qwen, smaller
  qwen, GLM". Three Qwen students do not show that the tools are free of Qwen names.
- **The student is `zai-org/GLM-4.6V-Flash`** (revision `411bb4d7`). Reasons: the user named GLM. The
  model takes images, so the same two benchmarks run on it. It has 10.3 billion weights (20.6 GB in
  bf16) and fits on the 32 GB GPU with no quantization. The licence is MIT. `transformers` 5.22 has
  the class (`Glm46VForConditionalGeneration`), so the load runs no code from the repository.
- **The test has two levels.** Level 1 checks the tools: load, `layout.find`, generate with an
  image, read, one edit, an exact reset, and the deviation. Level 2 runs one benchmark portion that
  the 4B model has, and repeats the habit recipe if the base model shows a fault of that kind.
  A base model with no such fault is a valid result for level 2.

## Answer

Run directory: `runs/glm-46v` (scripts `check.py`, `setup.py`, `base.py`, `step1.py` to `step6.py`,
`verdict.py`, `battery2.py`; numbers in `verdict.json`). The edit is `edit_pick_L34_m12.safetensors`.

The harness works on a student outside the Qwen family. The habit recipe gave a held-out gain on
GLM-4.6V-Flash 30 minutes after the first pass. Two parts of the harness assumed Qwen, and both are
now general.

### Level 1: the tools

Every tool ran with no code change: the load (19.2 GB), `layout.find` (40 layers of width 4096),
generation with text, with thinking, and with an image, `tools.read` with inputs, the token
direction, one edit with its manifest, the deviation, and a reset that restores the weight exactly.

### Level 2: Meta-World

Base model: 56.5% of 200 practice episodes (Qwen3.5-4B: 63.0%).

| Minute | Step |
|---|---|
| 0.0 | Start of the base practice pass. |
| 5.6 | Base practice pass done. Button-press 1 of 20, with the same `close` habit as the 4B model. |
| 7.2 | Forced replay on button-press: 1 of 20. The habit is not the fault there. The model moves to a point beside the button. |
| 13.6 | Forced replay on all tasks: pick-place goes from 11 to 20 of 20 with `move` forced after the first move. |
| 14.4 | Attribution has an error of 1.06. The split is not valid. |
| 20.0 | Cause found and fixed: a norm between each MLP and the residual stream. Error 0.006. MLP 34 pushes hardest against `move`. |
| 23.9 | Fit and margin sweep. On unseen seeds `move` goes from 0.31 to 0.60, 0.82, and 0.98 (margin 4, 8, 16). |
| 30.2 | Practice trial at margin 12: 56.5% to 60.5%, 9 episodes fixed and 1 broken. |

The fault: in pick-place the model moves above the object and then writes `close` before it
descends. At that step `close` has 0.78 and `move` has 0.22.

The edit: `habit.fit` on the MLP output matrix of layer 34 of 40, `move` over `close`, margin 12,
rank 1. Keep keys: step 1 of the nine other tasks, and steps 2 and 3 of all tasks. Protection: all
positions of 171 prompts.

| Held-out episodes (20 per task) | Base | Edit |
|---|---|---|
| pick-place (target) | 12 | 20 |
| peg-insert-side | 5 | 8 |
| The other eight tasks | 86 | 86 |
| All ten tasks | 51.5% | 57.0% |

- 11 episodes fixed and 0 broken, p = 0.001. Peg-insert has the same step (a `close` before the
  descent) in some episodes, so it gains too.
- Random control (same fit, random direction): pick-place 9 of 20.
- Edit size: 1 tensor of 704, 1 layer, 0.50% of the weights. Relative change of the matrix 0.082,
  of the model 0.0042.
- Deviation on neutral text: mean KL 0.0004, top-token agreement 99.0%, perplexity 10.741 to 10.743.
- Battery: MMLU-Pro 45.5 to 45.5, CountBench 71.0 to 71.0, GSM8K with thinking 90.0 to 89.0
  (2 fixed, 3 broken), ChartQA 86.0 to 85.5, RefCOCO 46.5 to 47.5. The RefCOCO score is low because
  GLM writes some boxes in a third format that the scorer does not read.

### What assumed Qwen, and the fix

1. **The MLP output does not join the residual stream directly.** A GLM-4 layer has an RMS norm
   after the MLP and after the attention (`post_mlp_layernorm`, `post_self_attn_layernorm`). The
   attribution read the MLP output before that norm, and the fit wrote a value before that norm as
   if it were a change of the residual stream. Fix: `layout.find` reports `mlp_write_norms` and
   `attention_write_norms`, the attribution reads the norm output, and `habit.fit` converts the
   values with `edits.values_before_norm`. The norm fixes the size of its output, so the value
   reaches the wanted output in direction only. The margin sweep shows that this is enough.
2. **Two battery scorers read only the Qwen answer format.** GLM writes "The answer is" and puts
   the answer between box tokens, and it writes a box as a bare list. ChartQA and RefCOCO scored 0.
   Fix: the two scorers read both formats.

### What this does not show

- A second benchmark on GLM. MiniWoB++ did not run on it.
- A LoRA comparison on GLM.
- A family with a different MLP form (for example a mixture of experts). `layout.find` raises an
  error there and names the list to extend.

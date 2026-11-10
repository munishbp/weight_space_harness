# Reproduce the published 0.8B edit

Type: task
Status: resolved

## Question

Does our harness reproduce the effect of the edit that the blog published for Qwen3.5-0.8B?

The task: load `Qwen/Qwen3.5-0.8B` at revision `2fc06364715b967f1860aea9cf38778875588b17` in the
kernel. Apply the `p50c` edit from the logs repository through `tools.save_edit`. Measure the share
of practice math answers that reach the token limit, for the base model and for the edited model.

The published reference is 93.0% for the base model and 33.1% for the 50-pass edit, on MATH-500 with a
32,768 token limit. A smaller sample and a lower limit are acceptable if the ticket states them. The
pass condition is a clear drop in the same direction.

This ticket is the first run of `wsp/kernel.py`, `wsp/client.py`, and `wsp/tools.py`. Record each
defect that the run exposes.

## Answer

Resolved on 2026-11-10. Run directory: `runs/calib-08b-reproduce`.

**The harness reproduces the published edit exactly.** The base tensors match the blog's base hashes.
After `tools.save_edit`, both edited tensors match the hashes of the blog's evaluated checkpoint, bit
for bit. `tools.reset_edit` restores the original bits. `tools.load_edit` restores the edit from the
run directory after a kernel restart.

**Our edit metrics agree with the blog's manifest.**

| Tensor | Changed weights | Columns | Rows | Relative change | Rank of the change |
|---|---|---|---|---|---|
| `layers.15.mlp.down_proj.weight` | 65,260 (1.78%) | 64 of 3584 | 1024 of 1024 | 0.106 | 1 |
| `layers.19.mlp.down_proj.weight` | 64,928 (1.77%) | 64 of 3584 | 1023 of 1024 | 0.081 | 1 |

The whole edit changes 130,188 weights: 0.0153% of the model, 2 of 473 tensors, layers 15 and 19.
The edited model is 0.24% away from the original in weight norm.

**The repetition fault drops.** 128 GSM8K practice problems, the blog's sampling, a 4,096 token limit,
the same seed for both models. File: `runs/calib-08b-reproduce/math_gsm8k_128.json`.

| Model | Reached the token limit | Correct | Mean new tokens |
|---|---|---|---|
| Base | 93.8% | 10.2% (95% interval 6.0 to 16.6) | 3987 |
| Published edit | 76.6% | 21.9% (95% interval 15.6 to 29.8) | 3718 |

Paired on the same items: 26 answers finish that did not finish before and 4 stop finishing
(p = 0.00006). 20 answers become correct and 5 become wrong (p = 0.004).

**Limit of the comparison.** The blog used a 32,768 token limit and reports 93.0% and 33.1%. Our base
figure matches. Our edited figure is higher because the limit is 8 times lower and the edited model
often needs more than 4,096 tokens. A full-limit run was not done: one pass costs much more time and
memory, and the bit-exact check already proves that the model is the published one.

**Defects that the first run exposed, all fixed.**

1. A restart command matched its own shell and killed it. The kernel now writes a pid file.
2. `run.json` recorded no revision and a wrong commit value.
3. A batch of 128 answers at 8,192 tokens ran out of GPU memory. The kernel survived, but it did not
   free the memory, because the cleanup ran while the traceback still held the tensors.
4. `tools.encode` used a deprecated argument style for padding.
5. The Gated DeltaNet layers ran on a slow fallback. The `flash-linear-attention` package doubles the
   generation speed. `causal_conv1d` is still on the fallback.
6. `str.splitlines` broke a JSONL file that holds Unicode separators inside strings.

**Open point for the harness.** A change to `wsp/tools.py` needs a kernel restart, which reloads the
model. This is cheap for the 0.8B model and will cost more for larger students.

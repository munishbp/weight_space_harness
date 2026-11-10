# Reproduce the published 0.8B edit

Type: task
Status: claimed

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

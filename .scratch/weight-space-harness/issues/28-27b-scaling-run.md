# Run the harness on the quantized 27B model

Type: task
Status: open

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

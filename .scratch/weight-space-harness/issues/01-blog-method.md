# What the blog method does and what it released

Type: research
Status: resolved

## Question

What is "weight-space programming", exactly, and what code, data, and results did the authors release?

## Answer

Researched on 2026-11-10 from the blog, the alphaXiv mirror, and the logs repository.

**Source.** "Sparks of Weight-Space Programming: LLMs can natively read & write neural network
weights", Takehi, Teoh, Zhang, Khattab (MIT CSAIL), 2026-11-02. It is a blog post. No paper and no
arXiv ID exist. The alphaXiv page is a mirror.

- Blog: https://rikiyat.github.io/blog/weight-space-programming/
- Logs: https://github.com/RikiyaT/lm_understand-logs
- Edited checkpoint: https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming

**Method.** No fixed algorithm exists. A frontier model (`gpt-6-astra`, reasoning effort `max`)
reasons about the weights and writes edits. In Experiment 8 (Qwen3.5-0.8B, text part, thinking on):

- The teacher has a persistent Python kernel over the weights: CPU only, no internet, at most 300
  code runs, and a `save_edit(name, tensor)` call.
- The teacher has a budget of 10, 25, or 50 forward passes. One pass is a generation on up to 512
  practice problems, or a teacher-forced read of up to 1,024 texts (log-probs, hidden states, MLP
  activations, attention).
- Forbidden: "no backward passes, autograd, optimizers or iterative fitting of weights to a loss".
- `save_edit` refuses the final norm, the embedding and output layer, and a constant scale of a tensor.
- The harness starts a new conversation every 40 turns with a handoff message.
- Practice data: GSM8K train, MATH train, AIME. None comes from a test set.

**Final edits.**

- 10 passes: remove a repetition direction from 30 output matrices in layers 8 to 22,
  `W' = W - 1.5 u(u^T W)`.
- 25 passes: zero the 256 output columns of attention head 7 in layer 15.
- 50 passes: subtract the repetition direction from 64 output columns per layer of the MLPs of layers
  15 and 19. This changes 131,072 weights, 0.016% of the model.

**Results (base / 10 / 25 / 50 passes).**

| Measure | Base | 10 | 25 | 50 |
|---|---|---|---|---|
| MATH-500 | 7.9 | 12.3 | 14.1 | 48.1 |
| GSM8K | 13.8 | 25.7 | 25.1 | 67.1 |
| IFEval | 21.0 | 49.0 | 39.9 | 62.3 |
| MATH-500 answers at the 32,768 token limit (%) | 93.0 | 87.2 | 86.4 | 33.1 |

Evaluation: vLLM, temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5.

**Cost.** 11.6 h, 8.5 h, and 20.3 h of wall-clock time for the three runs. One generation pass took
up to about 30 minutes on a rented GPU.

**Limits that the authors state.**

- On Gemma 3 270M, which rarely repeats, the teacher "could not improve it". The edit "could remove
  an obstacle that hid an ability, but it could not add an ability the model lacked".
- The largest model is 0.8B. No vision-language model was tested. No comparison with LoRA exists.

**Released.** The prompts, the logs, the edit files (`exp8_math/runs/<run>/edit/edit.npz` and
`manifest.json`), and `exp8_math/code/apply_edit.py`. The apply script targets `Qwen/Qwen3.5-0.8B`
at revision `2fc06364715b967f1860aea9cf38778875588b17`, with tensor names `model.language_model.*`.

**Not released.** The agent harness (the sandbox and the forward-pass server).

**Not verified.** An alphaXiv commenter says that the first prompt names the repetition fault. See
[How the blog's teacher found the direction](07-blog-direction-method.md).

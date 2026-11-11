# How the blog's teacher computed the loop direction (run p50c)

Source: the public logs at https://github.com/RikiyaT/lm_understand-logs, folder `exp8_math/runs/p50c/`.
All paths below are relative to that folder.
A turn number `tNNN` is the turn folder `turns/tNNN/`.
The code of a turn is in `turns/tNNN/answer.md`.
The output of that code is in `turns/tNNN+1/prompt.txt`.

Short answer:

- The teacher wrote four pairs of short thinking texts by hand.
  Each pair has the same problem and the same opening steps.
  Then one text continues with a productive check, and the other repeats one doubt sentence word for word.
- The loop direction is the mean of (repetitive minus productive) residual-stream vectors over the last 36 tokens of each text.
  The teacher took it after layer 15 (hidden state 16) and after layer 19 (hidden state 20).
- Yes, the first prompt names the fault. It says: "if the model often fails because it repeats itself and never finishes, finding and fixing what drives that repetition counts."

## The first prompt (`turns/t001/prompt.txt`)

These sentences name or hint at the fault. The quotes are exact.

- "Repairing a specific failure mechanism is in scope: for example, if the model often fails because it repeats itself and never finishes, finding and fixing what drives that repetition counts."
- "Global knobs (making the whole output distribution sharper or flatter) are not what we are looking for, and neither is making the model skip its thinking or close it early regardless of the problem; repairing a specific failure inside the thinking (such as repetition) is in scope."
- "Look at how the model fails on practice problems: a wrong final answer, finishing without an answer, or being cut off at the token limit (often while repeating itself in its thinking)."
- "Go after the largest category you can explain."
- "A lower share of cut-off answers does not by itself mean more correct answers: judge edits by accuracy."
- "one teacher-forced pass can read the reference solutions of up to 1,024 practice problems (or the model's own good versus failing continuations) and gives a signal on every token."
- "Edits that change a tensor by a fraction of a percent usually have effects too small to see; test strengths large enough to matter and back off if they hurt."

Tools that the prompt gives:

- A persistent Python kernel with no internet and no GPU. The kernel has `load`, `load_original`, `save_edit`, `drop_edit`, `edited` and `tok`. Gradients are off. The limit is 300 code runs.
- The practice files: GSM8K train (7,469), MATH train (7,275, with level 1-5) and AIME (714, answers only).
- A forward-pass tool with a budget of 50 passes. A pass is one of two types:
  - Generation: up to 512 generations with the evaluation protocol (thinking on, temperature 0.6, `max_tokens` up to 32,768). The mode `"both"` gives a paired comparison with the same seeds. The pass can save the residual stream and the MLP `down_proj` inputs, averaged over the first 8,192 answer tokens.
  - Teacher-forced: up to 1,024 texts and 524,288 tokens. The pass returns token log-probabilities, top-k, hidden states, MLP activations, and attention for full-attention layers.

Rules that the prompt gives:

- No gradient descent, no training, and no iterative fitting of weights to a loss.
- No simulation of the network in the kernel. The model runs only through the forward-pass tool.
- No edit to the final norm, the embedding, or the output head. No edit that only multiplies a whole tensor by a constant.
- The prompt asks for causal evidence, a dose-response over several strengths, and paired tests on problems that did not select the edit.

Other fact: at turn 29 the experimenters added one instruction: "EXPLAIN EVERY STEP" (`turns/t029/prompt.txt`). It asks for a plain-English explanation before each block.

## 1. The exact definition of the loop direction

Texts (pass 3, request in `turns/t009/answer.md`, results in `fwd_4`):

- Eight raw texts in the thinking template, with no generation. The teacher wrote all eight by hand.
- They make four matched pairs: a pay problem, a polynomial remainder, an arithmetic sequence, and a necklace total.
- In each pair, the user message and the first steps are the same. Then the texts split:
  - Productive text: a short correct check, for example "Checking: 200+160+400=760, consistent with the total."
  - Repetitive text: one doubt sentence, three or four times, word for word, for example "But is 20 the total? No, it is the remaining time." (three times).
- Text lengths in tokens: 124, 123, 131, 136, 107, 146, 98, 135 (`turns/t045/prompt.txt`).
- The first try (pass 2, `fwd_2`, turns t006 to t008) used `chats`. The teacher found that "chat template adds an empty think block" and repeated the pass with raw text (`work_notes/experiment_history.md`, line 3).

Positions: the last 36 tokens of each text. The code is `a[i,lens4[i]-36:lens4[i]].mean(0)` (`turns/t010/answer.md`).

Point in the layer: the residual stream (the returned hidden states). It is not the MLP output.

- For the MLP of layer `l`, the code uses `diffH[min(l+1,23)]`. The prompt defines hidden state 0 as the embeddings, so hidden state `l+1` is the residual stream after layer `l`.
- For layer 15 the teacher used hidden state 16. For layer 19 it used hidden state 20. The final report says "contrast at layer outputs16,20" (`work_notes/notes_before_pass26.md`, line 4) and "residual outputs16/20" (`work_notes/final_report.md`).

Formula: a difference of means, then a normalization to unit length (`turns/t010/answer.md`, `turns/t013/answer.md`).

```python
meansH = {l: np.stack([a[i, lens4[i]-36:lens4[i]].mean(0) for i in range(8)]) for l, a in H4.items()}
diffH  = {l: (a[1::2] - a[::2]).mean(0) for l, a in meansH.items()}   # repetitive minus productive, mean of 4 pairs
d = diffH[min(l+1, 23)]; d = d / d.norm()
```

The teacher did not use a principal component or a probe.

Later history of the vector:

- After kernel restarts, the teacher rebuilt `diffH` with the same code (t017, t025).
- At t044 it saved the unit vectors to `repetition_directions.pt`. It took them from the largest column of the alpha-1 weight change (`turns/t044/answer.md`).
- The notes say that the saved vectors have "cosine .996/.995 with reconstructed contrasts" (`work_notes/notes_before_pass26.md`, line 4).
- At t068 the teacher set the sign of the vector so that it points from productive to repetitive, over the last 24 tokens of the transfer texts in `fwd_24`.

## 2. How the teacher chose layers 15 and 19 and the 64 neurons

Layers (t010, output in `turns/t011/prompt.txt`):

- For each layer, the teacher split the growth of the contrast along the unit direction into an MLP part and an attention part.
- The MLP part is `(W_down @ mean activation difference) · d`.
- Layers 15 and 19 had the two largest MLP parts and the two largest block growths:

| Layer | Block growth | MLP part | Attention part |
|---|---:|---:|---:|
| 14 | 0.3617 | 0.1495 | 0.2122 |
| 15 | 0.472 | 0.2634 | 0.2086 |
| 19 | 0.5304 | 0.2497 | 0.2807 |
| 21 | 0.2575 | 0.1637 | 0.0938 |

- In the same run, the best single neuron was layer 19 neuron 11. Its score was 0.12001, and the next score was 0.02709.
- At t026 the teacher wrote this hypothesis: "repetition is maintained by amplification at both layers 15 and 19, so removing the shared direction at those two sites should have a larger causal effect."
- Layers 15 and 19 are full-attention layers (`turns/t001/prompt.txt`). I did not find a statement that the teacher used this fact in its choice.

Neurons (t068, output in `turns/t069/prompt.txt`):

- The neurons are columns of `mlp.down_proj.weight`.
- The score of neuron `j` is `(u @ W_down)[j]` times the mean, over the four `fwd_4` pairs, of (repetitive minus productive) activation of neuron `j`. The activation is the `down_proj` input, averaged over the last 24 tokens.
- The teacher ranked the neurons by this signed score, from high to low, and kept the top 64. It used only the original-model pass `fwd_4`, not correctness labels. The code:

```python
proj = u @ w
score = proj * torch.tensor((means[1::2] - means[::2]).mean(0))
top = torch.argsort(score, descending=True); neuron_subsets[layer] = top[:64]
```

- Share of the positive score mass in the top 64: 0.498 for layer 15 and 0.555 for layer 19.
- Rank correlation with the same score on the six new transfer pairs (`fwd_24`): 0.666 and 0.583. The two top-64 sets share 55 and 36 neurons.
- The notes say that the top 64 hold "~91%/98% of transfer repetitive-minus-productive writing contrast, but 50%/40% of full directional weight norm" (`work_notes/experiment_history.md`, lines 86 to 87).

## 3. The exact edit formula and the strength

Formula (`work_notes/final_report.md`; code in `turns/t105/answer.md`):

```text
W = W0 - 6 * outer(u, mask64 * (u @ W0))
```

- `W0` is the original `mlp.down_proj.weight` of layer 15 or layer 19, with shape [1024, 3584].
- `u` is the unit loop direction of that layer.
- `mask64` keeps the entries of `u @ W0` for the 64 selected columns and sets all other entries to zero.
- The code is `masked[ix] = writer[ix]` and then `w - float(alpha) * torch.outer(u, masked)`.

Strength: alpha = 6.

- Derived fact, from the formula: for a selected column, the new component along `u` is (1 - 6) = -5 times the old component.
- So the final edit does not only remove the direction. It reverses the direction and makes it five times larger in those 64 columns. Alpha = 1 is the only strength that removes the direction exactly.
- Relative change of the matrix: 10.64% for layer 15 and 8.11% for layer 19 (`edit/manifest.json`).

How the teacher chose the strength (dose ladder, `work_notes/experiment_history.md`, lines 61 to 101):

1. Full rank-one edit (all 3,584 columns) at alpha 1, 2 and 3. Each step gave a larger paired gain.
2. Full alpha 4. The generation engine did not start, so the teacher measured it only with teacher forcing (fwd32).
3. At a 16,384 token limit, the top-64 edit at alpha 4 and the full edit at alpha 4 gave the same gain (+26.56 pp). The full edit had 10 cut-offs that end in whitespace. The top-64 edit had 2.
4. At t105 the teacher chose to raise only the top-64 edit to alpha 6: "increasing only the selected repetition-writer columns yields a better benefit/side-effect tradeoff."
5. A teacher-forced check (pass 29, fwd44) and a 64-problem generation (pass 30) supported alpha 6.
6. The teacher then froze the edit: "No further dose increase before independent confirmation." (line 102). It tested no strength above 6.

## 4. What the teacher measured between candidate edits

Teacher-forced probes, with no generation:

- `fwd_4`: the four hand-written pairs.
- `fwd_19`: one productive text and one repetitive text with attention outputs.
- `fwd_24` and later passes: six new hand-written pairs plus two premature-close controls (`work_notes/transfer_probe_request.json`, 14 texts).
- The metric is the mean change in log-probability per token on the branch tokens, for productive branches and for repetitive branches. The teacher also measured the log-probability of a premature close.
- Results for repetitive / productive branches (nats per token, `work_notes/experiment_history.md`, lines 80 to 99):

| Edit | Repetitive | Productive |
|---|---:|---:|
| Full alpha 1 | -0.02410 | +0.00232 |
| Full alpha 2 | -0.04870 | +0.00343 |
| Full alpha 4 | -0.10372 | +0.00453 |
| Layer 15 only, alpha 4 | -0.08130 | -0.00058 |
| Layer 19 only, alpha 4 | -0.03446 | +0.00346 |
| Random control | +0.00955 | +0.00392 |
| Top-64 alpha 4 | -0.0645 | -0.0047 |
| Top-64 alpha 6 (final) | -0.10311 | -0.00903 |

- The random control used directions orthogonal to the loop direction, with the same Frobenius norm as full alpha 4.

Paired generations:

- All generations used GSM8K and MATH practice problems.
- The token limit was 32,768, except passes 27 and 28, which used 16,384.
- The sets were these:
  - Pass 1: 96 diagnostic problems, original model only.
  - Passes 4 to 6: a reused set of 32 problems.
  - Pass 10: validation_1, 64 GSM8K and 64 MATH problems (t044).
  - Passes 11 and 12: a dose set of 64 diagnostic problems, 4 samples each.
  - Pass 19: validation_2, 128 problems, 2 samples each.
  - Passes 27 and 28: 64 problems, 2 samples each, at 16,384 tokens.
  - Pass 30: 64 problems, 1 sample, at 32,768 tokens.
  - Passes 33 and 34: validation_3, 128 new problems.
  - Passes 35 to 50: validation_4 and final_1 to final_3 (`work_notes/fresh_splits.json`).

Numbers for the final edit (top 64, alpha 6) on practice data:

- Pass 29, teacher-forced: repetitive -0.103 nats/token, productive -0.009 nats/token.
- Pass 30, 64 problems, 32,768 tokens: 7/64 to 35/64, +43.75 pp, SE 6.63. The cut-off share went from 89.06% to 7.81%. Edited answers: 20 finished wrong and 4 finished with no answer.
- Passes 33 and 34, validation_3, 128 problems: 9/128 to 74/128, +50.78 pp, SE 4.44, 65 rescues, 0 losses.
- Pooled confirmation, 1,152 held-out problems, 1 sample each, 32,768 tokens (`work_notes/final_report.md`):
  - Accuracy went from 10.6% to 60.4%, a gain of +49.8 pp with SE 1.5. There were 579 rescues and 5 losses.
  - GSM8K went from 11.6% to 70.8%. MATH went from 9.5% to 50.0%.
  - Cut-off answers went from 1,038 to 212.
  - The median length of a correct answer went from 5,310.5 tokens to 9,171.5 tokens.
  - All 696 correct edited answers closed the thinking block.

## 5. What the teacher tried and rejected before the final edit

- Layer 19 neuron 11 output set to zero (t010, pass 4): +6.25 pp, SE 4.35 on 32 problems. The teacher called it "weak positive, not confirmed".
- Full rank-one removal in layer 19 only, alpha 1 (t013, pass 5): +6.25 pp, SE 4.35 on 32 problems. The teacher found it too weak.
- Full rank-one removal in the MLPs of layers 12 to 22 (t015): it did not run. The code failed with `NameError` after a kernel restart (`turns/t016/prompt.txt`), and the teacher did not try it again.
- Output of attention heads L15h7 and L19h5 multiplied by 0.5 (`fwd_22`): it was "less selective". The generation engine did not start, so the teacher removed the edit.
- Random orthogonal directions with the same norm (pass 23): +1.56 pp, SE 3.52. This was a control. The teacher removed it.
- Full layers 15 and 19, alpha 1 to 3: alpha 1 gave +24.22 pp (SE 3.96) on 128 held-out problems, alpha 2 gave +34.77 pp (SE 3.53), and alpha 3 gave +39.84 pp (SE 3.82, exploratory). Each strength was replaced by a stronger one.
- Full layers 15 and 19, alpha 4 at 16,384 tokens (pass 28): +26.56 pp, but with 10 whitespace cut-offs. The teacher replaced it with the top-64 edit.
- Top-64 alpha 4 at 16,384 tokens (pass 27): +26.56 pp, SE 4.45. The teacher replaced it with alpha 6.

## 6. Did the teacher measure the base model with thinking turned off?

Not found.

- The generation tool runs "with the evaluation's protocol (thinking enabled, the same sampling)" (`turns/t001/prompt.txt`). The prompt gives no switch to turn thinking off for a generation.
- Only the teacher-forced tool has an `enable_thinking` field. The teacher used `"enable_thinking": true` in pass 2 (t006, t007). After pass 2 it used raw text in the thinking template.
- I found no request with thinking off in `tool_calls.jsonl`.
- The only related mention is a check that the edit does not skip the thinking: "Correct edited median8072 tokens; not skipping thinking." (`work_notes/notes_before_pass26.md`, line 6).

## 7. How the teacher described the fault

The teacher describes repeated re-checking of a fact that is already known, written as the same text again and again. Its own words:

- t003: "the dominant failure is a long-reasoning attractor, not arithmetic errors: 85/96 outputs were cut off and every completed answer was correct."
- t004: "recurring lexical loops share a residual/MLP direction distinct from useful mathematical reasoning".
- t005: "recurrent self-correction has a shared activation signature".
- Turn 45 (`explanations.md`): "The saved diagnostic pairs contrast productive mathematical checking with repeated reconsideration of facts already established."
- Turn 55 (`explanations.md`): "contrasts useful verification with seven repetitions of the same doubt."
- Turn 47 (`explanations.md`) and `work_notes/final_report.md`: the cut-off tails have "about 95% duplicate eight-word sequences".

The base model's cut-off tails show the same pattern (`turns/t005/prompt.txt`). An example: "Wait, if he spends 1 hour walking, he cannot reach the end." repeats word for word.

The hand-written repetitive texts are strict word-for-word repetition of one doubt-and-answer sentence. The productive texts contain one check that is not repeated. So the contrast measures strict repetition inside a re-check, not re-checking itself.

## Differences from our method

Facts about the direction:

- Texts:
  - Blog: eight hand-written texts in four pairs. Each pair has the same problem and the same opening steps, and then splits into a productive check or a word-for-word repeated doubt.
  - Ours: a difference-of-means "repeated text" direction. I do not have the source texts of our direction, so I cannot compare the texts in more detail.
- Positions:
  - Blog: the last 36 tokens of each short text, about 100 to 150 tokens in total. It did not compare late positions with early positions in long generations.
  - Ours: not known to me.
- Point in the layer:
  - Blog: the residual stream after layer 15 and after layer 19 (hidden states 16 and 20), not the MLP output.
  - Ours: not known to me.
- Target token:
  - Blog: no specific token. The direction is a contrast of repetitive and productive states. The teacher measured `</think>` only as a control for premature close, and it found that probability "still extremely unlikely" (about -19 nats).
  - Ours: the direction "`</think>` minus the indent token" of the output head.
- Neuron selection:
  - Blog: the top 64 columns by (write along `u`) times (repetitive minus productive activation) on the hand-written pairs.
  - Ours: direct logit attribution at natural stopping points, which found the MLPs of layers 21 and 22 and the attention of layer 18.
- Strength:
  - Blog: alpha = 6, so the component along `u` of the 64 columns becomes -5 times its old value. This is a reversal, not a removal. Alpha = 1 removes the direction exactly.
- Token limit of the measurements:
  - Blog: 32,768 tokens, and 16,384 tokens in two passes. The blog never measured at 2,048 tokens.

Inference: why our layers-15-and-19 trial had no effect at 2,048 tokens.

These points are inference from the blog's numbers. They are not measured.

1. The token limit is the most likely cause.
   - The blog's edit works by ending loops that otherwise run to the 32,768 token limit: 576 of the 579 rescues were answers that the original model cut off.
   - In the blog's data, the median correct answer is 5,310 tokens for the original model and 9,172 tokens for the edited model.
   - At a 2,048 token limit, almost all answers of both models are cut off before the first loop matters. Thus a loop edit has no room to change the outcome.
   - The blog's own gain was smaller at 16,384 tokens (+26.56 pp) than at 32,768 tokens (+43.75 pp to +50.78 pp).
2. If our trial removed the direction (alpha 1) from 64 columns only, it was weaker than any edit that the blog measured with generation.
   - The blog's top-64 columns hold about 50% (layer 15) and 40% (layer 19) of the full directional weight norm.
   - The weakest generation edit in the blog was the full-column alpha 1 edit. Its gain was +24 pp at 32,768 tokens.
   - In teacher forcing, the full alpha 1 edit lowered repetitive branches by only 0.024 nats per token.
3. If our direction came from a different contrast, it can point elsewhere in the residual stream than the blog's vector.
   - For example, our direction can come from long model generations, a different layer point, or the MLP output.
   - The blog's vector had cosine about 0.7 with a direction from its own generations (pass 2, `work_notes/experiment_history.md`, line 3). So even two related contrasts can differ.

## Open items

- I did not read the details of our own "repeated text" trial: its texts, positions, layer point, strength (alpha), and practice set. Inference points 2 and 3 depend on them.
- I did not verify the 64 neuron indices in `work_notes/frozen_weights_manifest.json` against `edit/edit.npz`. I did not run any code on the edit file.
- I did not read the logs of p10c and p25c. They can use a different direction or a different strength.
- In `explanations.md`, turn 29 attributes the pass 6 result to "layer 19's MLP". But `turns/t028/prompt.txt` shows that the edits in layers 15 and 19 were both active. The teacher's later notes use the two-layer reading.
- I did not read the blog post itself, only the logs.

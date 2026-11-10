# Find the 0.8B repetition fault again as the teacher

Type: task
Status: resolved
Blocked by: 04, 05

## Question

Can Claude, as the teacher, find and remove the repetition fault of Qwen3.5-0.8B with the harness
tools and no published edit file, and how much wall-clock time and how many forward passes does it take?

Rules: no gradient, no optimizer, closed-form edits only. The teacher must not load the published
`edit.npz` or read the run notes of the blog. The teacher already knows the summary in
[What the blog method does and what it released](01-blog-method.md), so this test is not blind. The
answer states that limit.

Record the ledger totals, the final edit manifest (edit size and deviation), the token-limit share before and after, and each
tool that the teacher needed and did not have.

## Answer

Resolved on 2026-11-10. Run directory: `runs/calib-08b-teacher`. The teacher was Claude (Fable 5.2).

**Yes. The teacher found a fault and removed part of it in 29 minutes.** The fault that it found is
not the one that the blog describes, and the blog's deeper fault stays.

### Cost

| Measure | This run | Blog, 50-pass run |
|---|---|---|
| Wall-clock time of the search | 29 minutes (44 with the held-out evaluation) | 20.3 hours |
| Generation passes | 23 in the search, 31 in total | 50 passes of both kinds |
| Read passes | 14 | |
| Code runs | 22 | 80 |
| Student compute | 30 minutes of generation, 46 seconds of reads | |

A generation pass here is 64 problems at a 2,048 token limit and takes 42 seconds. A pass in the blog
is up to 512 problems at a 32,768 token limit and takes up to 30 minutes. The time gap comes mostly
from that pass size.

### What the teacher found

1. **The fault at this scale is not strict repetition.** 62 of 64 base answers reach the limit, and
   2 end in an exact loop. The model reaches an answer, then checks it again without end. Thinking
   closes in 2 of 64 answers.
2. **The closing token is suppressed.** In 127,000 positions, the probability of `</think>` is above
   1% at 6 positions. After the line "Final Answer: 123." it is 29%.
3. **Three components do the suppression.** Direct logit attribution at the natural stopping points
   shows that the MLPs of layers 21 and 22 and the attention of layer 18 push `</think>` down by
   about 6 nats against the indent token. At other positions they write almost nothing in that
   direction. The parts of the attribution sum to the final state with 0.6% error.

### The two edits

- **Edit A, remove the suppression.** `W' = W - 1.75 g (g^T W)` on three matrices, where `g` is the
  unit direction of `</think>` minus the indent token in the output head. Rank one per matrix.
- **Edit B, add a connection.** 150 neurons in the MLPs of layers 21 to 23 fire after a written
  conclusion (held-out AUC about 0.9). Each one also writes along `g`, in proportion to its
  selectivity. The strength gives +14 nats at the median conclusion.

### Result on 128 held-out problems

GSM8K practice items 100 to 227, the blog's sampling, thinking on, 4,096 token limit, one seed.
The search used items 1000 to 1063 only. Files: `heldout_128.json`, `deviation_wikitext.json`.

| Model | Correct (95% interval) | At the token limit | Weights changed | Model distance | KL on WikiText | Top-1 agreement |
|---|---|---|---|---|---|---|
| Base | 10.2% (6.0 to 16.6) | 93.8% | 0 | 0 | 0 | 100% |
| Blog's published edit | 21.9% (15.6 to 29.8) | 76.6% | 0.015% | 0.24% | 0.0162 | 94.1% |
| Edit A | 38.3% (30.3 to 46.9) | 56.2% | 0.98% | 0.17% | 0.0011 | 98.1% |
| Edit B | 37.5% (29.6 to 46.1) | 60.2% | 0.018% | 0.11% | 0.0052 | 96.5% |
| Base, thinking off | 56.2% (47.6 to 64.5) | 7.8% | 0 | 0 | | |
| Edit A, thinking off | 57.8% (49.2 to 66.0) | 8.6% | 0.98% | 0.17% | | |

Paired against the published edit: edit A fixes 27 items and breaks 6 (p = 0.0003). Edit B fixes 28
and breaks 8 (p = 0.001). Edit A leaves the thinking-off mode unchanged (56.2% to 57.8%). WikiText
perplexity stays at 24.8 for both edits.

### Limits that the writeup must state

- **Not blind.** The teacher knew the blog's summary before the search.
- **The edits help the model exit. They do not stop the second-guessing.** At higher strength both
  edits close the thinking after a few tokens, and the score then equals the thinking-off score.
- **A prompt switch beats every edit here.** The base model with thinking off scores 56.2%. The blog
  compared only against the base model with thinking on.
- **The token limit differs from the blog.** The blog's edit was made for a 32,768 token limit. At
  4,096 tokens it may be at a disadvantage. No run at the full limit was done.
- **One seed, 128 items.** The intervals are wide.

### Trials that failed, and why

- Removing a "repeated text" direction or a "about to write Wait" direction from the MLPs of layers
  8 to 22 made the fault worse. Zeroing a direction moves every position to the middle, and the
  normal state was on the far side.
- Removing the "Wait" token from six components cut the word from 3.65 to 0.27 per 1,000 tokens.
  The model then second-guessed with "If" and "Is". The fault is a state, not a word.
- The MLPs of layers 15 and 19 are the top writers of the repeated-text direction, as the blog
  reports. A neuron edit there with that direction had no effect at this limit.
- A random direction changed "finished" from 3% to 8%. A change of that size is noise at 64 items.

### Tools that the teacher needed and did not have at the start

All are now in `wsp/tools.py`: activations at chosen positions (`read_tokens` with `positions`),
the input of a module (`inputs=True`, for neuron activations), next-token log-probabilities for
chosen tokens (`read_logprobs`), token ids in each reply, and a deviation measure on text
(`deviation`). Still missing: output that streams during a long run, a helper for direct logit
attribution, and a helper that runs one edit trial. The teacher wrote the last two as code in the run.

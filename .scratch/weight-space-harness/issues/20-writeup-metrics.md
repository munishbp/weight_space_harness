# Choose the metric set for the writeup

Type: grilling
Status: resolved

## Question

Which measures does every run record, and which extra experiments does the writeup need, so that a
reader can judge the claim "a teacher edits weights and the student gets better at one task"?

The list below is the teacher's proposal of 2026-11-10. The user decides which of the costly items
are required. The free items go into the harness without a decision.

### Free: the harness records them on every run

- **Provenance.** The student name and revision hash, the harness commit, the teacher model and
  effort, the dataset revision, the sampling settings, the seeds, the hardware, and library versions.
- **Cost of the search.** The wall-clock time split into student compute and teacher thinking, the
  forward passes by kind, the student tokens processed, the code runs, the number of candidate edits
  tried and rejected, and the number of practice examples that the teacher saw.
- **Edit size and weight-space deviation.** Already in the edit manifest: the percent of weights,
  neurons, tensors, and layers changed; the relative change; the rank of the change. Add the module
  types touched (attention, MLP, vision tower) and the size of the edit file against the model size.
- **Rounding loss.** The change that the teacher asked for against the change that survives bfloat16.
- **Exact undo.** A check that `reset_edit` restores the original bits.
- **Uncertainty.** A confidence interval on each score, and a paired test on the items that flip
  (fixed against broken) between the original and the edited model.
- **Behavior side effects.** The answer length, the share of answers at the token limit, the share of
  replies with a format error, and the tokens generated per answer (this is the inference cost).

### Cheap: one extra run each

- **Output-space deviation.** `tools.output_deviation` on fixed reference prompts, and the perplexity
  on a neutral text set.
- **Representation drift.** The cosine between the hidden states of the original and the edited model,
  per layer, on the reference prompts.
- **The practice against held-out gap.** A large gap shows that the edit fits the practice items.
- **Near transfer.** The rest of the same benchmark, outside the target portion.
- **Dose response.** The score as the edit strength goes from 0 to 2 times the chosen value.
- **A random control.** An edit of the same size and place in a random direction. It shows that the
  gain comes from the direction and not from the disturbance.
- **A gain-against-time curve.** The held-out score after each candidate edit.

### Costly: the user decides

- **A prompt-only baseline.** The teacher spends the same time on the best system prompt and examples,
  with no weight change. A weight edit that does not beat a prompt has a weak claim.
- **An activation steering baseline.** The same direction added at inference, with no weight change.
- **Repeat runs of the teacher.** Does a second independent run find the same edit? This measures the
  variance of the method itself.
- **The gap to the next model size.** The share of the distance to the next larger model that the edit
  closes (for the 4B model, the published 9B score).
- **Survival under quantization.** Does the edit still work after 4-bit quantization? Users run small
  models quantized, and the 27B copies on this machine are quantized.
- **Teacher cost in tokens and money.** The harness cannot see the teacher's tokens. The teacher
  reports them from its own client.
- **Edits that stack.** Do two edits for two targets work together? This matters for a user with more
  than one application.

## Answer

Decided by the teacher on 2026-11-11 under the user's delegation ("I think you can do the rest
without me"). The rule: every block reports the six measures of the README, and the costly items
run where a block gives them at low cost.

| Item | State | Where |
|---|---|---|
| Provenance | recorded | `run.json` and `ledger.jsonl` of each run |
| Cost of the search | recorded | the ledger, and a time table in each ticket |
| Edit size, weight-space deviation, rank | recorded | `edit/manifest.json`, `verdict*.json` |
| Rounding loss | recorded | `kept_share_of_request` in each edit entry |
| Exact undo | checked | calibration (bit-exact), 27B (stored bytes) |
| Uncertainty | recorded | Wilson interval and paired flips in each table |
| Output-space deviation | recorded | KL, top-token agreement, perplexity on WikiText-2 |
| Practice against held-out gap | recorded | each ticket |
| Near transfer | recorded | text targets (ScreenSpot Pro), nine other robot tasks, 78 other web task types |
| Random control | done | Meta-World (two edits), MiniWoB++ |
| Dose response | partly | margin sweeps on unseen seeds |
| Prompt-only baseline | done | thinking off (0.8B), a system-prompt hint (Meta-World) |
| Edits that stack | done | button + door on Meta-World |
| Survival under quantization | done | the 27B edit lives in the NVFP4 format |
| Gap to the next model size | done for Meta-World | 4B base 63.0%, 4B with two edits 76.0%, 27B base 77.5% (practice seeds) |
| Base repeat for noise | added | MiniWoB++: 12 of 312 episodes differ between two base runs |
| Representation drift per layer | not done | the output-space measures cover the claim |
| Activation steering baseline | not done | it does not give the user a changed model |
| Repeat runs of the teacher | open | [Show that the teacher is interchangeable](18-teacher-interchange.md) |
| Teacher cost in tokens and money | not visible | the harness cannot see the tokens of the teacher |

One item is new: a repeat of the base run with the same batches. Without it a side-effect scan
reads noise as an effect.

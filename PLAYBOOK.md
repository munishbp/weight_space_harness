# Teacher playbook

Draft 3, written after seven runs: the calibration on Qwen3.5-0.8B; ScreenSpot Pro, Meta-World (two
faults), and MiniWoB++ on Qwen3.5-4B; MiniWoB++ on a quantized Qwen3.8-27B; and Meta-World on GLM-4.6V-Flash. A second teacher
(a different model with this file and the code only) then repeated the Meta-World edit in 30 minutes.
Draft 3 closes the gaps that it reported.

This playbook is for the teacher: a frontier model with a shell tool. The teacher reads the weights
and the activations of a smaller open model (the student) and writes edits to the weights directly.
The method uses no gradient and no optimizer. The playbook names no vendor. A teacher needs only the
ability to run shell commands and to read their output.

## 1. Start

```bash
uv sync
uv run python -m wsp.kernel --model <hugging-face-name> --run runs/<run-name>
```

The kernel loads the student one time and waits. It prints `kernel ready`. Send code to it:

```bash
uv run python -m wsp.client my_code.py      # or: echo "print(model)" | uv run python -m wsp.client -
```

The code runs in one Python namespace that persists between calls. The names `model`, `processor`,
`tools`, and `torch` exist at the start. Stop the kernel with `kill $(cat /tmp/wsp-kernel.pid)`.

The run directory records the work:

- `ledger.jsonl`: one line per forward pass, per edit, and per code run, with the time.
- `code/NNN.py` and `code/NNN.out`: each piece of teacher code and its output. The output file fills
  during the run, so `tail -f` shows the progress of a long run.
- `edit/manifest.json`: the size of the active edit and its distance from the original weights.
- `run.json`: the student, its revision, the harness commit, and the machine.

## 2. Rules

1. Do not use a gradient, an optimizer, or an iterative fit of weights to a loss. The kernel turns
   autograd off. A closed-form calculation from activations is allowed.
2. Change weights only through `tools.save_edit` or the functions in `wsp/edits.py`. They record
   every change.
3. Run the student only through `tools` and `wsp/evaluate.py`. They record every forward pass.
4. Split the data before the first pass: practice items for the search, held-out items for the
   verdict. Do not look at a held-out item during the search.
5. Fix the seed. The original and the edited model must see the same items and the same random draws.
6. Report a failed trial as clearly as a good trial.

## 3. Tools

| Call | Purpose |
|---|---|
| `tools.generate(conversations, max_new_tokens, thinking, append_ids, **options)` | Generate replies. Each reply has `text`, `ids`, `new_tokens`, `hit_limit`. `append_ids` forces the start of each reply, and the student writes the rest. |
| `evaluate.run(items, conversation, score, batch_size, **options)` | Generate and score a benchmark. `evaluate.summarize` gives the score with a 95% interval. |
| `tools.read(conversations, modules, append_ids, logprobs_of, inputs)` | Activations at the last position. Works with images. `append_ids` puts the start of a reply after each prompt, so the last position is a chosen step of the reply. `logprobs_of=[token ids]` adds the key `logprobs`: the log-probability of each token at that step. |
| `tools.read_tokens(sequences, modules, positions, inputs)` | Activations at chosen positions of a text. `inputs=True` gives the input of a module, which for an MLP output matrix is the neuron activations. |
| `tools.read_logprobs(sequences, token_ids, positions)` | The log-probability that each chosen token comes next. Text only. For a prompt with an image, use `tools.read` with `logprobs_of`. |
| `replay(result, steps)` in `wsp/bench/metaworld_text.py` and `wsp/bench/miniwob_screen.py` | The prompt of a recorded episode after its first `steps` actions. It gives the prompt at a fault step. |
| `attribution.token_direction` and `attribution.attribute` | The share of each model part in the logit of one token. |
| `layout.find(model)` | The names of the layers, MLPs, attention blocks, and output matrices of the student. |
| `tools.get_weight(name)` | One weight tensor as float32 on the CPU. |
| `edits.remove_direction`, `edits.remove_direction_from_neurons`, `edits.add_to_neurons`, `edits.fit_keys_to_values` | Closed-form edits. Section 5 says which edit fits which fault. |
| `attribution.attribute_last` | The same split as `attribution.attribute` for conversations with images. |
| `tools.save_edit`, `tools.reset_edit`, `tools.load_edit` | Write one tensor, undo, and load an edit file. |
| `tools.deviation(sequences)` | Distance between the original and the edited model on neutral text. |
| `battery.run()` and `battery.compare(before, after)` | The generalization battery, and the change of each set with a paired test. |
| `stats.wilson_interval`, `stats.paired_flips` | The interval of a score, and the items that an edit fixed and broke. |

## 4. Order of work

1. **Measure the baseline** on the practice items with the reference prompt. Check the score against
   a published score if one exists. A large gap means that the evaluation is wrong.
2. **Try the prompt first.** Measure the student with each cheap prompt option (for example thinking
   on and thinking off). A weight edit that a prompt switch beats has a weak claim. In the
   calibration run, the thinking-off prompt beat every edit.
3. **Sort the failures by cause.** Read 10 to 20 failed replies. Count each cause. Look for one cause
   that covers most failures. In a simulator, replay the plan of the student with one change at a
   time (one step removed, one step replaced). The change that turns a failure into a success names
   the fault exactly. In the Meta-World run, the plan of the student with no `close` step solved the
   button task 8 times of 10.
   Where no simulator replay fits, force the correct token at the step and let the student write
   the rest: `tools.generate(conversations, append_ids=[prefix + [token]] * len(conversations))`.
   `prefix` holds the token ids of the reply before the step. The success rate with the forced
   token is the ceiling of an edit at that step.
4. **Find where the student almost succeeds.** Read the probability of the correct token at the
   failure step: `tools.read_logprobs` on text, or `tools.read` with `append_ids` and `logprobs_of`
   on a prompt with an image. A
   correct token that has a visible probability but loses is a fault that hides an ability. A correct
   token with no probability at all points to a missing ability, and an edit is unlikely to add it.
5. **Attribute.** At the positions from step 4, use `attribution.attribute` on the direction
   "correct token minus the token that wins". Look for a part that pushes hard against the correct
   token at those positions and writes almost nothing at other positions. Such a part is a safe
   target. Check that `error` is small.
6. **Edit and run a trial.** Apply one edit, generate on the practice items, compare with the
   baseline, and call `tools.reset_edit()`. One trial must be cheap, so keep the practice set and the
   token limit small. Before a slow trial, read the probability of the correct token with the edit
   active, on practice items that the edit did not use. That check takes seconds. Run the slow
   trial only when it moves.
7. **Sweep the strength** and watch for a collapse. A strong edit can reach a good score through a
   trivial behavior. Measure a second quantity that shows the collapse (in the calibration run: the
   length of the thinking).
8. **Run a random control.** Apply an edit of the same form with a random direction. The size of its
   effect is the noise of the trial.
9. **Run the verdict.** Evaluate the final edit on the held-out items and on a near-transfer set:
   items of the same benchmark that are not the target. Run the battery and `tools.deviation`. Copy
   `edit/manifest.json`. The battery alone can miss a loss next to the target.
10. **Record** the time, the passes, the edit size, the held-out change, the battery, the deviation,
    and each failed trial.

## 5. Which edit fits which fault

Three kinds of fault appeared in the runs. Find the kind first. It decides the edit.

| Fault | How it shows | Edit | Result |
|---|---|---|---|
| **A suppressed token.** The student has the answer and a few parts block it. | The correct token has a visible probability at the failure step. Attribution finds parts that push against it there and write almost nothing elsewhere. | `edits.remove_direction` on those parts, with a strength sweep. | 0.8B, thinking that never closes: held-out 10.2% to 38.3%. 4B, the same fault: 77.3% to 89.1% with one matrix at strength 2, in 8 minutes. |
| **A habit in the wrong context.** The student applies a default that is right in most contexts. | The correct token has a visible probability (12% in our case). The same parts write the habit in every context, so removal would break the other contexts. | `edits.fit_keys_to_values` on one late MLP. The keys are the neuron activations at the failure step of the target context. The value is one fixed token direction (correct token minus habit token). The protection set holds the same step of the other contexts. | 4B, Meta-World button press: 1 of 20 to 19 of 20 practice episodes. No other task changed. |
| **A wrong choice between candidates from the input.** Two screen elements, or two positions to copy. | Both candidates have a visible probability. The late layers write the choice of the student, right or wrong. No part pushes against the correct one. | None found. A fit with one value per example copies the belief of the student. A difference-of-means direction with a perfect AUC did not steer the choice. | 4B, ScreenSpot Pro icons and Meta-World push: negative. A LoRA fine-tune fixes this kind. |

The recipe for the second kind is in `wsp/habit.py`:

```python
from wsp import habit
g = attribution.token_direction(correct_token, [habit_token])
keys, scale = habit.step_keys(target_prompts, mlp_out, last_layer, prefix)   # the step that writes the token
keep, _ = habit.step_keys(other_prompts, mlp_out, last_layer, prefix)        # the same step of other contexts
protect = habit.protection(prompts_with_replies, battery.neutral_text(64, 256), mlp_out)
habit.fit(mlp_out, [(keys, scale, g)], keep, protect, margin=12.0)
habit.stray_write(another_prompt, reply_ids, mlp_out, original_weight, g)    # check before a trial
```

How the pieces fit:

- `prefix` holds the token ids of the reply before the step (for a JSON action: `{"action": "`).
  `habit.step_keys` reads the last position of the prefix, which is the position that writes the token.
- `mlp_out` is one name from `layout.find(model)["mlp_outputs"]`. Take a layer in the last quarter of
  the model. `attribution.attribute_last` on the direction `g` at the step ranks the layers: the MLP
  that pushes hardest against the correct token is a good choice. Layers 27 and 28 of 32 both
  worked on the 4B model, and layer 60 of 64 on the 27B model.
- Each entry of `prompts_with_replies` is `(conversation, reply_ids, skip)`. `reply_ids` are the
  token ids of the reply that the student wrote (encode the reply text again). `skip` is the index
  of the target step inside `reply_ids`, counted from the start of the reply (`len(prefix) - 1`), or
  `None` for a prompt of another context.
- Some families put a norm between the MLP and the residual stream (GLM-4: `post_mlp_layernorm`).
  `layout.find` lists it in `mlp_write_norms`, the attribution reads the output of that norm, and
  `habit.fit` converts the values to the change before the norm. On a new family, check `error` of
  `attribution.attribute_last` first. A value near 1 means that the layout misses a module of this
  kind: add its name to `wsp/layout.py`.
- `replay(result, steps)` of the benchmark module gives the prompt at step `steps` of a recorded
  episode. One call starts one episode, so 200 prompts take minutes. Build them one time.

Rules for the recipe, each from a failed attempt:

1. **Check the fault with a forced replay first.** Force the correct token at the step and let the
   student continue. If the task still fails, the fault is elsewhere. In the email tasks the replay
   found a third missing step that the failures did not show.
2. **Protect every position.** `prompts_with_replies` holds the prompts of the target and of other
   contexts, each with the reply that the student wrote. Leave out only the target step. A
   protection of the target step alone wrote the direction at image and template positions.
3. **Give the same step of other contexts as `keep`.** In the protection, one step is one position
   among hundreds, so it has no weight there.
4. **Fit on one half of the practice seeds and read the probability on the other half.** Sweep the
   margin. A margin of 4 moved a probability from 0.12 to 0.39, and a margin of 12 moved it to 0.92.
5. **Run the base model two times before you read a side effect.** Two base runs on 312 browser
   episodes differed in 12 episodes. A side-effect scan needs that number next to it.
6. **Two fits on one matrix stack.** Fit the second with the first loaded.

## 6. Lessons from the runs

- **Removal sets a component to zero.** `remove_direction` on a direction from a difference of means
  made the fault worse two times. The normal state sat on the negative side, so zero was closer to
  the faulty state. Removal is safe when the part writes near zero at normal positions.
- **A fault can be a state and not a word.** Removing the token "Wait" removed the word and kept the
  behavior.
- **A difference of means can read the surface.** A direction that separates two classes of position
  with a high AUC from the first layers is likely a feature of the local text.
- **Write contrast texts by hand.** The blog's teacher got its direction from four pairs of short
  texts that differ in one behavior only. A pair like that isolates the behavior. Natural text mixes
  it with the surface of the text.
- **Sweep the strength far past 1.** The blog's final edit uses a strength of 6, which reverses the
  direction. Our trial of the same layers at a strength of 1 and 2 had no effect.
- **64 items give a noise of about 5 points.** Do not act on a smaller difference.
- **A larger student can have the same fault.** The 27B model made the same three wrong choices in
  the email tasks as the 4B model. The contexts and the recipe carried over with a new layer number.
- **A new family can break a silent assumption.** On GLM the attribution had an error of 1.06,
  because a norm sits between each MLP and the residual stream. The `error` value showed it before
  any fit ran. Check it on the first read of a new student.
- **A habit can be present and not be the fault.** GLM has the `close` habit of the 4B model in
  button-press, and a forced replay solved 1 of 20 there. The same replay on all tasks found the
  task where the habit is the fault (pick-place, 11 to 20 of 20) in 6 minutes.
- **A slow student changes the plan.** At 3 tokens per second one pass of 100 browser episodes takes
  20 minutes. Read probabilities on unseen seeds in place of a trial, and keep trials for the verdict.
- **Batch composition changes the activations.** Two batch sizes gave keys with a cosine of 0.994 to
  0.998. Run the original and the edited model with the same batches.
- **A change to `wsp/tools.py` needs a kernel restart.** Save the state that you need to a file first.
- **A perfect AUC is not a lever.** In the Meta-World push task, a direction after layer 12 separates
  "first move to the object" from "first move to the goal" with an AUC of 1.00 on unseen episodes.
  A write of that direction into the weights did not change one choice.
- **A simulator gives more practice items at no cost.** When 20 episodes cannot separate a gain from
  noise, run 40 new seeds. A gain of 5 episodes on 20 became 4 on 40, with p = 0.13.
- **A prompt sentence can fix the same fault and break a neighbour.** A hint in the system prompt
  fixed the button task (18 of 20) and cost 8 episodes in two other tasks. The edit cost none.
- **Protect every position of the prompt, not only the step that you edit.** A fit that protected
  the action step alone wrote its direction at image and template positions, 170 times the intended
  amount per prompt. The later layers read those positions, and three other tasks broke. With all
  positions in the protection, the stray writes fell to 2% and no task broke. Measure the write at
  every position of one unrelated prompt before a trial.
- **A repeat of a known recipe is fast.** The first habit edit took 27 minutes. The second, on the
  same matrix, took 15 minutes with one failed attempt and its diagnosis. The two edits stack.
- **Training has a price next to the target.** A LoRA fine-tune on icon targets raised held-out icons
  by 12 points and lowered text targets of the same benchmark by 12.5 points. The battery did not
  show it.

## 7. When to stop

Stop the search and record a negative result when the time box ends, when step 4 shows no position
where the student almost succeeds, or when no trial beats the random control.

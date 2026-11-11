# Teacher playbook

Draft 0, written after the calibration run on Qwen3.5-0.8B. A later run on a larger model will change it.

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
| `tools.generate(conversations, max_new_tokens, thinking, **options)` | Generate replies. Each reply has `text`, `ids`, `new_tokens`, `hit_limit`. |
| `evaluate.run(items, conversation, score, batch_size, **options)` | Generate and score a benchmark. `evaluate.summarize` gives the score with a 95% interval. |
| `tools.read(conversations, modules)` | Activations at the last prompt position. Works with images. |
| `tools.read_tokens(sequences, modules, positions, inputs)` | Activations at chosen positions of a text. `inputs=True` gives the input of a module, which for an MLP output matrix is the neuron activations. |
| `tools.read_logprobs(sequences, token_ids, positions)` | The log-probability that each chosen token comes next. |
| `attribution.token_direction` and `attribution.attribute` | The share of each model part in the logit of one token. |
| `layout.find(model)` | The names of the layers, MLPs, attention blocks, and output matrices of the student. |
| `tools.get_weight(name)` | One weight tensor as float32 on the CPU. |
| `edits.remove_direction`, `edits.remove_direction_from_neurons`, `edits.add_to_neurons` | Closed-form edits. |
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
   that covers most failures.
4. **Find where the student almost succeeds.** Use `tools.read_logprobs` on the failed replies. A
   correct token that has a visible probability but loses is a fault that hides an ability. A correct
   token with no probability at all points to a missing ability, and an edit is unlikely to add it.
5. **Attribute.** At the positions from step 4, use `attribution.attribute` on the direction
   "correct token minus the token that wins". Look for a part that pushes hard against the correct
   token at those positions and writes almost nothing at other positions. Such a part is a safe
   target. Check that `error` is small.
6. **Edit and run a trial.** Apply one edit, generate on the practice items, compare with the
   baseline, and call `tools.reset_edit()`. One trial must be cheap, so keep the practice set and the
   token limit small.
7. **Sweep the strength** and watch for a collapse. A strong edit can reach a good score through a
   trivial behavior. Measure a second quantity that shows the collapse (in the calibration run: the
   length of the thinking).
8. **Run a random control.** Apply an edit of the same form with a random direction. The size of its
   effect is the noise of the trial.
9. **Run the verdict.** Evaluate the final edit on the held-out items. Run the battery and
   `tools.deviation`. Copy `edit/manifest.json`.
10. **Record** the time, the passes, the edit size, the held-out change, the battery, the deviation,
    and each failed trial.

## 5. Lessons from the calibration run

- **Removal sets a component to zero.** `remove_direction` on a direction from a difference of means
  made the fault worse two times. The normal state sat on the negative side, so zero was closer to
  the faulty state. Removal is safe when the part writes near zero at normal positions.
- **A fault can be a state and not a word.** Removing the token "Wait" removed the word and kept the
  behavior.
- **A difference of means can read the surface.** A direction that separates two classes of position
  with a high AUC from the first layers is likely a feature of the local text.
- **64 items give a noise of about 5 points.** Do not act on a smaller difference.
- **A change to `wsp/tools.py` needs a kernel restart.** Save the state that you need to a file first.

## 6. When to stop

Stop the search and record a negative result when the time box ends, when step 4 shows no position
where the student almost succeeds, or when no trial beats the random control.

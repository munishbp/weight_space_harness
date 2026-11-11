# Final edit: button-press-topdown-v3, the "close" habit at step 2

Student: Qwen/Qwen3.5-4B, thinking off, greedy decoding, `max_new_tokens=64`, batch of 20 episodes (one task per batch).

## Fault

At step 2 (the history holds one move to the button), the student writes `{"action": "close"}`.
The probability is 0.82 for "close" and 0.12 for "move" (mean of seeds 0 to 9).
A replay of the plan of the student without the close step solves 17 of 20 episodes.
A forced replay that bans "close" at step 2, with the student writing the rest, solves 19 of 20.

## Edit

- Tensor: `model.language_model.layers.27.mlp.down_proj.weight` (the most negative MLP in the attribution of "move minus close" at the step; error 0.005).
- Form: `habit.fit` (closed-form least squares, `edits.fit_keys_to_values`), rank 1.
- Direction: `attribution.token_direction(3288, [5378])`: "move" (3288) minus "close" (5378).
- Prefix of the step: token ids `[4754, 1265, 763, 328]` (`{"`, `action`, `":`, ` "`). The step is the last prefix position.
- Target keys: step 2 of button-press-topdown-v3, seeds 0 to 9.
- Keep keys: step 2 of the nine other tasks, seeds 0 to 9 (90 keys).
- Protection: all positions of step-2 prompts with the reply of the student, seeds 0 to 4 of all ten tasks (the target step of the button prompts is left out), plus `battery.neutral_text(64, 256)`.
- Margin: 12.0. Strength: 1.0.
- Size: 1 tensor, layer 27, relative change of the tensor 0.116, relative change of the model 0.0037, 0.48% of the weights.
- Deviation on 32 neutral sequences: mean KL 0.00034, top-1 agreement 99.06%, perplexity 15.334 to 15.331.

## Load the edit

```python
tools.reset_edit()
tools.load_edit("runs/mw-4b-teacher2/edit_final.safetensors")
```

The file holds the full edited tensor, so `load_edit` needs no other state.

## Practice results (seeds 0 to 19, 20 episodes per task)

| Task | Base | Edit | Fixed | Broken | Random control |
|---|---|---|---|---|---|
| reach | 20 | 20 | 0 | 0 | 20 |
| push | 10 | 9 | 0 | 1 | 9 |
| pick-place | 20 | 20 | 0 | 0 | 20 |
| door-open | 0 | 0 | 0 | 0 | 0 |
| drawer-open | 0 | 0 | 0 | 0 | 0 |
| drawer-close | 20 | 20 | 0 | 0 | 20 |
| button-press-topdown | 0 | 20 | 20 | 0 | 1 |
| peg-insert-side | 19 | 19 | 0 | 0 | 19 |
| window-open | 15 | 17 | 2 | 0 | 17 |
| window-close | 20 | 20 | 0 | 0 | 20 |

A second base run gave the same 200 results. The random control (same form, a random direction of the same norm, margin 12) has the same push and window-open flips, so those flips come from a perturbation of layer 27, not from the direction.

Margin sweep on held-out half (button seeds 10 to 19), mean p("move"): base 0.12, margin 4 0.31, margin 8 0.60, margin 12 0.84. The other nine tasks did not move by more than 0.02. Margin 8 solved 19 of 20 button episodes.

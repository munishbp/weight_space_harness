# Edit the 4B model for the Meta-World target tasks

Type: task
Status: resolved
Blocked by: 24

## Question

Does a closed-form edit by the teacher raise the success of Qwen3.5-4B on the target tasks of
Meta-World (button-press-topdown, push, window-open), and what does it cost in the other seven tasks
and in the battery?

The target faults come from [Install one simulator benchmark and run the base model](24-simulator-install.md):
the model closes the gripper before a press, and its first move goes to the goal in place of the object.

The practice episodes use seeds 0 to 19. The held-out episodes use seeds 100 to 119. The other seven
tasks are the near-transfer set. Record the six measures of the map. Try the prompt first, as the
playbook says.

## Answer

Resolved on 2026-11-10. Run directory: `runs/mw-4b-teacher`. The teacher was Claude (Fable 5.2).

**Yes for one fault, no for the other. A closed-form edit of one matrix raises button-press-topdown
from 0 of 20 to 18 of 20 held-out episodes. The other nine tasks, the battery, and neutral text do not
change. The teacher had the edit after 27 minutes. The second fault (the first move goes to the goal)
did not yield to a closed-form edit.**

### The six measures for the button edit

| Measure | Value |
|---|---|
| Held-out, target task (button-press-topdown, seeds 100 to 119) | 0 of 20 to **18 of 20** |
| Held-out, all ten tasks (200 episodes) | 63.0% to 71.5%. 18 fixed, 1 broken, p = 0.00008 |
| Held-out, the other nine tasks | 126 to 125 of 180 (push 14 to 13, all others equal) |
| Random control (same fit, random direction of the same length) | 2 of 20 on the target task |
| Time | 27 minutes to the edit, 32 minutes with the practice trial. 195 generation passes, 143 read passes, 7 code runs. The whole search with the second fault: 53 minutes. The verdict (held-out, control, battery, deviation): 19 minutes more. |
| Edit size | 1 tensor (the MLP output matrix of layer 28), rank 1. 0.48% of the weights of the model. Relative change of the model 0.0049, of the matrix 0.15. |
| Deviation on WikiText-2 | KL 0.0003, top-token agreement 99.3%, perplexity 13.384 to 13.388 |
| Battery | MMLU-Pro 48.0 to 47.0 (0 fixed, 2 broken). CountBench 85.0 to 85.0. ChartQA 85.0 to 85.0. RefCOCO 90.0 to 89.5. GSM8K with thinking 75.0 to 75.0. No major drop. |
| Prompt hint with no edit (practice episodes) | Button 18 of 20, but push 10 to 6 and window-open 16 to 12. Overall 67.5% against 71.5% for the edit. |

The LoRA comparison is in
[Run the LoRA baseline on the Meta-World target tasks](26-lora-baseline-metaworld.md).

### How the teacher found the button edit

1. Base run: 126 of 200 practice episodes. Button-press 1 of 20.
2. Replay in the simulator: the plan of the model (move to the button, close, move to the goal) fails
   0 of 10. The same plan with no `close` succeeds 8 of 10. The fault is one primitive at the second turn.
3. Probability at that step: `move` has 12% and `close` has 82%. The model almost succeeds.
4. Attribution: the attention of layers 27 and 23 writes `close` in every task (button -1.15 and
   -0.82, pick-place -1.74 and -1.48). No part suppresses `move` in the button task only. Removal
   would break the tasks that need the grasp.
5. Edit: a least-squares fit on the MLP output matrix of layer 28 (`edits.fit_keys_to_values`). The
   keys are the neuron activations at the action-name step of the 20 practice button episodes. The
   value is one fixed direction, `move` minus `close`. The protection set holds the same step of the
   other nine tasks, the first and third turn of all tasks, and WikiText-2.
6. Check before the simulator: a fit on seeds 0 to 9 moves P(`move`) on seeds 10 to 19 from 0.12 to
   0.37 (margin 4), 0.74 (margin 8), 0.92 (margin 12). No other task moves by more than 0.03.
7. Practice trial with margin 12: button 1 to 19 of 20. 18 fixed, 1 broken over 200 episodes.

### The second fault: negative

In push, the first move goes to the object in 51 of 80 episodes (42 succeed) and to the goal in 24
(0 succeed). Both candidates have a visible probability at the first number.

| Trial | Result |
|---|---|
| A direction "object first minus goal first" after layer 12 (AUC 1.00 on unseen seeds), written into the MLP of layer 12 or 20 for the first move of push | No change: the object token leads in 20 of 35 unseen episodes before and after. |
| A fit with one value per episode (first token of the object position minus first token of the goal position) on the MLP of layer 28, 40 fit seeds | 40 unseen push seeds: 21 to 24 (margin 6) and 21 to 25 (margin 12), p = 0.25 and 0.13. The matrix changes by 88% and 176% of its norm. |

This is the same kind of fault as the icon targets of ScreenSpot Pro: a choice between two
candidates from the input. Drawer-open and door-open need a motion that the model does not write, so
no edit was tried.

### What this means

- The method works on the 4B model for a fault of the right kind: a habit that is wrong in one
  context, where the right action already has a visible probability.
- The edit is exact in scope. It changes one task and nothing else that the harness measures.
- 27 minutes is the first-time cost, with the fault unknown. The recipe is now in the playbook.
- Meta-World tasks are short (2 to 6 primitives), and the interface gives positions from the
  simulator state. The result does not show a long-horizon gain.

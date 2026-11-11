# Install MiniWoB++, run the base model, and choose a portion

Type: task
Status: resolved
Blocked by: 22

## Question

Does MiniWoB++ (`miniwob` from PyPI with a matching ChromeDriver) run on this machine with
Qwen3.5-4B as the agent, what does the base model score per task type, and which portion is the
target of a teacher edit?

The task: install, check how the Gymnasium wrapper applies the 10-second episode limit to the think
time of the model, write the agent loop with screenshots and the `computer_use` click format, run
the base model on practice seeds, and sort the failures by cause. Compare the base score with the
published 61.1.

## Answer

Run directory: `runs/mwob-4b-teacher`. Benchmark module: `wsp/bench/miniwob_screen.py`. Numbers of
the verdict: `verdict.json`. The edit: `edit_email_v3_L28_m12.safetensors`.

### Install and interface

- `miniwob` 1.1.0 installs from PyPI. Selenium downloads the ChromeDriver for the installed Chrome.
- The page ends an episode after 10 seconds and lowers the reward with time. The module sets
  `core.EPISODE_MAX_TIME` to a large value after the first reset, so the model has no time limit.
- The interface is ours: one screenshot per turn (4 times larger than the page), the task text, the
  actions so far, and one JSON action per turn (`click` at a point, `type`, `key`, `done`). The
  model gets no DOM. At most 10 actions per episode.
- One pass of 332 episodes (83 task types, 4 seeds) takes 6.4 minutes.

### Base model

59.0% of 332 episodes on 83 task types that clicks, text, and keys can solve. The published 61.1
uses another interface (BrowserGym with a DOM), so the two numbers do not compare directly.

### Target portion: the email task types

Five task types (reply, forward, and three mixed inbox types), 20 practice seeds and 20 held-out
seeds each. Base: 36 of 100 practice episodes, with reply and forward at 0 of 40.

The fault has three steps, all a wrong action name:

1. After the click on the email, the model writes `type`. The page needs a click on Reply.
2. After the click on Reply, the model writes `type`. The text field has no focus, so the page
   needs a click on the field first. The typed text is lost.
3. After the text, the model writes `key` (Enter). The page needs a click on Send.

A forced replay (the action name `click` forced at these steps, the model writes the rest) gives 58
of 100. The first replay forced steps 1 and 3 only and gave no gain. The page elements after each
action showed step 2.

### The edit

One least-squares fit on the MLP output matrix of layer 28, rank 2: `click` over `type` at steps 1
and 2, and `click` over `key` at step 3. The step after the click on the field (where `type` is
correct) and the action steps of the other 78 task types are in the fit with a zero value. The
protection holds every position of 487 prompts with their replies and neutral text.

On unseen practice seeds the probability of `click` goes from 0.22 to 0.89 at step 1, from 0.06 to
0.68 at step 2, and from 0.27 to 0.89 at step 3. The `type` step keeps 0.77 (base 0.89).

### Held-out result

| Held-out episodes (20 per task type) | Base | Teacher edit |
|---|---|---|
| email-inbox-reply | 0 | 11 |
| email-inbox-forward | 0 | 3 |
| email-inbox | 5 | 6 |
| email-inbox-noscroll | 12 | 18 |
| email-inbox-star-reply | 12 | 19 |
| All five email task types | 29.0% | 57.0% |
| The other 78 task types (312 episodes) | 63.8% | 65.4% |

- Email: 29 episodes fixed, 1 broken (p = 6e-8). Practice seeds: 56.0%.
- Other task types: 6 fixed, 1 broken (p = 0.125). No loss.
- Random control (the same fit with random directions): 29 of 100, no episode changes.
- Edit size: 1 tensor, rank 2, 0.51% of the weights. Relative change of the model 0.0093.
- WikiText-2: KL 0.0005, top-token agreement 99.2%, perplexity 13.384 to 13.381.
- Battery: MMLU-Pro 48.0 to 48.0, CountBench 85.0 to 85.0, ChartQA 85.0 to 85.0, RefCOCO 90.0 to
  89.5, GSM8K with thinking 75.0 to 74.0. No major drop.

### Time

The clock starts after the base scan.

| Minute | Step |
|---|---|
| 2.9 | Base model on the 100 practice episodes. |
| 5.3 | Forced replay of two steps: no gain. |
| 8.1 | Forced replay of three steps: 58 of 100. |
| 16.2 | Protection and fit sweep. The probabilities move on unseen seeds. |
| 19.0 | Practice trial: 56 of 100, 21 fixed and 1 broken. This edit is the first that works. |
| 25.1 | Scan of the other task types: 4 fixed, 10 broken. click-pie loses 3 of 4. |
| 38.9 | Refit with the action steps of other task types as zero-value keys. click-pie loses 1. |
| 52.0 | Refit with the reply positions in the protection. Other task types: 3 fixed, 9 broken. |
| 57.6 | The base model a second time: 4 fixed, 8 broken against its own first run. |

The verdict took 31 minutes more. The last 30 minutes of the search chased a side effect that was
mostly noise between two base runs. The cause: the composition of a batch changes the activations
(cosine 0.994 to 0.998 between a batch of 5 and a batch of 16). A base repeat with the same batches
at minute 19 would have shown that.

### LoRA baseline, 15 minutes

Examples: the steps of the practice episodes that succeed in the forced replay (140 examples, 28
epochs). Files: `runs/mwob-4b-lora15/`.

| Held-out | Base | Teacher edit | LoRA, 15 minutes |
|---|---|---|---|
| email-inbox-reply (20) | 0 | 11 | 19 |
| email-inbox-forward (20) | 0 | 3 | 17 |
| email-inbox, three mixed types (60) | 29 | 43 | 51 |
| All five email task types | 29.0% | 57.0% | 87.0% |
| The other 78 task types (312 episodes) | 63.8% | 65.4% | 61.5% |
| Episodes of other task types that change | | 7 (6 fixed, 1 broken) | 61 (27 fixed, 34 broken) |
| Weights changed | | 0.51% | 64.6% |
| WikiText-2: top-token agreement | | 99.2% | 94.5% |
| WikiText-2: perplexity (base 13.38) | | 13.38 | 13.81 |
| Battery: ChartQA, RefCOCO | 85.0, 90.0 | 85.0, 89.5 | 82.5, 87.0 |
| Battery: GSM8K with thinking | 75.0 | 74.0 | 84.0 |

LoRA gains 30 points more than the edit on the target. It learns where to click in the forward
form, which the edit cannot give. It changes one in five episodes of the other task types, with no
significant net loss (p = 0.44). ChartQA and RefCOCO fall by 2.5 and 3.0 points (p = 0.06 and 0.07).

### What this shows

- The habit recipe works on a multi-step computer use task. The gain is 28 points on held-out
  seeds of the target portion, with no loss on 78 other task types and on the battery.
- The first working edit came at 19 minutes on a new benchmark. The target of 10 to 15 minutes
  needs the recipe as one call and a base repeat for the noise. `wsp/habit.py` now holds the recipe.
- The edit fixes the choice of the action. It does not teach a new skill: forward stays at 3 of 20,
  because the click on the field goes to a wrong place there.

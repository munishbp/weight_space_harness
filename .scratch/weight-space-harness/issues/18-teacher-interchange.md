# Show that the teacher is interchangeable

Type: grilling
Status: resolved
Blocked by: 15

## Question

What evidence shows that a frontier model other than the one that built the harness can take the
teacher role?

Decide with the user:

- The interface that a teacher needs: a shell tool only, or a defined tool schema as well.
- The test: a second teacher runs the calibration from the playbook alone, with no help from this session.
- The second teacher: another Claude model as a subagent (the user rule allows Opus only), or a model
  from another vendor through an account that the user supplies.

This ticket waits for [Draft the teacher playbook](15-teacher-playbook.md), because the playbook is
what the second teacher reads.

## Decision

Decided by the teacher on 2026-11-11 under the delegation of the user ("I think you can do the rest
without me"). The user can overrule it.

- **Interface: a shell tool only.** The playbook states that a teacher needs no more. A tool schema
  is a second interface with no caller, so it does not exist.
- **Second teacher: Claude Opus 5.5 as a subagent.** The user rule allows Opus only for agents. A
  model from another vendor needs an account from the user, and none exists in this session. Opus is
  a different model from the teacher that built the harness (Claude Fable 5.2), with a fresh context.
  It is the same vendor, so the test does not show that the playbook works across vendors.
- **Test: one Meta-World edit on the 4B model from `PLAYBOOK.md` and `wsp/` alone.** The second
  teacher gets a running kernel and the practice seeds. It must not read the earlier run
  directories, `RESULTS.md`, `.scratch/`, or the git history. The main session runs the held-out
  verdict on its edit. Meta-World is the choice because one pass of 200 episodes takes 3 minutes.
- **The test is not blind.** The playbook names the button-press fault as its example. The test shows
  that a second model can operate the harness to a working edit. It does not show a second discovery.
- **What counts as the result:** the held-out change of the edit, the time, and each place where the
  playbook was wrong or incomplete.

## Answer

Run directory: `runs/mw-4b-teacher2` (scripts `b01_baseline.py` to `b09*.py` and `final.md` from the
second teacher; `heldout_verdict.py` and `verdict.json` from the main session).

A second teacher (Claude Opus 5.5, fresh context, `PLAYBOOK.md` and `wsp/` only) reached a working
edit in 30.2 minutes and 46 tool calls. It read nothing outside its boundary and ran no held-out seed.

| Minute | Step of the second teacher |
|---|---|
| 0.5 | Read `PLAYBOOK.md` and `README.md`. |
| 6.7 | Baseline, 200 practice episodes. |
| 8.5 | Replays of the plans of the student. |
| 10.5 | Probabilities of the action names at step 2, all tasks. |
| 11.2 | Forced replay on button-press: 19 of 20 with `close` banned. |
| 17.0 | Attribution, fit, margin sweep (4, 8, 12). |
| 21.7 | Practice trial, 200 episodes: button-press 0 to 20. |
| 25.7 | Random control, 200 episodes: button-press 1 of 20. |
| 29.4 | Second base run, 200 episodes: equal to the first in all episodes. |
| 30.2 | Final edit saved, deviation measured. |

Its edit: `habit.fit` on the MLP output matrix of layer 27 (the first teacher used layer 28), `move`
over `close`, margin 12, rank 1, 0.48% of the weights.

Held-out verdict, run by the main session (seeds 100 to 119):

| Measure | Second teacher | First teacher ([ticket 25](25-teacher-edit-metaworld.md)) |
|---|---|---|
| Time to the edit | 30 minutes, with the playbook | 27 minutes, fault unknown |
| button-press-topdown, held-out | 0 to 16 of 20 | 0 to 18 of 20 |
| All ten tasks, held-out | 63.0% to 71.5% | 63.0% to 71.5% |
| Episodes fixed and broken | 17 and 0 (p = 0.00002) | 18 and 1 |
| Weights changed | 0.48%, layer 27 | 0.48%, layer 28 |
| WikiText-2 top-token agreement | 99.2% (KL 0.0003) | 99.3% |
| Battery, largest drop | 0.5 points (three sets) | 1.0 point |

### What the test shows

- A second model can operate the harness to a held-out gain with the playbook and the code alone.
- It chose a different layer and got the same result, so the recipe does not depend on one layer.
- It ran the checks of the playbook without a reminder: the forced replay, the margin sweep on the
  unseen half, the random control, and the second base run.

### What the test does not show

- A discovery. The playbook names this fault as its example. The second teacher said so itself.
- A teacher from another vendor. That needs an account from the user.

### Gaps that the second teacher reported, and the fix for each

1. The playbook told the teacher to force a token with `tools.encode`, and `tools.generate` took no
   forced start. Fix: `tools.generate` takes `append_ids`. The first teacher had called
   `model.generate` directly for this, which the ledger does not record.
2. The playbook named `tools.read_logprobs`, which takes text only. Fix: the tool table names
   `tools.read` with `append_ids` and `logprobs_of` for prompts with images.
3. The playbook gave no rule for the layer, and it did not say what `prefix` and `skip` mean. Fix: a
   new part "How the pieces fit" in section 5.
4. No helper gave the prompt at step k of a recorded episode. Fix: `replay(result, steps)` in both
   benchmark modules. Three run scripts of the first teacher had the same function by hand.

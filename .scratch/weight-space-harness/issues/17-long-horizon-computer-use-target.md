# Choose the long-horizon computer use target

Type: grilling
Status: resolved

## Question

The goal of the harness is a small model that completes a specific long-horizon task. ScreenSpot Pro
is a single-step benchmark: one screenshot, one instruction, one click. Does the computer use verdict
stay on ScreenSpot Pro, or does a multi-step benchmark follow it or replace it?

Decide with the user:

- ScreenSpot Pro only, as a fast first signal, with the long-horizon proof left to the robotics block.
- ScreenSpot Pro first, then one portion of a multi-step benchmark (OSWorld-Verified or AndroidWorld).
- A multi-step benchmark in place of ScreenSpot Pro.

Facts for the decision. Qwen3.5-4B scores 35.6 on OSWorld-Verified and 58.6 on AndroidWorld per its
model card. A multi-step benchmark needs virtual machines or an emulator, and one evaluation pass
takes much longer than a ScreenSpot Pro pass. The teacher needs many passes, so the pass time sets
the cost of the whole block.

## Answer

Decided by the teacher on 2026-11-10 under the user's delegation.

**ScreenSpot Pro first, then one portion of a multi-step benchmark.** ScreenSpot Pro stays as the
first computer use target, because one pass takes minutes and the teacher needs many passes. It is a
single-step benchmark, so it cannot carry the long-horizon claim alone. A multi-step block follows it.

The choice between OSWorld-Verified and AndroidWorld depends on which one installs and runs on this
machine. That check is the ticket
[Check which multi-step computer use benchmark runs here](22-multi-step-benchmark-setup.md).

**Why not multi-step first.** A multi-step pass is slow. If the edit method fails on the fast
benchmark, the slow one costs days for the same answer. If it works, the grounding edit is also a
candidate cause of gains in multi-step tasks, which the multi-step block can test.

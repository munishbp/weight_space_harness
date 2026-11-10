# Choose the long-horizon computer use target

Type: grilling
Status: open

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

# Measure the 4B baseline on ScreenSpot Pro

Type: task
Status: claimed
Blocked by: 04

## Question

What does the base Qwen3.5-4B score on ScreenSpot Pro in our harness, per group, per application, and
per `ui_type`, and does the overall score match the published 60.3?

The task: download the dataset, write the eval with the prompt and the parser from
[ScreenSpot Pro format and the Qwen3.5 grounding prompt](03-screenspot-pro-format.md), and run the
full set with thinking on and with thinking off. Record the score, the share of replies with no
parsable point, and the wall-clock time of one full pass.

A score far from 60.3 means that the prompt, the resolution, or the parser is wrong. Fix the eval
before any edit work starts.

# Measure the 4B baseline on ScreenSpot Pro

Type: task
Status: resolved
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

## Answer

Resolved on 2026-11-10. Run directory: `runs/ss-pro-4b-baseline`.

**The harness scores Qwen3.5-4B at 59.4% on the full ScreenSpot Pro** (1,581 items, 95% interval 57.0
to 61.8, thinking off, greedy decoding, the official prompt). The published score is 60.3. The
evaluation is valid. Every reply held a parsable click. One full pass takes 36 minutes (1.37 seconds
per screenshot at a batch of 4, about 8,500 prompt tokens per screenshot).

**Thinking off is the reference setting.** On 156 icon items, thinking on scores 27.6% against 35.9%
(2 fixed, 15 broken, p = 0.002).

| Slice | Items | Correct |
|---|---|---|
| Text targets | 977 | 74.1% |
| Icon targets | 604 | 35.6% |
| Office | 230 | 75.7% |
| Scientific | 254 | 59.8% |
| OS | 196 | 59.7% |
| Dev | 299 | 58.5% |
| Creative | 341 | 55.4% |
| CAD | 261 | 50.6% |
| Smallest third of targets (median width 23 px) | 527 | 34% |
| Middle third (57 px) | 527 | 61% |
| Largest third (181 px) | 527 | 83% |

Per application, the range is 29.4% (AutoCAD) to 94.0% (EViews).

**Failures (642).** 206 are near misses (the click is within one box diagonal of the target), 162 are
medium, and 274 are far. If every miss of 5 output units or less were fixed, the overall score would
be 63.5% and the icon score 40.7%. The last digit of the output coordinates is not uniform (4, 0, and
6 appear about twice as often as 5) at every screen width, so it is a number prior of the model.

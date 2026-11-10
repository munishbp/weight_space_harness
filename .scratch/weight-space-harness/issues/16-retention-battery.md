# Choose the generalization battery

Type: grilling
Status: resolved

## Question

Which fixed set of general benchmarks does the harness run before and after every edit, so that a
verdict reports how far the edited model moves away from the original model's general ability?

Decide with the user:

- The benchmarks. They must cover abilities outside the target: language reasoning, instruction
  following, general image understanding, and tool calling. Benchmarks with a published score for
  the student are better, because the base run can be checked against that score.
- The sample size per benchmark. One full battery must be cheap enough to run after every edit.
- The rule for a drop: which size of drop the verdict calls major, with the measurement noise stated.
- The reference prompts for `tools.output_deviation`, which measures the output-space distance.

Facts for the decision. The blog measured MATH-500, GSM8K, IFEval, MMLU-Pro, and GPQA Diamond on the
0.8B model. The Qwen3.5-4B model card publishes scores for RefCOCO, CountBench, ERQA, BFCL-V4, and
others; see [Qwen3.5-4B facts](02-qwen35-4b-facts.md).

## Answer

Decided by the teacher on 2026-11-10 under the user's delegation. All six datasets exist on Hugging
Face (checked through the Hub API).

**Battery, version 1.** Each set uses a fixed seed and a fixed subset, so the same items run before
and after every edit.

| Set | Dataset | Items | Ability | Setting |
|---|---|---|---|---|
| MMLU-Pro | `TIGER-Lab/MMLU-Pro` | 200 | language reasoning, multiple choice | thinking off |
| GSM8K | `openai/gsm8k` test | 100 | multi-step math, and the token-limit fault | thinking on, 4096 token limit |
| CountBench | `nielsr/countbench` | 200 | general image understanding | thinking off |
| ChartQA | `lmms-lab/ChartQA` | 200 | chart and document reading | thinking off |
| RefCOCO | `lmms-lab/RefCOCO` | 200 | grounding on natural images | thinking off |
| WikiText-2 | `Salesforce/wikitext` test | 64 passages | perplexity and `tools.output_deviation` | no generation |

**Why these.** They cover language, math, vision, and grounding outside the target. RefCOCO is close
to GUI grounding, so it shows a side effect on the nearest general skill. CountBench and RefCOCO have
published scores for Qwen3.5-4B (96.3 and 88.1), so the base run checks the eval code. One set runs
with thinking on, because the only known fault of this model family (endless repetition) appears there.

**The rule for a drop.** The verdict reports every change with a 95% interval and a paired test. It
calls a drop "major" when the drop is 5 points or more and the paired p-value is below 0.05. With 200
items the interval is about 7 points each side, so a smaller change is inside the noise and the
verdict says so.

**Gaps in version 1.** No tool calling set (BFCL needs its own harness) and no instruction following
set (IFEval needs its checker package). Add them if an edit touches behavior that they measure.

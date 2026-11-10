# Choose the generalization battery

Type: grilling
Status: open

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

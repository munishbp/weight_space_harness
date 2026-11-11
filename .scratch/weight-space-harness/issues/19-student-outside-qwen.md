# Choose a student outside the Qwen family

Type: grilling
Status: open
Blocked by: 06

## Question

Both planned students are Qwen3.5 models. Does the map need a run on a student from another family
(for example GLM) to show that the harness is not specific to Qwen, and which model?

Decide with the user after the calibration, when it is clear which parts of the tools know the
Qwen3.5 architecture. A fact for the decision: three of every four Qwen3.5 layers are Gated DeltaNet
layers and not standard attention, so module names and edit recipes differ between families.

## Decision

Decided by the teacher on 2026-11-11 under the delegation of the user ("I think you can do the rest
without me"). The user can overrule it.

- **A run on a student from another family is needed.** The user named the aim: "bigger qwen, smaller
  qwen, GLM". Three Qwen students do not show that the tools are free of Qwen names.
- **The student is `zai-org/GLM-4.6V-Flash`** (revision `411bb4d7`). Reasons: the user named GLM. The
  model takes images, so the same two benchmarks run on it. It has 10.3 billion weights (20.6 GB in
  bf16) and fits on the 32 GB GPU with no quantization. The licence is MIT. `transformers` 5.22 has
  the class (`Glm46VForConditionalGeneration`), so the load runs no code from the repository.
- **The test has two levels.** Level 1 checks the tools: load, `layout.find`, generate with an
  image, read, one edit, an exact reset, and the deviation. Level 2 runs one benchmark portion that
  the 4B model has, and repeats the habit recipe if the base model shows a fault of that kind.
  A base model with no such fault is a valid result for level 2.

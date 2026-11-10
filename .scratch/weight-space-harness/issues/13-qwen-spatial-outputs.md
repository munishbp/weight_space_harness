# Survey Qwen3.5 spatial outputs and action interfaces

Type: research
Status: resolved

## Question

What pointing and spatial outputs does Qwen3.5 document, and what action interfaces let a
text-output vision-language model control a simulated robot?

Report the documented output formats of Qwen3.5 or Qwen3-VL for points, boxes, and 3D grounding, with
the cookbook prompts. Report the published action interfaces for a model with no action head: a
pixel point with a scripted controller, a choice of named skills, generated policy code, and text
tokens for end-effector moves. For each interface, report the zero-shot success of small general
models where a source states it.

Findings: `../research/qwen-spatial-outputs.md`

## Answer

Resolved on 2026-11-10 by a research subagent. The findings file holds the sources and the prompts.

- **Qwen output formats.** `point_2d` and `bbox_2d` on a 0 to 1000 scale, and `bbox_3d` in the camera
  frame in meters (needs camera intrinsics). These come only from the Qwen3-VL cookbooks, where every
  example runs the 235B model. No Qwen3.5 source documents an output format.
- **Zero-shot interfaces have results only for large models.** Pointing, keypoint, and code interfaces
  (PIVOT, MOKA, ReKep, VoxPoser, Code as Policies) report results for GPT-4-class, Codex, or Gemini.
- **Small open models score low.** The only zero-shot simulator results for 7B to 8B open models are
  in EmbodiedBench: 9.6 for Qwen2.5-VL-7B with binned 7-D pose text (GPT-4o: 28.9), and 4.7 on
  EB-ALFRED and 14.3 on EB-Habitat for skill choice (Claude: about 68). No result was found for a
  2B to 4B model.
- **Every point interface needs more than the model:** depth, camera calibration, and a segmentation
  or grasp model. Code as Policies in simulation reads object positions from simulator state.
- **Qwen-VLA reaches 97.9 on LIBERO with Qwen3.5-4B,** but only with a trained 1.15B action decoder.
- **Open items.** The prompts behind Qwen's RefSpatialBench and ERQA scores were not found. The
  Qwen-VLA paper may hold a zero-shot baseline; it was not checked.

**What this means for the robotics block.** The published zero-shot floor for a small open model is
near 10% or lower. The blog found that an edit can remove a fault and cannot add a missing ability.
The robotics verdict has the highest risk of a negative result on this map.

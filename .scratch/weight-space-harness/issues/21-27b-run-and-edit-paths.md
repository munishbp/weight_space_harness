# Find a way to run and edit the 27B model

Type: research
Status: resolved

## Question

Through which paths can the harness read activations from Qwen3.8-27B, write an edit to its weights,
and run the edited model on this machine (RTX 5090 with 32 GB VRAM, 45 GB RAM)?

Facts from this machine, checked on 2026-11-10:

- `~/models/Qwen3.8-27B-NVFP4` (21 GB): three safetensors shards from RadixArk, a mixed NVFP4 W4A4
  recipe made with NVIDIA Model Optimizer. The config says `model_type: qwen3_5`, 64 layers, hidden
  size 5120, FFN size 17408, `full_attention_interval: 4`. The architecture family is the same as
  Qwen3.5. The user reports that this copy runs on this machine with SGLang.
- A second directory holds one `Q4_K_M` GGUF file (16.8 GB) of a modified variant. It is not the base
  model, so the harness does not use it.
- `~/models/Qwen3.8-27B-GGUF`: two vision projector files only, no language model weights.
- No full-precision copy is on disk. The bfloat16 weights are about 54 GB by parameter count.
- `vllm`, `sglang`, and `llama-server` are not on the shell path of the project.

**User decision, 2026-11-10:** "If they're quanted, just use them. It's not a direct scale but it's
okay given the capacity I have." The 27B run uses the quantized NVFP4 copy. Path 1 below is the
target. Paths 2 and 3 matter only where path 1 needs them. The writeup states that the 27B result is
on a 4-bit model and is not a direct comparison with the bfloat16 runs.

Report for each path what works, what it costs, and what is unknown:

1. **Edit the NVFP4 copy in place.** How the safetensors store an NVFP4 tensor (packed values and
   scales), whether PyTorch can load the copy for activation reads, and whether a changed tensor can
   be quantized again alone.
2. **Download the full-precision model.** Whether `Qwen/Qwen3.8-27B` exists on Hugging Face and its
   size. Whether transformers can load it split across the GPU and the CPU, or with 4-bit or 8-bit
   quantization at load time for the `qwen3_5` architecture, while the edit math uses the
   full-precision tensors one at a time from disk.
3. **Serve the edited model.** Whether SGLang can load a checkpoint with replaced tensors, and what
   quantization step that needs.

Findings: `../research/27b-run-and-edit-paths.md`

## Answer

Resolved on 2026-11-10 by a research subagent. The findings file holds the on-disk format, the number
format, six designs, and the sources.

- **No library loads this checkpoint into PyTorch.** transformers 5.22 does not accept the
  `modelopt` quantization method, and the ModelOpt restore needs a saved state that the directory
  does not have. A custom loader is possible, because the rule that turns stored values into real
  weights needs no full-precision weights.
- **The checkpoint mixes formats.** 193 layers (all MLPs and the output head) are NVFP4, and 208
  attention layers are FP8. The vision tower stays bfloat16.
- **Activations are quantized too.** SGLang runs 4-bit and FP8 layer inputs. A PyTorch pass with
  bfloat16 activations reads slightly different values from the ones that SGLang serves.
- **Rounding can erase a small edit.** One step of the 4-bit format is about a third of the largest
  value in its block of 16 values. Quantizing one changed tensor again needs only that tensor.
- **Memory.** The stored tensors need about 20 GB on the GPU. SGLang holds about 30 GB while it
  runs, so the kernel and SGLang cannot hold the model at the same time. A bfloat16 copy (55.6 GB)
  fits neither the GPU nor the RAM.

**Path chosen by the teacher under the user's delegation: design D2.** A custom loader keeps the
stored tensors on the GPU and turns each quantized layer into real weights inside its forward pass.
The kernel then reads activations, writes edits, and generates in PyTorch, as it does for the other
students. An edited tensor goes back to its stored format, so the edit that the verdict measures is
the edit that survives 4-bit rounding.

**Limits that the writeup must state.** The run uses 4-bit weights with bfloat16 activations, which
is not what SGLang serves. The dequantize rule is not yet checked against real tensor bytes. The
generation speed in PyTorch for this model is not measured. A test of the edited checkpoint in
SGLang is a later step, not part of the first 27B run.

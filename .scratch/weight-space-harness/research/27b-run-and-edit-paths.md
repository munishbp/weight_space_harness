# 27B NVFP4: paths to read, edit, and serve

Ticket: how can the harness (a) read activations, (b) read, change, and write a weight tensor, and
(c) generate with the edited weights, on `~/models/Qwen3.8-27B-NVFP4`?

Date: 2026-11-10. Research only. No design is recommended.

Claim marks:

- **[M]** verified on this machine (file, header, or installed code read in this session).
- **[S]** stated by a primary source (URL or local file next to the claim).
- **[I]** my own inference. Check it before you depend on it.

## On-disk format

### Files and totals

- The checkpoint has 3 shards and 2194 tensors. The index gives `total_size` 21,921,428,072 bytes and
  `total_parameters` 18,164,649,200 (packed count, not the 27B logical count). [M]
  `model.safetensors.index.json`, safetensors headers.
- `config.json`: `architectures: ["Qwen3_5ForConditionalGeneration"]`, `model_type: qwen3_5`,
  text `model_type: qwen3_5_text`, 64 layers, `hidden_size` 5120, `intermediate_size` 17408,
  `vocab_size` 248320, `tie_word_embeddings: false`, `mtp_num_hidden_layers: 1`. [M]
- `layer_types`: 48 `linear_attention` (Gated DeltaNet) and 16 `full_attention`, with
  `full_attention_interval: 4` (layers 3, 7, 11, ... are full attention). [M] `config.json`
- `quantization_config`: `quant_method: "modelopt"`, `quant_algo: "MIXED_PRECISION"`,
  `producer: modelopt 0.47.0.dev0`, `ignore: ["mtp*", "mtp.layers.0*"]`,
  `kv_cache_scheme: {num_bits: 8, type: float}`. [M] `config.json`
- `hf_quant_config.json` repeats the same per-layer map (`quantized_layers`) with
  `kv_cache_quant_algo: "FP8"` and `exclude_modules: ["mtp*", "mtp.layers.0*"]`. [M]
- Source checkpoint: `Qwen/Qwen3.8-27B` revision `e13a4f0e35203116364e3b3f3f0c82f6ef1afd3c`, 18 BF16
  shards, 1199 tensors. Calibration: `abisee/cnn_dailymail`, 1024 samples, sequence length 512,
  `algorithm: "max"`. Entry point `examples/llm_ptq/hf_ptq.py`. Produced on 4x GB300, torch
  2.11.0+cu130, transformers 5.12.1. [M] `conversion-manifest.json`
- The recipe file itself (`/scratch/qwen38-27b-official-w4a4/recipe.yaml`) is not in the directory.
  Only its hash is. [M] `conversion-manifest.json`

### One NVFP4 linear layer: `model.language_model.layers.0.mlp.gate_proj` [M]

| Tensor name | dtype | shape | meaning |
|---|---|---|---|
| `model.language_model.layers.0.mlp.gate_proj.weight` | `U8` | `[17408, 2560]` | packed E2M1, two 4-bit values per byte; logical shape `[17408, 5120]` |
| `model.language_model.layers.0.mlp.gate_proj.weight_scale` | `F8_E4M3` | `[17408, 320]` | one scale per block of 16 input elements (5120 / 16 = 320) |
| `model.language_model.layers.0.mlp.gate_proj.weight_scale_2` | `F32` | `[]` | per-tensor global weight scale |
| `model.language_model.layers.0.mlp.gate_proj.input_scale` | `F32` | `[]` | per-tensor global activation scale, from calibration |

`down_proj` follows the same rule: `weight` `U8 [5120, 8704]`, `weight_scale` `F8_E4M3 [5120, 1088]`
(17408 / 16). `lm_head` is NVFP4 too: `lm_head.weight` `U8 [248320, 2560]`, `lm_head.weight_scale`
`F8_E4M3 [248320, 320]`, plus `weight_scale_2` and `input_scale`. [M]

### One FP8 linear layer: `model.language_model.layers.3.self_attn.q_proj` [M]

| Tensor name | dtype | shape |
|---|---|---|
| `model.language_model.layers.3.self_attn.q_proj.weight` | `F8_E4M3` | `[12288, 5120]` |
| `model.language_model.layers.3.self_attn.q_proj.weight_scale` | `F32` | `[]` (per tensor) |
| `model.language_model.layers.3.self_attn.q_proj.input_scale` | `F32` | `[]` (per tensor) |

`q_proj` has 12288 rows because `attn_output_gate: true` (24 heads x 256 x 2). [I]

### Precision by module (counts) [M]

| Precision | Modules | Count | Bytes on disk |
|---|---|---|---|
| NVFP4 (W4A4, group 16) | `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj` in all 64 layers, and `lm_head` | 193 linear modules | language-model `U8` 8.56 GB, `lm_head` 0.64 GB + 0.08 GB scales |
| FP8 E4M3 (W8A8, per tensor) | `linear_attn.in_proj_qkv`, `in_proj_z`, `out_proj` in 48 layers; `self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj` in 16 layers | 208 linear modules | language-model `F8_E4M3` 8.28 GB (FP8 weights and NVFP4 block scales together) |
| BF16 | `embed_tokens` (2.54 GB), `model.language_model.norm`, all layer norms, `q_norm`, `k_norm`, `linear_attn.in_proj_a`, `in_proj_b`, `conv1d`, `A_log`, `dt_bias`, `linear_attn.norm` | 450 language-model tensors | 2.60 GB |
| BF16 | vision tower `model.visual.*` | 333 tensors | 0.92 GB |
| BF16 | MTP head `mtp.*` | 15 tensors | 0.85 GB |
| F32 scalars | `weight_scale`, `weight_scale_2`, `input_scale` | 800 language-model + 2 `lm_head` | negligible |

- `.quant_summary.txt` agrees: every `model.visual.*` quantizer is `TensorQuantizer(disabled)`. 834
  quantizers are enabled and 1420 are disabled. [M]
- `tensor-audit.json`: `algorithm_base_counts: {FP8: 208, NVFP4: 193}`, `input_scale_tensor_count: 401`,
  `verdict: pass`. [M]

### Activations are quantized too

- NVFP4 layers: both the weight quantizer and the input quantizer are
  `(2, 1) bit fake block_sizes={-1: 16, 'type': 'dynamic', 'scale_bits': (4, 3)}`. Example:
  `layers.0.mlp.down_proj.input_quantizer ... amax=6.91e+00`. [M] `.quant_summary.txt`
- The manifest states `"nvfp4_scope": "MLP and lm_head weights with dynamic NVFP4 activations"`. [M]
  `conversion-manifest.json`
- So in the served model, each MLP input and the `lm_head` input are cast to 4-bit E2M1 with a
  per-16-element FP8 block scale computed at run time. The global activation scale is the stored
  static `input_scale`. [I], from the two lines above and the `input_scale` tensors.
- FP8 layers: input quantizer `(4, 3) bit fake per-tensor`, static calibrated amax (example
  `layers.3.self_attn.q_proj.input_quantizer ... amax=9.20e+01`). [M] `.quant_summary.txt`
- KV cache: FP8 E4M3. The checkpoint has no `k_scale` or `v_scale` tensors (0 found in the headers).
  The `k_bmm_quantizer` and `v_bmm_quantizer` show `amax=4.48e+02(const)`. [M]
  `qualification.json` records `"scales_calibrated": false` and the SGLang warning "Defaulting to
  scaling factors of 1.0". [M]

What this means for a plain PyTorch forward pass [I]:

- A forward pass on dequantized BF16 weights with BF16 activations is a W4A16 / W8A16 computation.
  It is not the W4A4 / W8A8 computation that SGLang runs. Activations read in PyTorch differ from
  the served activations by the activation quantization error.
- To match the served numerics, the PyTorch forward must fake-quantize each quantized layer input:
  NVFP4 per-16-block with `input_scale` as the global scale for the 193 NVFP4 layers, FP8 per-tensor
  with `input_scale` for the 208 FP8 layers, and FP8 for K and V in the 16 full-attention layers.
- The static `input_scale` values come from the original weights. An edit that moves a layer's
  input range above the calibrated amax causes clipping in the served model. Nothing re-calibrates
  them unless the harness does it.

### How the user runs it

- The user runs this checkpoint with SGLang in a separate private project on this machine. The launch
  details stay in that project and are not copied here.
- SGLang 0.5.20 or newer is required there. SGLang holds about 30 GB of the 32 GB card while it runs,
  so the harness kernel and SGLang cannot hold the model at the same time. [M]
- The environment that holds SGLang: not found.
- The model card command uses `sglang serve --trust-remote-code --model-path RadixArk/Qwen3.8-27B-NVFP4 --tp-size 4 ...`
  with no `--quantization` flag. [M] local `README.md` (copy of the Hugging Face card).

### Harness environment on this machine [M]

- `.venv`: torch 2.15.0, transformers 5.22.2, safetensors 0.8.0,
  accelerate 1.15.0, triton 3.9.0. Not installed: `nvidia-modelopt`, `compressed-tensors`, `sglang`,
  `vllm`, `flashinfer-python`, `flash-linear-attention`, `causal-conv1d`.
- `torch.float4_e2m1fn_x2` and `torch.float8_e4m3fn` exist in torch 2.15.
- GPU: RTX 5090, 32607 MiB, compute capability 12.0, 2928 MiB in use at the time of the check (no
  SGLang server up). RAM: 45 GB total, 30 GB available.

## NVFP4 number format

- Element: E2M1 (1 sign, 2 exponent, 1 mantissa bit). Magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6. Range
  about -6 to 6. Block: 16 values with one FP8 E4M3 scale. A second-level FP32 per-tensor scale.
  Native on Blackwell. [S]
  https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- ModelOpt code (`NVFP4QTensor`) [S]
  https://raw.githubusercontent.com/NVIDIA/Model-Optimizer/main/modelopt/torch/quantization/qtensor/nvfp4_tensor.py:
  - value table: `e2m1_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])`
    (bit 3 is the sign).
  - packing: `packed_weight = torch.bitwise_left_shift(q_weight[..., 1::2], 4) | q_weight[..., 0::2]`.
    The even element goes in the low nibble, the odd element in the high nibble.
  - global scale: `reduce_amax(input).float() / (E2M1_MAX * E4M3_MAX)`, that is `amax / (6 * 448)`.
  - block scale: `per_block_amax / (E2M1_MAX * weights_scaling_factor_2)`, zero replaced by 1.0,
    clamped to `[2**-9, 448]`, cast to `torch.float8_e4m3fn`.
  - dequantize: `per_block_scale = q_per_block_scale * per_block_quant_scale`, then each E2M1 value
    times the scale of its block.

The rule from stored values to real weights, for row `i`, column `j`, block `b = j // 16`:

```
byte  = weight[i, j // 2]                      # U8
code  = byte & 0x0F  if j is even else  byte >> 4
W[i, j] = E2M1[code] * float(weight_scale[i, b]) * float(weight_scale_2)
```

- The mapping of the code's `per_block_quant_scale` to the stored name `weight_scale_2` is [I]. It is
  consistent with the ModelOpt code above and with a third-party reconstruction (onnxruntime mobius
  PR, https://github.com/onnxruntime/mobius/pull/446, not a primary source). I did not decode
  tensor bytes on this machine to check it, because the boundary forbids loading tensor data.
- FP8 layers: `W = float(weight_fp8) * weight_scale`, one scale per tensor. vLLM states the format
  is "per-tensor weight scale (+ optional static activation scale)" [S]
  https://docs.vllm.ai/en/latest/features/quantization/modelopt/. The multiply direction is [I].

Resolution of an edit [I]: inside a block, the E2M1 grid steps are 0.5, 1, and 2 units of the block
scale, and the block scale is `block_amax / 6`. A step near the block maximum is about a third of
`block_amax`. An edit smaller than about a sixth of `block_amax` in an element can round to zero
change. FP8 E4M3 has 3 mantissa bits, so its relative step is 1/8 to 1/16 of the element magnitude.
Small low-rank edits can partly vanish at requantization in both formats.

## Reading activations

PyTorch paths:

- transformers 5.22 cannot load this checkpoint as quantized. `quant_method: "modelopt"` is not in
  `AUTO_QUANTIZATION_CONFIG_MAPPING`; `supports_quant_method` logs "Unknown quantization type". [M]
  `.venv/.../transformers/quantizers/auto.py` lines 296-298. The model would then load as BF16 and
  the `U8 [17408, 2560]` tensors would not match the `[17408, 5120]` parameters. [I]
- transformers has an `nvfp4` quantizer, but it raises "Loading pre-quantized NVFP4 checkpoints is
  not supported yet." [M] `.venv/.../transformers/quantizers/quantizer_nvfp4.py` line 54. It also
  requires sm100+ (line 79) and the `kernels` package.
- transformers 5.22 has `models/qwen3_5/modeling_qwen3_5.py`. The Gated DeltaNet and conv1d use hub
  kernels with a torch fallback (`use_kernel_func_from_hub_with_fallback`). [M] lines 235-286. The
  fallback speed for 48 linear-attention layers is not measured.
- ModelOpt restore (`mto.restore`, `restore_from_modelopt_state`) restores a `modelopt_state` saved
  by `mto.save`. The examples start from "# Initialize the original model". [S]
  https://nvidia.github.io/Model-Optimizer/guides/2_save_load.html. This directory has no
  `modelopt_state`. [M] Restore from an exported unified checkpoint: not found.
- The unified export (`export_hf_checkpoint`) targets TensorRT-LLM, vLLM, and SGLang. Load back into
  PyTorch or transformers: not found. [S]
  https://nvidia.github.io/Model-Optimizer/deployment/3_unified_hf.html
- compressed-tensors: a loader for the ModelOpt naming (`weight_scale_2`): not found.
- A custom PyTorch loader is possible from facts above [I]: build the transformers `qwen3_5` text model
  on the meta device, replace the 401 quantized `nn.Linear` modules with a module that holds the
  stored tensors and dequantizes with the rule above, and load the rest as BF16. Forward hooks then
  give any module output. This needs the original full-precision weights: no.

Server paths:

- SGLang: a request field `return_hidden_states` and a server mode `--return-hidden-states-mode last`;
  the result is in `meta_info["hidden_states"]`. [S]
  https://raw.githubusercontent.com/sgl-project/sglang/main/examples/runtime/hidden_states/hidden_states_server.py
  and https://docs.sglang.io/docs/basic_usage/sampling_params.md ("Whether to return hidden
  states."). Which layer "last" means, and any per-layer or per-module output: not found. Support for
  `qwen3_5` with this option: not found.
- vLLM: `extract_hidden_states` saves chosen layer outputs (`eagle_aux_hidden_state_layer_ids`) to a
  `.safetensors` file per request, shape `[num_tokens, num_extracted_layers, hidden_size]`. Chunked
  prefill must be off. [S]
  https://docs.vllm.ai/en/stable/features/speculative_decoding/extract_hidden_states/. Support for
  `qwen3_5` and this checkpoint: not found. Module outputs (not layer outputs): not found.
- vLLM loads ModelOpt `MIXED_PRECISION` with `quantization="modelopt_mixed"`. Without a native FP4
  GEMM it "falls back to weight-only (W4A16) execution via Marlin". [S]
  https://docs.vllm.ai/en/latest/features/quantization/modelopt/. vLLM is not installed here. [M]

## Writing an edit

- Read a tensor: the safetensors header gives the byte offsets. The teacher can read one tensor from
  disk on the CPU without a server. [M] for the header format, [I] for the use.
- Requantize one changed tensor without calibration [S for the functions, I for the procedure]:
  `NVFP4QTensor.quantize(input, block_size, weights_scaling_factor=None, weights_scaling_factor_2=None, ...)`
  returns the packed tensor, the block scale, and the global scale. `get_weights_scaling_factor_2(input)`
  computes the global scale from the tensor alone. Source: the ModelOpt file above. These functions
  need only the tensor. They do not need a calibration run.
- Two choices for the global scale [I]:
  - keep the stored `weight_scale_2`. The block scales clamp at 448, so an edit that raises
    `block_amax` above `6 * 448 * weight_scale_2` clips.
  - recompute `weight_scale_2` from the edited tensor. This is what the dynamic ModelOpt path does.
- FP8 tensors: new `weight_scale = amax / 448`, then cast `W / weight_scale` to `float8_e4m3fn`. [I]
- `input_scale` of the edited layer and of later layers stays at the calibrated value. A new
  calibration needs a forward pass over calibration data, but only `amax` statistics, not a full
  ModelOpt run. [I]
- ModelOpt is not installed. The same rule is about 30 lines of torch. [M] for the install state, [I]
  for the size.

## Serving the edited model

- SGLang `update_weights_from_disk` (`model_path`, `load_format`, `flush_cache`, `recapture_cuda_graph`,
  others) and `update_weights_from_tensor` (`serialized_named_tensors`, inside `begin_weight_update()`
  and `end_weight_update()`). The updates "only write checkpoint-format values", and
  `end_weight_update` runs `process_weights_after_loading` on quantized layers. [S]
  https://docs.sglang.io/docs/advanced_features/sglang_for_rl.md
- `update_weights_from_disk` is "Only applicable for models with the same architecture and parameter
  size." [S] https://docs.sglang.io/docs/basic_usage/native_api.md
- So the harness sends packed `U8` + `F8_E4M3` + `F32` values, not BF16. [I] from the two lines above.
- A SGLang endpoint to read a weight by name: not found in the pages read.
- Behaviour of these updates for ModelOpt NVFP4 on SM120: not found.
- A restart with a new directory also works as a fallback. Boot takes about 140 s. [M] migration doc
  line 377. Unchanged shards can be hard links. [I]
- MTP drafts (`--speculative-algorithm NEXTN`) come from the unedited BF16 `mtp.*` head. Speculative
  decoding verifies drafts, so output stays correct; the acceptance rate can fall. [I]
- The SGLang cookbook says "NVFP4 weights ~16.5GB"; the files are 21.9 GB. [S]
  https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B and [M].

## Fallback fact: full-precision weights

- `Qwen/Qwen3.8-27B` exists on Hugging Face. Licence: Apache 2.0. Tensor type BF16. 18 shards.
  `total_size` 55,562,855,904 bytes (55.6 GB). [S] https://huggingface.co/Qwen/Qwen3.8-27B and
  https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/model.safetensors.index.json
- BF16 for the text model only is about 53.8 GB (55.6 minus 0.92 vision and 0.85 MTP). [I]

## Workable designs

Memory figures are [I], computed from the header sizes above. A BF16 MLP block is
3 x 17408 x 5120 x 2 B = 0.53 GB. A BF16 attention block is about 0.21-0.23 GB.

| Design | What it needs | Memory | Unknowns |
|---|---|---|---|
| D1. Dequantize all 401 quantized modules to BF16, plain transformers forward, generate in PyTorch | custom loader (transformers cannot load it); no ModelOpt | about 53.8 GB BF16 text model. Does not fit 32 GB VRAM or 45 GB RAM (30 GB free). Needs disk offload or layer streaming | offload speed; W4A16 numerics differ from served W4A4 |
| D2. Keep stored tensors in PyTorch; dequantize each quantized module to BF16 inside its forward; hooks for (a); requantize for (b); PyTorch `generate` for (c) | custom module and loader, about 30 lines for the dequant rule; transformers `qwen3_5` modeling | about 20.1 GB resident on GPU (21.9 GB minus vision and MTP) plus about 0.5 GB transient per MLP and activations. Fits 32 GB only when SGLang is stopped | generation speed with torch fallback for Gated DeltaNet and no fused FP4 kernels; whether to fake-quantize activations to match SGLang |
| D3. D2 for (a) and (b); SGLang for (c) through `update_weights_from_disk` or `update_weights_from_tensor` or a restart | D2, plus a write path in checkpoint format | GPU cannot hold D2 and SGLang (about 30 GB) at the same time. Either swap them on the GPU, or run D2 on the CPU (about 20 GB packed in 30 GB free RAM, slow) | in-place update of ModelOpt NVFP4 on SM120; CUDA-graph recapture; activation mismatch between the D2 reading and the SGLang serving |
| D4. SGLang only: (a) through `return_hidden_states`, (b) read tensors from the safetensors files on CPU, write through `update_weights_from_tensor`, (c) SGLang | SGLang 0.5.20 env (location not found) | about 30 GB GPU for SGLang; CPU for tensor math, one tensor at a time | which layer "last" returns; no per-layer or per-module activations found; `qwen3_5` support for hidden states |
| D5. vLLM with `modelopt_mixed`: (a) `extract_hidden_states` per layer to disk, (b) as D4, (c) vLLM | vLLM install (not installed) | similar to SGLang; not measured | `qwen3_5` and SM120 W4A4 support; weight update API for this format not checked; module outputs not found |
| D6. Download BF16 `Qwen/Qwen3.8-27B`, edit in BF16, quantize to serve | 55.6 GB download, Apache 2.0 | BF16 does not fit 32 GB VRAM either; same offload issue as D1 | full ModelOpt quantization run per serve, or per-tensor requantization with old `input_scale` |

## Open items

1. Decode a few bytes of one NVFP4 tensor and one FP8 tensor to confirm the dequant rule and the
   meaning of `weight_scale_2`. This needs approval to read tensor data.
2. Find the SGLang environment and its launch flags. They are in a private project of the user.
3. SGLang: which tensor `return_hidden_states` gives for `qwen3_5`, and whether a per-layer capture
   exists. Not found.
4. SGLang: whether `update_weights_from_disk` or `update_weights_from_tensor` works for a ModelOpt
   `MIXED_PRECISION` checkpoint on SM120. Not found. Needs a test.
5. Speed of transformers `qwen3_5` with the torch fallback for 48 Gated DeltaNet layers. Not measured.
6. Whether the harness must fake-quantize activations in PyTorch to match served numerics. This is a
   design question for the user.
7. Policy for `weight_scale_2` and `input_scale` after an edit (keep or recompute). This is a design
   question for the user.
8. How much of a typical teacher edit survives NVFP4 and FP8 requantization. Needs a measurement.
9. The migration doc says the checkpoint has FP8 KV calibration scales. The files have none.

# Qwen3.5-4B facts

Type: research
Status: resolved

## Question

Does Qwen3.5-4B take image input, and what does the harness need to know to load and edit it?

## Answer

Researched on 2026-11-10 from the model card, the file list, and `config.json` of
https://huggingface.co/Qwen/Qwen3.5-4B.

- **Modalities.** Image and video input. The card says "Type: Causal Language Model with Vision Encoder".
- **Class.** `Qwen3_5ForConditionalGeneration`. The Hub widget loads it with `AutoModelForMultimodalLM`.
  The ScreenSpot Pro wrapper loads it with `AutoModelForImageTextToText`.
- **Language model.** 32 layers, hidden size 2560, FFN size 9216, vocabulary 248,320, tied output head.
- **Layer layout.** Eight groups of three Gated DeltaNet (linear attention) layers and one gated full
  attention layer. `full_attention_interval` is 4. Three of every four layers have no attention heads
  of the standard kind, so an edit recipe for a standard transformer does not carry over directly.
- **Vision encoder.** 24 layers, hidden size 1024, patch size 16, spatial merge 2, output width 2560.
  One visual token covers 32 by 32 pixels.
- **Weights.** Two safetensors shards, 9.34 GB, bfloat16. Downloaded to the Hugging Face cache on 2026-11-10.
- **Context.** 262,144 tokens.
- **Thinking.** On by default. Turn it off with `enable_thinking=False` in the chat template.
- **Licence.** Apache 2.0.
- **Published scores for the 4B model.** ScreenSpot Pro 60.3, OSWorld-Verified 35.6, AndroidWorld
  58.6, ERQA 54.0, RefSpatialBench 54.6, EmbSpatialBench 81.3.
- **Not stated on the card.** The grounding prompt, the coordinate format, and the settings for the
  ScreenSpot Pro score.

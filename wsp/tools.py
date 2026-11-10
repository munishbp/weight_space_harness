"""Functions that the teacher calls inside the kernel.

Every model call goes through `generate` or `read`, and each call writes one ledger line.
The ledger is the record of the forward-pass count and the wall-clock time of a run.
"""

import json
import re
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

MODEL = None
PROCESSOR = None
RUN_DIR: Path | None = None
# The original value of each edited tensor, on the CPU. `reset_edit` restores from here.
ORIGINALS: dict[str, torch.Tensor] = {}
# The metrics of each edited tensor, from `tensor_metrics`. The manifest sums them.
ENTRIES: dict[str, dict] = {}
_MODEL_SQUARED_NORM: float | None = None


def start_run(run_dir: Path) -> None:
    global RUN_DIR
    RUN_DIR = run_dir
    (RUN_DIR / "edit").mkdir(parents=True, exist_ok=True)


def log(kind: str, **fields) -> None:
    line = {"t": round(time.time(), 2), "kind": kind, **fields}
    with (RUN_DIR / "ledger.jsonl").open("a") as ledger:
        ledger.write(json.dumps(line) + "\n")


def load_model(name: str, revision: str | None = None) -> None:
    global MODEL, PROCESSOR
    from transformers import AutoModelForImageTextToText, AutoProcessor

    PROCESSOR = AutoProcessor.from_pretrained(name, revision=revision)
    # Left padding keeps the last position of every row on the final prompt token,
    # which batched generation and last-token reads both need.
    PROCESSOR.tokenizer.padding_side = "left"
    MODEL = AutoModelForImageTextToText.from_pretrained(
        name, revision=revision, dtype=torch.bfloat16, device_map="cuda"
    )
    MODEL.eval()
    # The manifest reports each edit as a share of this norm, so measure it before any edit exists.
    _model_squared_norm()
    _write_run_file(name, revision)


def _write_run_file(name: str, revision: str | None) -> None:
    """Record what a reader needs to repeat the run: the student, the code, and the machine."""
    import subprocess

    import transformers

    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    info = {
        "student": name,
        # No value means that the run used the newest revision at the time of the download.
        "student_revision": revision,
        "student_weights": sum(parameter.numel() for parameter in MODEL.parameters()),
        "student_dtype": str(MODEL.dtype),
        # No value means that the harness code had no commit when the run started.
        "harness_commit": git.stdout.strip() if git.returncode == 0 else None,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (RUN_DIR / "run.json").write_text(json.dumps(info, indent=2))


def encode(conversations: list[list[dict]], thinking: bool = False) -> dict:
    """Turn a batch of chat conversations into model inputs on the GPU."""
    inputs = PROCESSOR.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"padding": True},
        enable_thinking=thinking,
    )
    return inputs.to(MODEL.device)


class PresencePenalty:
    """Lower the logit of each token that the reply already holds, as `presence_penalty` does in vLLM."""

    def __init__(self, penalty: float) -> None:
        self.penalty = penalty
        self.seen: torch.Tensor | None = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.seen is None:
            # The first call sees the prompt only. The prompt does not count as presence.
            self.seen = torch.zeros_like(scores, dtype=torch.bool)
        else:
            self.seen.scatter_(1, input_ids[:, -1:], True)
        return scores - self.penalty * self.seen


def generate(
    conversations: list[list[dict]],
    max_new_tokens: int = 64,
    thinking: bool = False,
    presence_penalty: float = 0.0,
    **options,
) -> list[dict]:
    """One forward pass of kind `generate` for a batch of conversations.

    Decoding is greedy unless `options` say otherwise. `options` go to `model.generate`.
    Each reply holds `text`, `ids`, `new_tokens`, and `hit_limit`. `ids` are the token ids of the
    reply. `hit_limit` is true when the model wrote `max_new_tokens` tokens and no end token.
    """
    started = time.time()
    inputs = encode(conversations, thinking)
    if presence_penalty:
        # The penalty holds state for one generation, so each call needs a new one.
        options["logits_processor"] = [*options.get("logits_processor", []), PresencePenalty(presence_penalty)]
    output = MODEL.generate(**inputs, max_new_tokens=max_new_tokens, **({"do_sample": False} | options))
    new = output[:, inputs["input_ids"].shape[1] :]
    end_ids = MODEL.generation_config.eos_token_id
    end_ids = torch.tensor(end_ids if isinstance(end_ids, list) else [end_ids], device=new.device)
    ended = torch.isin(new, end_ids).any(dim=1)
    # A row that ends early carries padding after its end token.
    lengths = (new != PROCESSOR.tokenizer.pad_token_id).sum(dim=1)
    texts = PROCESSOR.batch_decode(new, skip_special_tokens=True)
    log(
        "generate",
        n=len(conversations),
        prompt_tokens=int(inputs["attention_mask"].sum()),
        new_tokens=int(lengths.sum()),
        hit_limit=int((~ended).sum()),
        seconds=round(time.time() - started, 2),
    )
    return [
        {"text": text, "ids": row[: int(length)].tolist(), "new_tokens": int(length), "hit_limit": not bool(done)}
        for text, row, length, done in zip(texts, new, lengths, ended)
    ]


def prompt_ids(conversation: list[dict], thinking: bool = False) -> list[int]:
    """The token ids of one prompt. `prompt_ids(c) + reply["ids"]` is the full text that the model saw."""
    return encode([conversation], thinking)["input_ids"][0].tolist()


def read_tokens(sequences: list[list[int]], modules: list[str]) -> dict[str, list[torch.Tensor]]:
    """One forward pass of kind `read` over token sequences: the output of each named module at every position.

    The result maps a module name to one float32 CPU tensor of shape [length, width] per sequence.
    This works for text tokens only, because image tokens need their pixel input.
    """
    started = time.time()
    named = dict(MODEL.named_modules())
    missing = [name for name in modules if name not in named]
    if missing:
        raise KeyError(f"These module names do not exist in the model: {missing}. No forward pass ran.")
    longest = max(len(sequence) for sequence in sequences)
    pad = PROCESSOR.tokenizer.pad_token_id
    # Left padding, as in `encode`, so position -1 is the last real token of every row.
    ids = torch.tensor([[pad] * (longest - len(s)) + list(s) for s in sequences], device=MODEL.device)
    mask = torch.tensor([[0] * (longest - len(s)) + [1] * len(s) for s in sequences], device=MODEL.device)
    captured: dict[str, list[torch.Tensor]] = {}

    def hook_for(name: str):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[name] = [row[longest - len(s) :].float().cpu() for row, s in zip(hidden, sequences)]

        return hook

    handles = [named[name].register_forward_hook(hook_for(name)) for name in modules]
    try:
        MODEL(input_ids=ids, attention_mask=mask)
    finally:
        for handle in handles:
            handle.remove()
    log(
        "read",
        n=len(sequences),
        tokens=sum(len(s) for s in sequences),
        modules=len(modules),
        seconds=round(time.time() - started, 2),
    )
    return captured


def read(conversations: list[list[dict]], modules: list[str], thinking: bool = False) -> dict[str, torch.Tensor]:
    """One forward pass of kind `read`: the output of each named module at the last prompt position.

    The result maps a module name to a float32 CPU tensor of shape [batch, width].
    """
    started = time.time()
    named = dict(MODEL.named_modules())
    missing = [name for name in modules if name not in named]
    if missing:
        raise KeyError(f"These module names do not exist in the model: {missing}. No forward pass ran.")
    captured: dict[str, torch.Tensor] = {}

    def hook_for(name: str):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[name] = hidden[:, -1, :].float().cpu()

        return hook

    handles = [named[name].register_forward_hook(hook_for(name)) for name in modules]
    try:
        inputs = encode(conversations, thinking)
        MODEL(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    log("read", n=len(conversations), modules=len(modules), seconds=round(time.time() - started, 2))
    return captured


def get_weight(name: str) -> torch.Tensor:
    """A float32 CPU copy of one parameter."""
    return MODEL.get_parameter(name).detach().float().cpu()


def save_edit(name: str, tensor: torch.Tensor) -> dict:
    """Replace one parameter with `tensor`, in the model and in the edit directory of the run."""
    parameter = MODEL.get_parameter(name)
    if tuple(tensor.shape) != tuple(parameter.shape):
        raise ValueError(
            f"The edit for {name} has shape {tuple(tensor.shape)}, but the parameter has shape "
            f"{tuple(parameter.shape)}. The model is unchanged."
        )
    if not torch.isfinite(tensor).all():
        raise ValueError(f"The edit for {name} contains NaN or infinity. The model is unchanged.")
    if name not in ORIGINALS:
        ORIGINALS[name] = parameter.detach().cpu().clone()
    parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
    # The stored dtype can be bfloat16, so a small change can round to zero.
    # The metrics describe the change that survives in the model, not the change that the teacher asked for.
    original = ORIGINALS[name].to(parameter.device).float()
    entry = {"name": name, **tensor_metrics(original, parameter.detach().float())}
    ENTRIES[name] = entry
    _write_edit_files()
    log("save_edit", **entry)
    return entry


def tensor_metrics(original: torch.Tensor, edited: torch.Tensor) -> dict:
    """The size of one edit and its distance from the original tensor. Both inputs are float32."""
    delta = edited - original
    changed = delta != 0
    original_norm = float(original.double().norm())
    metrics = {
        "shape": list(delta.shape),
        "weights": delta.numel(),
        "changed_weights": int(changed.sum()),
        "changed_weights_pct": 100 * int(changed.sum()) / delta.numel(),
        # A tensor of zeros has no relative scale.
        "relative_change": float(delta.double().norm()) / original_norm if original_norm else None,
        "max_abs_change": float(delta.abs().max()),
        "cosine_to_original": float(
            torch.nn.functional.cosine_similarity(original.flatten().double(), edited.flatten().double(), dim=0)
        ),
        "delta_squared_norm": float(delta.double().pow(2).sum()),
    }
    if delta.ndim == 2:
        # In a linear layer, a row holds the input weights of one output neuron,
        # and a column holds the output weights of one input neuron.
        rows = int(changed.any(dim=1).sum())
        columns = int(changed.any(dim=0).sum())
        singular = torch.linalg.svdvals(delta)
        metrics |= {
            "rows_changed": rows,
            "rows_changed_pct": 100 * rows / delta.shape[0],
            "columns_changed": columns,
            "columns_changed_pct": 100 * columns / delta.shape[1],
            # Rounding to bfloat16 adds noise of full rank, so the rank counts only
            # the singular values above 1% of the largest one.
            "delta_rank": int((singular > 0.01 * singular[0]).sum()) if singular[0] > 0 else 0,
        }
    return metrics


def output_deviation(conversations: list[list[dict]], thinking: bool = False) -> dict:
    """Two forward passes: the distance between the next-token distributions of the original and the edited model.

    The result holds the mean KL divergence from the original to the edited model and the share of
    positions where both models rank the same token first, over all attended positions.
    """
    started = time.time()
    inputs = encode(conversations, thinking)
    mask = inputs["attention_mask"].bool()
    edited = torch.log_softmax(MODEL(**inputs).logits.float(), dim=-1)
    edited_values = {name: MODEL.get_parameter(name).detach().clone() for name in ORIGINALS}
    try:
        for name, original in ORIGINALS.items():
            parameter = MODEL.get_parameter(name)
            parameter.copy_(original.to(parameter.device))
        base = torch.log_softmax(MODEL(**inputs).logits.float(), dim=-1)
    finally:
        # The edit must return when the second pass fails. If it does not, the kernel holds
        # the original model while the edit files say that the edit is active.
        for name, value in edited_values.items():
            MODEL.get_parameter(name).copy_(value)
    kl = (base.exp() * (base - edited)).sum(dim=-1)[mask]
    agree = (base.argmax(dim=-1) == edited.argmax(dim=-1))[mask]
    result = {
        "mean_kl": float(kl.mean()),
        "max_kl": float(kl.max()),
        "top1_agreement_pct": 100 * float(agree.float().mean()),
        "positions": int(mask.sum()),
    }
    log("output_deviation", n=len(conversations), seconds=round(time.time() - started, 2), **result)
    return result


def reset_edit(names: list[str] | None = None) -> None:
    """Restore the original value of the named parameters, or of every edited parameter."""
    for name in list(ORIGINALS) if names is None else names:
        parameter = MODEL.get_parameter(name)
        parameter.copy_(ORIGINALS.pop(name).to(parameter.device))
        ENTRIES.pop(name)
    _write_edit_files()
    log("reset_edit", names=names)


def load_edit(path: str) -> list[dict]:
    """Apply an `edit.safetensors` file from an earlier run to the model."""
    return [save_edit(name, tensor) for name, tensor in load_file(path).items()]


def _model_squared_norm() -> float:
    """The squared norm of all original weights. The first call computes it, and later calls reuse it."""
    global _MODEL_SQUARED_NORM
    if _MODEL_SQUARED_NORM is None:
        if ORIGINALS:
            raise RuntimeError(
                "The model norm must be measured before the first edit. "
                "Call tools.reset_edit(), then save the edit again."
            )
        _MODEL_SQUARED_NORM = sum(float(p.detach().double().pow(2).sum()) for p in MODEL.parameters())
    return _MODEL_SQUARED_NORM


def _write_edit_files() -> None:
    edit_dir = RUN_DIR / "edit"
    tensors = {name: MODEL.get_parameter(name).detach().cpu() for name in ORIGINALS}
    total_weights = sum(parameter.numel() for parameter in MODEL.parameters())
    total_tensors = sum(1 for _ in MODEL.parameters())
    changed = sum(entry["changed_weights"] for entry in ENTRIES.values())
    delta_squared_norm = sum(entry["delta_squared_norm"] for entry in ENTRIES.values())
    layers = {int(match.group(1)) for name in ENTRIES if (match := re.search(r"layers\.(\d+)\.", name))}
    manifest = {
        "tensors_changed": len(ENTRIES),
        "tensors_changed_pct": 100 * len(ENTRIES) / total_tensors,
        "layers_changed": sorted(layers),
        "changed_weights": changed,
        "changed_weights_pct": 100 * changed / total_weights,
        # The distance of the whole edited model from the original model, as a share of the original norm.
        "relative_change_of_model": (delta_squared_norm / _model_squared_norm()) ** 0.5,
        "tensors": [ENTRIES[name] for name in sorted(ENTRIES)],
    }
    if tensors:
        save_file(tensors, str(edit_dir / "edit.safetensors"))
    else:
        (edit_dir / "edit.safetensors").unlink(missing_ok=True)
    (edit_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

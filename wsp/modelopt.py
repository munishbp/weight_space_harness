"""Load a ModelOpt mixed-precision checkpoint (NVFP4 and FP8 weights) into PyTorch.

No library loads this checkpoint format into PyTorch. This module keeps each quantized weight in
its stored format on the GPU and turns it into real numbers inside each forward pass. A 27B model
then needs about 21 GB, where full BF16 weights need 54 GB.

Stored formats (ModelOpt `NVFP4QTensor`, checked against the tensors of the checkpoint):

- NVFP4: each weight is a 4-bit E2M1 code. Two codes share one byte, the even column in the low
  half. Each block of 16 columns has one FP8 scale (`weight_scale`). The tensor has one FP32 scale
  (`weight_scale_2`). The weight is `E2M1[code] * weight_scale * weight_scale_2`.
- FP8: each weight is one FP8 E4M3 number. The tensor has one FP32 scale. The weight is their product.

Limit: the served model also rounds the input of each quantized layer to 4 or 8 bits. This module
does not. Its activations differ from the served activations by that rounding error.

An edit of a packed weight goes back into the stored format, so the edited model is still a valid
checkpoint of the same format. A change that is smaller than the step of the format rounds to zero.
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open

# The value of each 4-bit code. Bit 3 is the sign.
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_MAX, E4M3_MAX = 6.0, 448.0
BLOCK = 16
# The midpoints between neighbouring E2M1 values. `bucketize` maps a magnitude to its nearest code.
_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def _pair_table(device, dtype) -> torch.Tensor:
    """For each byte value, the two weights that it holds: the low half first, then the high half."""
    values = torch.tensor(E2M1 + [-value for value in E2M1])
    byte = torch.arange(256)
    return torch.stack([values[byte & 15], values[byte >> 4]], dim=1).to(device=device, dtype=dtype)


def decode_nvfp4(packed: torch.Tensor, block_scale: torch.Tensor, scale_2: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """The real weights of one NVFP4 tensor, with shape [rows, 2 * packed columns]."""
    values = _pair_table(packed.device, dtype)[packed.int()].flatten(1)
    scale = (block_scale.float() * scale_2.float()).to(dtype).repeat_interleave(BLOCK, dim=1)
    return values * scale


def encode_nvfp4(weight: torch.Tensor, scale_2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The NVFP4 codes and block scales that are nearest to `weight`, for a fixed tensor scale."""
    rows, columns = weight.shape
    blocks = weight.float().view(rows, columns // BLOCK, BLOCK)
    scale = blocks.abs().amax(dim=2) / (E2M1_MAX * scale_2.float())
    scale = torch.where(scale == 0, torch.ones_like(scale), scale).clamp(2.0**-9, E4M3_MAX)
    block_scale = scale.to(torch.float8_e4m3fn)
    units = blocks / (block_scale.float() * scale_2.float())[:, :, None]
    codes = torch.bucketize(units.abs(), _BOUNDS.to(weight.device)).to(torch.uint8)
    codes = (codes | ((units < 0).to(torch.uint8) << 3)).view(rows, columns)
    return codes[:, 0::2] | (codes[:, 1::2] << 4), block_scale


class PackedLinear(torch.nn.Module):
    """A linear layer with no bias whose weight stays in its stored NVFP4 or FP8 format."""

    def __init__(self, kind: str, stored: dict[str, torch.Tensor]) -> None:
        super().__init__()
        if kind not in ("nvfp4", "fp8"):
            raise ValueError(f"Unknown packed format {kind!r}.")
        self.kind = kind
        for name, tensor in stored.items():
            self.register_buffer(name, tensor)
        rows, columns = stored["weight_packed"].shape
        # One NVFP4 byte holds two weights.
        self.shape = (rows, 2 * columns if kind == "nvfp4" else columns)

    def state(self) -> dict[str, torch.Tensor]:
        """A CPU copy of the stored tensors, for `load_state`."""
        return {name: buffer.detach().cpu().clone() for name, buffer in self.named_buffers()}

    def load_state(self, state: dict[str, torch.Tensor]) -> None:
        for name, buffer in self.named_buffers():
            buffer.copy_(state[name].to(buffer.device))

    def weight_float(self, state: dict[str, torch.Tensor] | None = None, dtype=torch.float32) -> torch.Tensor:
        """The real weights on the device of the layer: the current weights, or the weights of a saved state."""
        device = self.weight_packed.device
        stored = dict(self.named_buffers()) if state is None else {name: tensor.to(device) for name, tensor in state.items()}
        if self.kind == "nvfp4":
            return decode_nvfp4(stored["weight_packed"], stored["weight_scale"], stored["weight_scale_2"], dtype)
        return stored["weight_packed"].to(dtype) * stored["weight_scale"].to(dtype)

    @property
    def weight(self) -> torch.Tensor:
        return self.weight_float()

    def set_weight(self, tensor: torch.Tensor) -> None:
        """Store `tensor` in the packed format. A block (NVFP4) or an element (FP8) that equals the current weight keeps its bits."""
        current = self.weight_float()
        tensor = tensor.to(device=current.device, dtype=torch.float32)
        if self.kind == "nvfp4":
            limit = E2M1_MAX * E4M3_MAX * float(self.weight_scale_2)
        else:
            limit = E4M3_MAX * float(self.weight_scale)
        if float(tensor.abs().max()) > limit:
            raise ValueError(
                f"The largest new weight is {float(tensor.abs().max()):.4g}, but the stored format holds at most "
                f"{limit:.4g} with the scale of this tensor. The model is unchanged. Use a smaller edit."
            )
        if self.kind == "nvfp4":
            rows, columns = tensor.shape
            changed = (tensor != current).view(rows, columns // BLOCK, BLOCK).any(dim=2)
            packed, block_scale = encode_nvfp4(tensor, self.weight_scale_2)
            self.weight_scale.copy_(torch.where(changed, block_scale, self.weight_scale))
            changed_bytes = changed.repeat_interleave(BLOCK // 2, dim=1)
            self.weight_packed.copy_(torch.where(changed_bytes, packed, self.weight_packed))
        else:
            new = (tensor / self.weight_scale.float()).to(torch.float8_e4m3fn)
            self.weight_packed.copy_(torch.where(tensor != current, new, self.weight_packed))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight_float(dtype=x.dtype))


def load(path: str, device: str = "cuda"):
    """Build the model of a ModelOpt checkpoint directory. Quantized linear layers become `PackedLinear`."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers import AutoConfig, AutoModelForImageTextToText

    root = Path(path).expanduser()
    index_file = root / "model.safetensors.index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"{index_file} does not exist. The directory is not a sharded safetensors checkpoint.")
    shard_of = json.loads(index_file.read_text())["weight_map"]
    config = AutoConfig.from_pretrained(root)
    # The loader of transformers does not know the quantization method, so the config must not name it.
    config.quantization_config = None
    with init_empty_weights():
        model = AutoModelForImageTextToText.from_config(config, dtype=torch.bfloat16)

    quantized = {name.removesuffix(".weight_scale") for name in shard_of if name.endswith(".weight_scale")}
    expected = set(model.state_dict())
    loaded, skipped = set(), []
    for shard in sorted(set(shard_of.values())):
        with safe_open(str(root / shard), framework="pt", device=device) as file:
            for name in file.keys():
                owner, _, leaf = name.rpartition(".")
                if owner in quantized:
                    # The loop handles a quantized layer one time, at its weight tensor.
                    if leaf != "weight":
                        continue
                    stored = {"weight_packed": file.get_tensor(name), "weight_scale": _get(file, shard_of, root, f"{owner}.weight_scale", device)}
                    kind = "fp8"
                    if f"{owner}.weight_scale_2" in shard_of:
                        kind = "nvfp4"
                        stored["weight_scale_2"] = _get(file, shard_of, root, f"{owner}.weight_scale_2", device)
                    parent, _, child = owner.rpartition(".")
                    setattr(model.get_submodule(parent), child, PackedLinear(kind, stored))
                    loaded.add(name)
                elif name in expected:
                    set_module_tensor_to_device(model, name, device, value=file.get_tensor(name), dtype=torch.bfloat16)
                    loaded.add(name)
                else:
                    skipped.append(name)
    missing = sorted(expected - loaded)
    if missing:
        raise RuntimeError(
            f"The checkpoint holds no tensor for {len(missing)} weights of the model, for example {missing[:3]}. "
            "The model is not usable."
        )
    model.eval()
    model.packed_load_report = {"packed_layers": len(quantized), "skipped_tensors": len(skipped), "skipped_examples": skipped[:5]}
    return model


def _get(file, shard_of: dict, root: Path, name: str, device: str) -> torch.Tensor:
    """One tensor by name. It can be in another shard than the open one."""
    if name in file.keys():
        return file.get_tensor(name)
    with safe_open(str(root / shard_of[name]), framework="pt", device=device) as other:
        return other.get_tensor(name)

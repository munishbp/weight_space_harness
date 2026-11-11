"""Direct logit attribution: how much each part of the model pushes one token up or down.

The residual stream at the last layer is the sum of the embedding and the output of every
attention block and MLP. The final norm and the output head turn that sum into logits. For an
RMS norm, a logit is linear in the residual stream up to one scale per position, so the logit
splits into one share per part.
"""

import torch

from wsp import layout, tools


def token_direction(token_id: int, against: list[int]) -> torch.Tensor:
    """The direction in the residual stream that raises `token_id` against the mean of the `against` tokens."""
    parts = layout.find(tools.MODEL)
    norm = tools.MODEL.get_submodule(parts["final_norm"])
    if "RMS" not in type(norm).__name__:
        raise NotImplementedError(
            f"The final norm is {type(norm).__name__}. This attribution handles an RMS norm only, "
            "because another norm also subtracts a mean."
        )
    head = tools.MODEL.get_output_embeddings()
    tokens = [token_id, *against]
    if hasattr(head, "weight_rows"):
        # A packed head decodes only the rows of these tokens. Its full float copy can be larger than the free memory.
        rows = torch.cat([head.weight_rows(slice(token, token + 1)) for token in tokens]).float().cpu()
    else:
        rows = head.weight.detach()[tokens].float().cpu()
    return (rows[0] - rows[1:].mean(dim=0)) * norm.weight.detach().float().cpu()


def _writers(parts: dict) -> dict[str, str]:
    """Block name to the module whose output joins the residual stream: the block, or the norm after it."""
    blocks = parts["mlps"] + parts["attentions"]
    norms = parts["mlp_write_norms"] + parts["attention_write_norms"]
    return {block: norm or block for block, norm in zip(blocks, norms)} | {parts["embedding"]: parts["embedding"]}


def attribute(sequences: list[list[int]], positions: list[list[int]], direction: torch.Tensor) -> dict:
    """One read pass: the share of each part in the logit difference along `direction`, at each position.

    The result holds `shares` (part name to a tensor with one value per position, all sequences
    joined), `total` (the full logit difference per position), and `error` (the relative gap
    between the sum of the parts and the final residual stream; a large value means that the
    split is not valid for this model).
    """
    parts = layout.find(tools.MODEL)
    last = parts["layers"][-1]
    writers = _writers(parts)
    outputs = tools.read_tokens(sequences, [*writers.values(), last], positions=positions)
    final = torch.cat(outputs[last])
    scale = final.pow(2).mean(dim=1).sqrt()
    joined = {block: torch.cat(outputs[module]) for block, module in writers.items()}
    error = float((sum(joined.values()) - final).norm() / final.norm())
    return {
        "shares": {name: (value @ direction) / scale for name, value in joined.items()},
        "total": (final @ direction) / scale,
        "error": error,
    }


def attribute_last(
    conversations: list[list[dict]],
    direction: torch.Tensor,
    thinking: bool = False,
    append_ids: list[list[int]] | None = None,
) -> dict:
    """The same split as `attribute`, at the last position of each conversation. Conversations can hold images.

    `direction` is one direction for all conversations, or one row per conversation.
    """
    parts = layout.find(tools.MODEL)
    last = parts["layers"][-1]
    writers = _writers(parts)
    outputs = tools.read(conversations, [*writers.values(), last], thinking=thinking, append_ids=append_ids)
    final = outputs[last]
    scale = final.pow(2).mean(dim=1).sqrt()
    joined = {block: outputs[module] for block, module in writers.items()}
    error = float((sum(joined.values()) - final).norm() / final.norm())
    along = (lambda value: (value * direction).sum(dim=1)) if direction.ndim == 2 else (lambda value: value @ direction)
    return {
        "shares": {name: along(value) / scale for name, value in joined.items()},
        "total": along(final) / scale,
        "error": error,
    }

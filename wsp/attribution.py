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
    head = tools.MODEL.get_output_embeddings().weight.detach().float().cpu()
    return (head[token_id] - head[against].mean(dim=0)) * norm.weight.detach().float().cpu()


def attribute(sequences: list[list[int]], positions: list[list[int]], direction: torch.Tensor) -> dict:
    """One read pass: the share of each part in the logit difference along `direction`, at each position.

    The result holds `shares` (part name to a tensor with one value per position, all sequences
    joined), `total` (the full logit difference per position), and `error` (the relative gap
    between the sum of the parts and the final residual stream; a large value means that the
    split is not valid for this model).
    """
    parts = layout.find(tools.MODEL)
    last = parts["layers"][-1]
    modules = parts["mlps"] + parts["attentions"] + [last, parts["embedding"]]
    outputs = tools.read_tokens(sequences, modules, positions=positions)
    final = torch.cat(outputs[last])
    scale = final.pow(2).mean(dim=1).sqrt()
    joined = {name: torch.cat(outputs[name]) for name in modules if name != last}
    error = float((sum(joined.values()) - final).norm() / final.norm())
    return {
        "shares": {name: (value @ direction) / scale for name, value in joined.items()},
        "total": (final @ direction) / scale,
        "error": error,
    }

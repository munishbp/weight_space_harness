"""The habit recipe: at one step of a reply, in one context, raise the correct token over the token of a habit.

The recipe changes one MLP output matrix with one least-squares solve (`edits.fit_keys_to_values`).
It needs four inputs:

- Target steps: prompts where the student writes the habit token and the correct token has a
  visible probability. The fit maps the MLP input at that step to a shift toward the correct token.
- Keep steps: the same step of other contexts. The fit maps them to no change.
- A protection matrix: the second moment of the MLP input at every position of prompts and replies,
  and on neutral text. Without it, the change writes the direction at image and template positions,
  and the later layers read it there.
- A margin: the size of the logit shift. Sweep it, because later layers undo a part of the shift.

Every function reads the student through `wsp.tools`, so the ledger counts the forward passes.
"""

import torch

from wsp import edits, layout, tools

# A prompt for the protection: the conversation, the token ids of the forced reply, and the index
# of one reply position to leave out (the target step), or None.
Prompt = tuple[list[dict], list[int], int | None]


def step_keys(
    conversations: list[list[dict]], mlp_out: str, last_layer: str, prefix: list[int], batch: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    """The MLP input at the step after `prefix`, and the scale of the final residual at that step.

    `prefix` holds the token ids of the reply up to the step, so the step is the last position.
    The result is (keys of shape [n, neurons], scale of shape [n]).
    """
    keys, scales = [], []
    for start in range(0, len(conversations), batch):
        part = conversations[start : start + batch]
        forced = [prefix] * len(part)
        scales.append(tools.read(part, [last_layer], append_ids=forced)[last_layer].pow(2).mean(1).sqrt())
        keys.append(tools.read(part, [mlp_out], append_ids=forced, inputs=True)[mlp_out])
    return torch.cat(keys), torch.cat(scales)


def all_position_keys(conversation: list[dict], reply_ids: list[int], mlp_out: str) -> torch.Tensor:
    """The MLP input at every position of one prompt followed by `reply_ids`, of shape [positions, neurons]."""
    # One prompt per call has no padding, so every returned position is a real position.
    return tools.read([conversation], [mlp_out], append_ids=[reply_ids], last_positions=1_000_000, inputs=True)[mlp_out][0]


def protection(prompts: list[Prompt], neutral: list[list[int]], mlp_out: str) -> torch.Tensor:
    """The protection matrix: the mean of k k^T over all positions of `prompts`, plus the same on neutral text.

    `neutral` holds token sequences of unrelated text (`battery.neutral_text`). The two parts have
    equal weight. The result is a float64 CPU tensor of shape [neurons, neurons].
    """
    if not prompts:
        raise ValueError("The protection needs at least one prompt. No forward pass ran.")
    device = tools.MODEL.device
    # One accumulator, changed in place: for a wide MLP the matrix takes gigabytes, and a second copy may not fit.
    total, count = None, 0
    for conversation, reply_ids, skip in prompts:
        keys = all_position_keys(conversation, reply_ids, mlp_out).to(device).double()
        if skip is not None:
            # `skip` counts from the start of the reply. The prompt positions come before the reply.
            index = len(keys) - len(reply_ids) + skip
            keys = torch.cat([keys[:index], keys[index + 1 :]])
        if total is None:
            total = torch.zeros(keys.shape[1], keys.shape[1], dtype=torch.float64, device=device)
        total.addmm_(keys.T, keys)
        count += len(keys)
    total.div_(count)
    positions = sum(len(sequence) for sequence in neutral)
    # The text goes in small batches. One batch of all sequences does not fit next to a large student.
    for start in range(0, len(neutral), 8):
        for part in tools.read_tokens(neutral[start : start + 8], [mlp_out], inputs=True)[mlp_out]:
            part = part.to(device).double()
            total.addmm_(part.T, part, alpha=1.0 / positions)
    return total.cpu()


def fit(
    mlp_out: str,
    targets: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    keep: torch.Tensor | None,
    protect: torch.Tensor,
    margin: float,
    strength: float = 1.0,
) -> dict:
    """Write the edit. Each target is (keys, scale, direction): its keys shift along its direction by `margin`.

    `direction` comes from `attribution.token_direction(correct, [habit])`. `keep` holds keys that
    must map to no change. The result is the entry of `tools.save_edit`.
    """
    keys = [group[0] for group in targets]
    values = [edits.logit_shift_values(direction, scale, margin) for _, scale, direction in targets]
    parts = layout.find(tools.MODEL)
    if mlp_out not in parts["mlp_outputs"]:
        raise ValueError(f"{mlp_out} is not an MLP output matrix. Take a name from layout.find(model)['mlp_outputs'].")
    norm = parts["mlp_write_norms"][parts["mlp_outputs"].index(mlp_out)]
    if norm is not None:
        # A norm sits between this matrix and the residual stream (GLM-4). The values must be the change before it.
        weight = tools.get_weight(mlp_out + ".weight")
        gain = tools.MODEL.get_submodule(norm).weight.detach().float().cpu()
        values = [edits.values_before_norm(group.float().cpu() @ weight.T, value.float().cpu(), gain) for group, value in zip(keys, values)]
    if keep is not None and len(keep):
        keys.append(keep)
        values.append(torch.zeros(len(keep), values[0].shape[1], dtype=values[0].dtype, device=values[0].device))
    return edits.fit_keys_to_values(mlp_out + ".weight", torch.cat(keys), torch.cat(values), protect, strength)


def stray_write(
    conversation: list[dict], reply_ids: list[int], mlp_out: str, original: torch.Tensor, direction: torch.Tensor
) -> dict:
    """How much of `direction` the current edit adds at each position of one prompt.

    `original` is the weight before the edit (`tools.get_weight` after `tools.reset_edit`).
    Run this on a prompt of another context before a trial. A large `sum_abs` next to the intended
    write at a target step shows that the protection misses positions.
    """
    device = tools.MODEL.device
    keys = all_position_keys(conversation, reply_ids, mlp_out).to(device).double()
    delta = tools.get_weight(mlp_out + ".weight").to(device).double() - original.to(device).double()
    unit = direction.to(device).double()
    unit = unit / unit.norm()
    amount = (keys @ delta.T) @ unit
    return {"positions": len(amount), "sum_abs": float(amount.abs().sum()), "max_abs": float(amount.abs().max()), "last": float(amount[-1])}

"""Closed-form edits. Each one changes weights through `tools.save_edit` and uses no gradient.

A weight matrix of a linear layer has one row per output unit and one column per input unit.
For a matrix that writes into the residual stream, column `j` is the vector that input unit `j`
(a neuron, for an MLP output matrix) adds to the residual stream.
"""

import torch

from wsp import tools


def unit(direction: torch.Tensor) -> torch.Tensor:
    return direction / direction.norm()


def remove_direction(names: list[str], direction: torch.Tensor, strength: float = 1.0) -> list[dict]:
    """Stop the named matrices from writing along `direction`: W' = W - strength * u (u^T W).

    A strength of 1 removes the component. A strength above 1 reverses it.
    Removal sets the component to zero. It helps only when the normal state is near zero along
    the direction and the unwanted state is far from zero. Check that before the edit.
    """
    u = unit(direction)
    entries = []
    for name in names:
        weight = tools.get_weight(name)
        entries.append(tools.save_edit(name, weight - strength * torch.outer(u, u @ weight)))
    return entries


def remove_direction_from_neurons(
    name: str, neurons: torch.Tensor, direction: torch.Tensor, strength: float = 1.0
) -> dict:
    """The same removal, for the chosen columns of one matrix only."""
    u = unit(direction)
    weight = tools.get_weight(name)
    weight[:, neurons] -= strength * torch.outer(u, u @ weight[:, neurons])
    return tools.save_edit(name, weight)


def add_to_neurons(name: str, neurons: torch.Tensor, direction: torch.Tensor, amounts: torch.Tensor) -> dict:
    """Add a connection: neuron `neurons[k]` also writes `amounts[k] * u` per unit of its activation."""
    weight = tools.get_weight(name)
    weight[:, neurons] += torch.outer(unit(direction), amounts)
    return tools.save_edit(name, weight)


def logit_shift_values(direction: torch.Tensor, scales: torch.Tensor, margin: float) -> torch.Tensor:
    """Values for `fit_keys_to_values` that raise the logit gap along `direction` by about `margin`.

    `direction` comes from `attribution.token_direction`: one direction for all keys, or one row
    per key. `scales` holds the RMS of the final residual stream at each key position. Later
    layers undo a part of the shift, so the gap that results is smaller than `margin`.
    """
    rows = direction if direction.ndim == 2 else direction[None, :].expand(len(scales), -1)
    return margin * scales[:, None] * rows / rows.pow(2).sum(dim=1, keepdim=True)


def fit_keys_to_values(
    name: str, keys: torch.Tensor, values: torch.Tensor, protect: torch.Tensor, strength: float = 1.0
) -> dict:
    """Solve one least-squares problem for a weight change that maps each key to its value.

    `keys` has one row per example: the input of the matrix `name` (for an MLP output matrix, the
    neuron activations). `values` has one row per example: the vector to add to the output of the
    matrix for that key. `protect` is the second moment of the same input on unrelated data,
    `mean(k k^T)`. The change is

        delta = values^T keys (strength * n * protect + keys^T keys)^-1

    A larger `strength` keeps the output on unrelated data closer to the original and fits the
    examples less. This is the closed form of the MEMIT update (Meng et al., 2022). It uses no
    gradient and no iteration.
    """
    keys, values, protect = keys.double(), values.double(), protect.double()
    system = strength * len(keys) * protect + keys.T @ keys
    delta = torch.linalg.solve(system, keys.T @ values).T
    return tools.save_edit(name, tools.get_weight(name) + delta.float())

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

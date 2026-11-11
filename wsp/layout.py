"""Find the parts of a student model by name, so the other modules do not assume one model family.

The search uses the naming conventions of Hugging Face decoder models. A model with other names
needs a new entry in the option lists below.
"""

import re

import torch

MLP_NAMES = ("mlp", "feed_forward", "ffn")
ATTENTION_NAMES = ("self_attn", "linear_attn", "attn", "attention")
MLP_OUTPUT_NAMES = ("down_proj", "fc2", "c_proj")
ATTENTION_OUTPUT_NAMES = ("o_proj", "out_proj", "c_proj")
FINAL_NORM_NAMES = ("norm", "ln_f", "final_layer_norm")


def find(model: torch.nn.Module) -> dict:
    """The module names of the language model: layers, MLPs, attention blocks, and their output weights.

    Every value is a name for `model.get_submodule` or `model.get_parameter`. The lists run from the
    first layer to the last layer.
    """
    named = dict(model.named_modules())
    stacks = [
        (name, module)
        for name, module in named.items()
        if isinstance(module, torch.nn.ModuleList) and not re.search("visual|vision", name)
    ]
    if not stacks:
        raise LookupError("The model has no stack of layers outside its vision tower.")
    # The language model is the deepest stack that is not part of the vision tower.
    stack_name, stack = max(stacks, key=lambda pair: len(pair[1]))
    layers = [f"{stack_name}.{index}" for index in range(len(stack))]

    def child(parent: str, options: tuple[str, ...]) -> str:
        for option in options:
            if f"{parent}.{option}" in named:
                return f"{parent}.{option}"
        raise LookupError(f"The module {parent} has no child named one of {options}. Add its name to wsp/layout.py.")

    mlps = [child(layer, MLP_NAMES) for layer in layers]
    attentions = [child(layer, ATTENTION_NAMES) for layer in layers]
    embedding = next(name for name, module in named.items() if module is model.get_input_embeddings())
    return {
        "layers": layers,
        "mlps": mlps,
        "attentions": attentions,
        "mlp_outputs": [child(mlp, MLP_OUTPUT_NAMES) for mlp in mlps],
        "attention_outputs": [child(attention, ATTENTION_OUTPUT_NAMES) for attention in attentions],
        "final_norm": child(stack_name.rsplit(".", 1)[0], FINAL_NORM_NAMES),
        "embedding": embedding,
    }

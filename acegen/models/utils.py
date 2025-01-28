import warnings
from typing import Union

import torch
from tensordict.nn import TensorDictModule


def adapt_state_dict(source_state_dict: dict, target_state_dict: dict):
    """Adapt the source state dict to the target state dict.

    This is useful when loading a model checkpoint from a different model.
    It will only work if the source and target models have the same number of parameters and the
    same parameter shapes.

    Args:
        source_state_dict (dict): The source state dict.
        target_state_dict (dict): The target state dict.
    """
    if len(source_state_dict) != len(target_state_dict):
        raise ValueError(
            "The source and target state dicts don't have the same number of parameters."
        )

    for key_source, value_source, key_target, value_target in zip(
        source_state_dict.keys(),
        source_state_dict.values(),
        target_state_dict.keys(),
        target_state_dict.values(),
    ):
        if value_source.shape != value_target.shape:
            warnings.warn(
                f"The shape of source key {key_source} ({value_source.shape}) "
                f"and target key {key_target} ({value_target.shape}) do not match."
            )
            continue
        target_state_dict[key_target] = value_source

    return target_state_dict


def reinitialize_model(
    model: Union[torch.nn.Module, TensorDictModule], seed: int = 101
):
    """Random initialization of a models parameters."""
    torch.manual_seed(seed)
    for p in model.parameters():
        if len(p.shape) == 1:
            torch.nn.init.constant_(p, 0)
        else:
            torch.nn.init.uniform_(p)

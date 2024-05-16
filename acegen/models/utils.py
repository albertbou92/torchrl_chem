import warnings

from torchrl.envs import Compose


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


def get_primers_from_module(module):
    """Get all tensordict primers from all submodules of a module."""
    primers = []

    def make_primers(submodule):
        if hasattr(submodule, "make_tensordict_primer"):
            primers.append(submodule.make_tensordict_primer())

    module.apply(make_primers)
    if not primers:
        import warnings

        raise warnings.warn("No primers found in the module.")
    elif len(primers) == 1:
        return primers[0]
    else:
        return Compose(primers)

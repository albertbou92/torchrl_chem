import torch
from tensodict import TensorDictBase
from torchrl.envs import Transform


class BurnInTransform(Transform):
    """Transform to burn in the recurrent state of an RNN.

    This transform is useful for obtaining up-to-date recurrent states when
    they are not available by burning in a few steps along the time dimension.
    It is intended to be used as a replay buffer transform, not as an environment
    transform.

    Args:
        modules (list): A list of modules to burn in.
        burn_in (int): The number of time steps to burn in.

    Examples:
        >>> import torch
        >>> from torchrl.envs import TensorDict
        >>> from torchrl.envs.transforms import BurnInTransform
        >>> from torchrl.modules import LSTM

        >>> burn_in_transform = BurnInTransform(
        ...     modules=[LSTM(1, 1, batch_first=True)],
        ...     burn_in=5,
        ... )

    """

    def __init__(self, modules, burn_in):
        super().__init__()
        self.modules = modules
        self.burn_in = burn_in

    def __call__(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise RuntimeError(
            "BurnInTransform can only be used when appended to a ReplayBuffer."
        )

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        raise RuntimeError(
            "BurnInTransform can only be used when appended to a ReplayBuffer."
        )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:

        td_device = tensordict.device or "cpu"

        # Split the tensor dict into the burn in and the rest.
        td_burn_in = tensordict[..., : self.burn_in]
        td_out = tensordict[..., self.burn_in :]

        # Burn in the recurrent state.
        with torch.no_grad():
            for module in self.modules:
                td_burn_in = td_burn_in.to(module.device)
                td_burn_in = module(td_burn_in)
        td_burn_in = td_burn_in.to(td_device)

        # Update the next state.
        td_out[..., 0].update(td_burn_in["next"][..., -1])
        return td_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(burn_in={self.burn_in})"

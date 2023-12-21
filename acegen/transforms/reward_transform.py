import warnings
from typing import Callable

import torch
from tensordict import TensorDictBase
from torchrl.envs.transforms.transforms import Transform

from acegen.vocabulary.base import Vocabulary


class SMILESReward(Transform):
    """Transform to add a reward to a SMILES.

    Args:
        reward_function: A callable that takes a list of SMILES and returns a list of rewards.
        vocabulary: A vocabulary object with at least encode and decode methods.
        in_keys: The key in the tensordict that contains the encoded SMILES.
        out_keys: The key in the tensordict to store the reward.
        reward_scale: The scale to apply to the reward.
    """

    def __init__(
        self,
        reward_function: Callable,
        vocabulary: Vocabulary,
        in_keys=None,
        out_keys=None,
        reward_scale=1.0,
    ):
        if not isinstance(reward_function, Callable):
            raise ValueError("reward_function must be a callable.")

        if out_keys is None:
            out_keys = [("next", "reward")]
        if in_keys is None:
            in_keys = [("next", "SMILES")]
        self.reward_scale = reward_scale

        super().__init__(in_keys, out_keys)

        self.vocabulary = vocabulary
        self.reward_function = reward_function

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:

        # Get steps where trajectories end
        device = tensordict.device
        done = tensordict.get(("next", "done")).squeeze(-1)

        if len(done) == 0:
            return tensordict

        sub_tensordict = tensordict.get_sub_tensordict(done)

        # Get reward and smiles
        reward = sub_tensordict.get(self.out_keys[0])
        smiles = sub_tensordict.get(self.in_keys[0])

        # Get smiles as strings
        smiles_list = []
        for smi in smiles:
            smiles_list.append(
                self.vocabulary.decode(smi.cpu().numpy(), ignore_indices=[-1])
            )

        # Calculate reward - try multiple times in case of RuntimeError
        max_attempts = 3
        for i in range(max_attempts):
            try:
                reward[:, 0] += torch.tensor(
                    self.reward_function(smiles_list), device=device
                )
                break
            except RuntimeError:
                if i == max_attempts - 1:
                    raise
                else:
                    warnings.warn(
                        "RuntimeError in reward function. Trying again. Attempt {}/{}".format(
                            i + 1, max_attempts
                        )
                    )
                    continue

        sub_tensordict.set(self.out_keys[0], reward * self.reward_scale, inplace=True)

        return tensordict

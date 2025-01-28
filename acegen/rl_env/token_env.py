from typing import Optional

import torch
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import (
    Composite,
    Categorical,
    OneHotDiscreteTensorSpec,
    Unbounded,
)
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs import EnvBase


class TokenEnv(EnvBase):
    """Reinforcement learning environment for token-based generation.

    Given a start token, end token, and length of vocabulary, this environment generates token-based
    sequences step-by-step, one token at a time. The environment terminates when the end token is
    provided as an action. The environment also terminates if the maximum length of the episodes is
    reached.

    Args:
        start_token (int): Start token for an episode.
        end_token (int): End token for an episode.
        length_vocabulary (int): Length of vocabulary.
        max_length (int, optional): Maximum length of an episode. Defaults to 100.
        device (DEVICE_TYPING, optional): Device to use. Defaults to None.
        batch_size (int, optional): number of episodes to generate in parallel. Defaults to 1.
        one_hot_action_encoding (bool, optional): Whether to use one-hot encoding for actions. Defaults to False.
        one_hot_obs_encoding (bool, optional): Whether to use one-hot encoding for observations. Defaults to False.

    Examples:
        >>> from acegen import TokenEnv
        >>> rl_env = TokenEnv(
        ...     start_token=0,
        ...     end_token=1,
        ...     length_vocabulary=2,
        ...     max_length=10,
        ...     batch_size=2,
        ... )
        >>> obs = rl_env.reset()
    """

    def __init__(
        self,
        start_token: int,
        end_token: int,
        length_vocabulary: int,
        max_length: int = 100,
        device: DEVICE_TYPING = None,
        batch_size: int = 1,
        one_hot_action_encoding: bool = False,
        one_hot_obs_encoding: bool = False,
    ):
        super().__init__(
            device=device,
            batch_size=torch.Size([batch_size]),
        )

        self.num_envs = batch_size
        self.max_length = max_length
        self.end_token = int(end_token)
        self.start_token = int(start_token)
        self.length_vocabulary = length_vocabulary
        self.one_hot_obs_encoding = one_hot_obs_encoding
        self.one_hot_action_encoding = one_hot_action_encoding
        self.episode_length = torch.ones(
            self.num_envs, device=self.device, dtype=torch.int32
        )

        if self.one_hot_obs_encoding:
            start_obs = torch.zeros(
                batch_size, length_vocabulary, device=self.device, dtype=torch.int32
            )
            start_obs[:, self.start_token] = 1
            self.sequence = torch.zeros(
                self.num_envs,
                self.max_length,
                length_vocabulary,
                device=self.device,
                dtype=torch.int32,
            )
            self.sequence[:, 0, self.start_token] = 1
        else:
            start_obs = (
                torch.ones(self.num_envs, device=self.device, dtype=torch.int32)
                * self.start_token
            )
            self.sequence = torch.zeros(
                self.num_envs, self.max_length, device=self.device, dtype=torch.int32
            )
            self.sequence[:, 0] = self.start_token

        self.sequence_mask = torch.zeros(
            self.num_envs, self.max_length, device=self.device, dtype=torch.bool
        )
        self.sequence_mask[:, 0] = True

        self._reset_tensordict = TensorDict(
            {
                "observation": start_obs,
                "done": torch.zeros(
                    self.num_envs, 1, device=self.device, dtype=torch.bool
                ),
                "truncated": torch.zeros(
                    self.num_envs, 1, device=self.device, dtype=torch.bool
                ),
                "terminated": torch.zeros(
                    self.num_envs, 1, device=self.device, dtype=torch.bool
                ),
                "sequence": self.sequence.clone(),
                "sequence_mask": self.sequence_mask.clone(),
            },
            device=self.device,
            batch_size=self.batch_size,
        )

        self._set_specs()

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        if tensordict is not None:
            next_tensordict = tensordict
            next_tensordict.update(self._reset_tensordict.clone())
            _reset = tensordict.get(
                "_reset",
                torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
            )
        else:
            next_tensordict = self._reset_tensordict.clone()
            _reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self.episode_length[_reset] = 1
        self.sequence[_reset] = next_tensordict["sequence"][_reset]
        self.sequence_mask[_reset] = next_tensordict["sequence_mask"][_reset]

        return next_tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Get actions
        actions = tensordict.get("action")
        if self.one_hot_action_encoding:
            actions = torch.argmax(actions, dim=-1)

        # Update episode length
        self.episode_length += 1

        # Create termination flags
        terminated = (actions == self.end_token).unsqueeze(-1)
        truncated = (self.episode_length == self.max_length).unsqueeze(-1)
        done = terminated | truncated

        # Create next_tensordict
        obs = actions.clone().long()
        if self.one_hot_obs_encoding:
            obs = torch.nn.functional.one_hot(obs, num_classes=self.length_vocabulary)

        # Update sequence
        no_done = ~done.squeeze(-1)
        self.sequence[no_done, self.episode_length[no_done] - 1] = obs[no_done].int()
        self.sequence_mask[no_done, self.episode_length[no_done] - 1] = True

        next_tensordict = TensorDict(
            {
                "done": done,
                "truncated": truncated,
                "terminated": terminated,
                "reward": torch.zeros(self.num_envs, device=self.device),
                "observation": obs,
                "sequence": self.sequence.clone(),
                "sequence_mask": self.sequence_mask.clone(),
            },
            device=self.device,
            batch_size=self.batch_size,
        )
        next_tensordict.update(tensordict.get("next", {}))
        return next_tensordict

    def _set_seed(self, seed: Optional[int] = -1) -> None:
        torch.manual_seed(seed)

    def _set_specs(self) -> None:
        obs_spec = (
            OneHotDiscreteTensorSpec
            if self.one_hot_obs_encoding
            else Categorical
        )
        self.observation_spec = Composite(
            {
                "observation": obs_spec(
                    n=self.length_vocabulary,
                    shape=(
                        torch.Size([self.max_length, self.length_vocabulary])
                        if self.one_hot_obs_encoding
                        else torch.Size([self.max_length])
                    ),
                    dtype=torch.int32,
                    device=self.device,
                ),
                "sequence": obs_spec(
                    n=self.length_vocabulary,
                    shape=(
                        torch.Size([self.max_length, self.length_vocabulary])
                        if self.one_hot_obs_encoding
                        else torch.Size([self.max_length])
                    ),
                    dtype=torch.int32,
                    device=self.device,
                ),
                "sequence_mask": obs_spec(
                    n=2,
                    dtype=torch.bool,
                    shape=(
                        torch.Size([self.max_length, 2])
                        if self.one_hot_obs_encoding
                        else torch.Size([self.max_length])
                    ),
                    device=self.device,
                ),
            }
        ).expand(self.num_envs)
        action_spec = (
            OneHotDiscreteTensorSpec
            if self.one_hot_action_encoding
            else Categorical
        )
        self.action_spec = Composite(
            {
                "action": action_spec(
                    n=self.length_vocabulary,
                    dtype=torch.int32,
                    device=self.device,
                )
            }
        ).expand(self.num_envs)
        self.reward_spec = Composite(
            {
                "reward": Unbounded(
                    shape=(1,),
                    dtype=torch.float32,
                    device=self.device,
                )
            }
        ).expand(self.num_envs)

        self.done_spec = (
            Composite(
                {
                    "done": Categorical(
                        n=2, dtype=torch.bool, device=self.device
                    ),
                    "truncated": Categorical(
                        n=2, dtype=torch.bool, device=self.device
                    ),
                    "terminated": Categorical(
                        n=2, dtype=torch.bool, device=self.device
                    ),
                }
            )
            .expand(self.num_envs)
            .unsqueeze(-1)
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}, start_token={self.start_token},"
            f" end_token={self.end_token} batch_size={self.batch_size})"
        )

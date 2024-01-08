import numpy as np
import torch
from tensordict import TensorDict

from acegen.vocabulary.base import Vocabulary


class Experience(object):
    """Class for prioritized experience replay.

    This class remembers the highest scored sequences seen and samples
    from them with probabilities relative to their scores.
    """

    def __init__(self, vocabulary: Vocabulary, max_size: int = 100):
        self.memory = []
        self.max_size = max_size
        self.voc = vocabulary

    def add_experience(self, experience):
        """Experience should be a list of (smiles, score, prior likelihood) tuples."""
        self.memory.extend(experience)
        if len(self.memory) > self.max_size:
            # Remove duplicates
            idxs, smiles = [], []
            for i, exp in enumerate(self.memory):
                if exp[0] not in smiles:
                    idxs.append(i)
                    smiles.append(exp[0])
            self.memory = [self.memory[idx] for idx in idxs]
            self.memory.sort(key=lambda x: x[1], reverse=True)
            self.memory = self.memory[: self.max_size]

    def sample_smiles(self, n, decode_smiles=False):
        """Sample a batch size n of experience."""
        if len(self.memory) < n:
            raise IndexError(
                "Size of memory ({}) is less than requested sample ({})".format(
                    len(self), n
                )
            )
        else:
            scores = [x[1].item() + 1e-10 for x in self.memory]
            sample = np.random.choice(
                len(self), size=n, replace=False, p=scores / np.sum(scores)
            )
            sample = [self.memory[i] for i in sample]
            smiles = [x[0] for x in sample]
            scores = [x[1] for x in sample]
            prior_likelihood = [x[2] for x in sample]
        if decode_smiles:
            encoded = [
                torch.tensor(self.voc.encode(smile), dtype=torch.int32)
                for smile in smiles
            ]
            smiles = collate_fn(encoded)
        return smiles, torch.tensor(scores), torch.tensor(prior_likelihood)

    def sample_replay_batch(self, batch_size, device="cpu"):
        """Create a TensorDict data batch from replay data."""
        replay_smiles, replay_rewards, _ = self.sample_smiles(
            batch_size, decode_smiles=False
        )

        td_list = []

        for smiles, rew in zip(replay_smiles, replay_rewards):

            encoded = self.voc.encode(smiles)
            smiles = torch.tensor(encoded, dtype=torch.int32, device=device)

            observation = smiles[:-1].reshape(1, -1, 1).clone()
            action = smiles[1:].reshape(1, -1).clone()
            tensor_shape = (1, observation.shape[1], 1)
            reward = torch.zeros(tensor_shape, device=device)
            reward[0, -1] = rew
            done = torch.zeros(tensor_shape, device=device, dtype=torch.bool)
            is_init = torch.zeros(tensor_shape, device=device, dtype=torch.bool)
            is_init[0, 0] = True

            next_observation = smiles[1:].reshape(1, -1, 1).clone()
            next_done = torch.zeros(tensor_shape, device=device, dtype=torch.bool)
            next_done[0, -1] = True

            td_list.append(
                TensorDict(
                    {
                        "done": done,
                        "action": action,
                        "terminated": done.clone(),
                        "observation": observation,
                        "next": TensorDict(
                            {
                                "observation": next_observation,
                                "terminated": next_done.clone(),
                                "reward": reward,
                                "done": next_done,
                            },
                            batch_size=tensor_shape[0:2],
                            device=device,
                        ),
                    },
                    batch_size=tensor_shape[0:2],
                    device=device,
                )
            )

        cat_data = torch.cat(td_list, dim=-1)
        return cat_data

    def __len__(self):
        return len(self.memory)


def collate_fn(arr):
    """Function to take a list of encoded sequences and turn them into a batch."""
    max_length = max([seq.size(0) for seq in arr])
    collated_arr = torch.zeros(len(arr), max_length)
    for i, seq in enumerate(arr):
        collated_arr[i, : seq.size(0)] = seq
    return collated_arr


# 1. generate a batch of padded sequences

# 2. I can always unbind them, apply mask, and then rebind them

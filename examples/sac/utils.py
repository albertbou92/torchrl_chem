#!/usr/bin/env python3

import copy
from pathlib import Path
import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import ExplorationType
from torchrl.modules import (
    GRUModule,
    LSTMModule,
    MLP,
    ActorValueOperator,
    ProbabilisticActor,
    QValueActor,
)
from torchrl.modules.distributions import OneHotCategorical
from torchrl.envs import ExplorationType, TensorDictPrimer
from torchrl.data.tensor_specs import UnboundedContinuousTensorSpec
from torchrl.data import DiscreteTensorSpec


class Embed(torch.nn.Module):
    """Implements a simple embedding layer."""

    def __init__(self, input_size, embedding_size):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self._embedding = torch.nn.Embedding(input_size, embedding_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        *batch, L = inputs.shape
        if len(batch) > 1:
            inputs = inputs.flatten(0, len(batch) - 1)
        out = self._embedding(inputs)
        if len(batch) > 1:
            out = out.unflatten(0, batch)
        out = out.squeeze(
            -1
        )  # If time dimension is 1, remove it. Ugly hack, should not be necessary
        out = out.squeeze(
            -2
        )  # If time dimension is 1, remove it. Ugly hack, should not be necessary
        return out


def create_net(vocabulary_size, batch_size, net_name="actor"):
    embedding_module = TensorDictModule(
        Embed(vocabulary_size, 128),
        in_keys=["observation"],
        out_keys=["embed"],
    )

    gru_module = GRUModule(
        dropout=0.0,
        input_size=128,
        hidden_size=512,
        num_layers=3,
        in_keys=["embed", f"recurrent_state_{net_name}", "is_init"],
        out_keys=[
            "features",
            ("next", f"recurrent_state_{net_name}"),
        ],
        python_based=True,
    )
    mlp = TensorDictModule(
        MLP(
            in_features=512,
            out_features=vocabulary_size,
            num_cells=[],
        ),
        in_keys=["features"],
        out_keys=["logits"] if net_name == "actor" else ["action_value"],
    )

    model_inference = TensorDictSequential(embedding_module, gru_module, mlp)
    model_training = TensorDictSequential(embedding_module, gru_module.set_recurrent_mode(True), mlp)

    if net_name == "actor":
        model_inference = ProbabilisticActor(
            module=model_inference,
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=torch.distributions.Categorical,
            return_log_prob=True,
            default_interaction_type=ExplorationType.RANDOM,
        )
        model_training = ProbabilisticActor(
            module=model_training,
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=torch.distributions.Categorical,
            return_log_prob=True,
            default_interaction_type=ExplorationType.RANDOM,
        )
    else:
        model_inference = QValueActor(
            module=model_inference,
            in_keys=["action_value"],
            spec=DiscreteTensorSpec(vocabulary_size),
            action_space="categorical",
        )
        model_training = QValueActor(
            module=model_training,
            in_keys=["action_value"],
            spec=DiscreteTensorSpec(vocabulary_size),
            action_space="categorical",
        )

    primers = {
        (f"recurrent_state_{net_name}",):
            UnboundedContinuousTensorSpec(
                shape=torch.Size([batch_size, 3, 512]),
                dtype=torch.float32,
            ),
    }
    transform = TensorDictPrimer(primers)

    return model_inference, model_training, transform


def create_sac_models(vocabulary_size, batch_size):

    actor_inference, actor_training, actor_transform = create_net(vocabulary_size, batch_size, net_name="actor")
    # ckpt = torch.load(Path(__file__).resolve().parent / "priors" / "chembl_actor.prior")
    # ckpt = adapt_sac_ckpt(ckpt)
    # actor_inference.load_state_dict(ckpt)
    # actor_training.load_state_dict(ckpt)

    critic_inference, critic_training, critic_transform = create_net(vocabulary_size, batch_size, net_name="critic")
    # ckpt = torch.load(Path(__file__).resolve().parent / "priors" / "chembl_actor.prior")
    # ckpt = adapt_sac_ckpt(ckpt)
    # critic_inference.load_state_dict(ckpt)
    # critic_training.load_state_dict(ckpt)

    return actor_inference, actor_training, critic_inference, critic_training # , actor_transform, critic_transform


def adapt_sac_ckpt(ckpt):
    """Adapt the PPO ckpt from the AceGen ckpt format."""

    keys_mapping = {
        'module.0.module._embedding.weight': "module.0.module.0.module._embedding.weight",
        'module.1.lstm.weight_ih_l0': "module.0.module.1.lstm.weight_ih_l0",
        'module.1.lstm.weight_hh_l0': "module.0.module.1.lstm.weight_hh_l0",
        'module.1.lstm.bias_ih_l0': "module.0.module.1.lstm.bias_ih_l0",
        'module.1.lstm.bias_hh_l0': "module.0.module.1.lstm.bias_hh_l0",
        'module.1.lstm.weight_ih_l1': "module.0.module.1.lstm.weight_ih_l1",
        'module.1.lstm.weight_hh_l1': "module.0.module.1.lstm.weight_hh_l1",
        'module.1.lstm.bias_ih_l1': "module.0.module.1.lstm.bias_ih_l1",
        'module.1.lstm.bias_hh_l1': "module.0.module.1.lstm.bias_hh_l1",
        'module.1.lstm.weight_ih_l2': "module.0.module.1.lstm.weight_ih_l2",
        'module.1.lstm.weight_hh_l2': "module.0.module.1.lstm.weight_hh_l2",
        'module.1.lstm.bias_ih_l2': "module.0.module.1.lstm.bias_ih_l2",
        'module.1.lstm.bias_hh_l2': "module.0.module.1.lstm.bias_hh_l2",
        'module.2.module.0.module.0.weight': "module.0.module.2.module.0.weight",
        'module.2.module.0.module.0.bias': "module.0.module.2.module.0.bias",
    }

    new_ckpt = {}
    for k, v in ckpt.items():
        new_ckpt[keys_mapping[k]] = v

    return new_ckpt

import torch
from typing import Optional, List, Union
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import ExplorationType, TensorDictPrimer
from torchrl.modules import (
    MLP,
    GRUModule,
    ValueOperator,
    ActorValueOperator,
    ProbabilisticActor,
)


class Embed(torch.nn.Module):
    """Implements a simple embedding layer.

    It handles the case of having a time dimension (RL training) and not having it (RL inference).

    Args:
        input_size (int): The number of possible input values.
        embedding_size (int): The size of the embedding vectors.
    """

    def __init__(self, input_size, embedding_size):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self._embedding = torch.nn.Embedding(input_size, embedding_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        *batch, L = inputs.shape
        if len(batch) > 1:
            inputs = inputs.flatten(0, len(batch) - 1)
        inputs = inputs.squeeze(-1)  # Embedding creates an extra dimension
        out = self._embedding(inputs)
        if len(batch) > 1:
            out = out.unflatten(0, batch)
        return out


def create_gru_components(
        vocabulary_size: int,
        embedding_size: int = 128,
        hidden_size: int = 512,
        num_layers: int = 3,
        dropout: float = 0.0,
        output_size: Optional[int] = None,
        in_key: str = "observation",
        out_key: str = "logits",
        recurrent_state: str = "recurrent_state",
):
    """Create all GRU model components: embedding, GRU, and head.

    These modules handle the case of having a time dimension (RL training) and not having it (RL inference).

    Args:
        vocabulary_size (int): The number of possible input values.
        embedding_size (int): The size of the embedding vectors.
        hidden_size (int): The size of the GRU hidden state.
        num_layers (int): The number of GRU layers.
        dropout (float): The GRU dropout rate.
        output_size (int): The size of the output logits.
        in_key (str): The input key name.
        out_key (str): The output key name.
        recurrent_state (str): The name of the recurrent state.

    Example:
    ```python
    training_model, inference_model = create_model(10)
    ```
    """

    embedding_module = TensorDictModule(
        Embed(vocabulary_size, embedding_size),
        in_keys=[in_key],
        out_keys=["embed"],
    )
    gru_module = GRUModule(
        dropout=dropout,
        input_size=embedding_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        in_keys=["embed", recurrent_state, "is_init"],
        out_keys=["features", ("next", recurrent_state)],
    )
    head = TensorDictModule(
        MLP(
            in_features=hidden_size,
            out_features=output_size or vocabulary_size,
            num_cells=[],
        ),
        in_keys=["features"],
        out_keys=[out_key],
    )

    return embedding_module, gru_module, head

def create_gru_actor(
        vocabulary_size: int,
        embedding_size: int = 128,
        hidden_size: int = 512,
        num_layers: int = 3,
        dropout: float = 0.0,
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
        in_key: str = "observation",
        out_key: str = "logits",
        recurrent_state: str = "recurrent_state_actor",
):
    """Create one GRU-based actor model for inference and one for training.

    Args:
        vocabulary_size (int): The number of possible input values.
        embedding_size (int): The size of the embedding vectors.
        hidden_size (int): The size of the GRU hidden state.
        num_layers (int): The number of GRU layers.
        dropout (float): The GRU dropout rate.
        distribution_class (torch.distributions.Distribution): The distribution class to use.
        return_log_prob (bool): Whether to return the log probability of the action.
        in_key (str): The input key name.
        out_key (str):): The output key name.
        recurrent_state (str): The name of the recurrent state.

    Example:
    ```python
    training_actor, inference_actor = create_gru_actor(10)
    ```
    """
    embedding, gru, head = create_gru_components(
        vocabulary_size, embedding_size, hidden_size, num_layers, dropout,
        vocabulary_size, in_key, out_key, recurrent_state
    )
    policy_head = ProbabilisticActor(
        module=head,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=distribution_class,
        return_log_prob=return_log_prob,
        default_interaction_type=ExplorationType.RANDOM,
    )

    actor_inference_model = TensorDictSequential(embedding, gru, policy_head)
    actor_training_model = TensorDictSequential(embedding, gru.set_recurrent_mode(True), policy_head)
    return actor_training_model, actor_inference_model

def create_gru_critic(
        vocabulary_size: int,
        embedding_size: int = 128,
        hidden_size: int = 512,
        num_layers: int = 3,
        dropout: float = 0.0,
        critic_value_per_action=False,
        in_key: str = "observation",
        out_key: str = "action_value",
        recurrent_state: str = "recurrent_state_critic",

):
    """Create one GRU-based critic model for inference and one for training.

    Args:
        vocabulary_size (int): The number of possible input values.
        embedding_size (int): The size of the embedding vectors.
        hidden_size (int): The size of the GRU hidden state.
        num_layers (int): The number of GRU layers.
        dropout (float): The GRU dropout rate.
        critic_value_per_action (bool): Whether the critic should output a value per action or a single value.
        in_key (Union[str, List[str]]): The input key name.
        out_key (Union[str, List[str]]): The output key name.
        recurrent_state (str): The name of the recurrent state.

    Example:
    ```python
    training_critic, inference_critic = create_gru_critic(10)
    ```
    """

    output_size = vocabulary_size if critic_value_per_action else 1

    embedding, gru, head = create_gru_components(
        vocabulary_size, embedding_size, hidden_size, num_layers, dropout,
        output_size, in_key, out_key, recurrent_state)

    critic_inference_model = TensorDictSequential(embedding, gru, head)
    critic_training_model = TensorDictSequential(embedding, gru.set_recurrent_mode(True), head)
    return critic_training_model, critic_inference_model


def create_gru_actor_critic(
        vocabulary_size: int,
        embedding_size: int = 128,
        hidden_size: int = 512,
        num_layers: int = 3,
        dropout: float = 0.0,
        distribution_class=torch.distributions.Categorical,
        return_log_prob=True,
        critic_value_per_action=False,
        in_key: str = "observation",
        out_key: str = "logits",
        recurrent_state: str = "recurrent_state",
):
    """Create a GRU-based actor-critic model for inference and one for training.

    Args:
        vocabulary_size (int): The number of possible input values.
        embedding_size (int): The size of the embedding vectors.
        hidden_size (int): The size of the GRU hidden state.
        num_layers (int): The number of GRU layers.
        dropout (float): The GRU dropout rate.
        distribution_class (torch.distributions.Distribution): The distribution class to use.
        return_log_prob (bool): Whether to return the log probability of the action.
        critic_value_per_action (bool): Whether the critic should output a value per action or a single value.
        in_key (str): The input key name.
        out_key (str): The output key name.
        recurrent_state (str): The name of the recurrent state.

    Example:
    ```python
    training_actor, inference_actor, training_critic, inference_critic = create_gru_actor_critic(10)
    ```
    """

    embedding, gru, actor_head = create_gru_components(
        vocabulary_size, embedding_size, hidden_size, num_layers, dropout,
        vocabulary_size, in_key, out_key, recurrent_state
    )
    actor_head = ProbabilisticActor(
        module=actor_head,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=distribution_class,
        return_log_prob=return_log_prob,
        default_interaction_type=ExplorationType.RANDOM,
    )
    critic_head = TensorDictModule(
        MLP(
            in_features=hidden_size,
            out_features=vocabulary_size if critic_value_per_action else 1,
            num_cells=[],
        ),
        in_keys=["features"],
        out_keys=["action_value"],
    )

    # Wrap modules in a single ActorCritic operator
    actor_critic_inference = ActorValueOperator(
        common_operator=TensorDictSequential(embedding, gru),
        policy_operator=actor_head,
        value_operator=critic_head,
    )

    actor_critic_training = ActorValueOperator(
        common_operator=TensorDictSequential(
            embedding, gru.set_recurrent_mode(True)
        ),
        policy_operator=actor_head,
        value_operator=critic_head,
    )

    actor_inference = actor_critic_inference.get_policy_operator()
    critic_inference = actor_critic_inference.get_value_operator()
    actor_training = actor_critic_training.get_policy_operator()
    critic_training = actor_critic_training.get_value_operator()

    return actor_training, actor_inference, critic_training, critic_inference


def create_rhs_transform():
    pass
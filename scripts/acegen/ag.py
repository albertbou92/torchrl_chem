#! /usr/bin/python3
import datetime
import json
import os
import random
import shutil
from copy import deepcopy
from functools import partial
from pathlib import Path

import hydra
import numpy as np

import torch
import tqdm
import yaml
from acegen.models import adapt_state_dict, create_gru_actor, models, register_model

from acegen.models.utils import reinitialize_model
from acegen.rl_env import generate_complete_smiles, TokenEnv
from acegen.rl_env.baselines import LeaveOneOutBaseline, MovingAverageBaseline
from acegen.scoring_functions import (
    custom_scoring_functions,
    register_custom_scoring_function,
    Task,
)
from acegen.vocabulary import Vocabulary
from omegaconf import OmegaConf, open_dict
from tensordict.utils import isin
from torch.distributions.kl import kl_divergence

from torchrl.data import (
    LazyTensorStorage,
    PrioritizedSampler,
    RandomSampler,
    TensorDictMaxValueWriter,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.envs import InitTracker, TransformedEnv
from torchrl.modules.utils import get_primers_from_module
from torchrl.record.loggers import get_logger

try:
    import molscore
    from molscore import MolScoreBenchmark, MolScoreCurriculum
    from molscore.manager import MolScore

    _has_molscore = True
except ImportError as err:
    _has_molscore = False
    MOLSCORE_ERR = err

# hydra outputs saved in /tmp
os.chdir("/tmp")


@hydra.main(
    config_path=".",
    config_name="config_molopt_denovo",
    version_base="1.2",
)
def main(cfg: "DictConfig"):

    if isinstance(cfg.seed, int):
        seeds = [cfg.seed]
    else:
        seeds = cfg.seed

    for seed in seeds:

        # Set seed
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        cfg.seed = int(seed)

        # Define save_dir and save config
        current_time = datetime.datetime.now()
        timestamp_str = current_time.strftime("%Y_%m_%d_%H%M%S")
        os.chdir(os.path.dirname(__file__))
        save_dir = (
            f"{cfg.log_dir}/{cfg.experiment_name}_{cfg.agent_name}_{timestamp_str}"
        )
        with open_dict(cfg):
            cfg.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        with open(Path(save_dir) / "config.yaml", "w") as yaml_file:
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)
            yaml.dump(cfg_dict, yaml_file, default_flow_style=False)

        # Define training task and run
        if cfg.get("molscore_task", None):

            if not _has_molscore:
                raise RuntimeError(
                    "MolScore library not found. Unable to create a scoring function. "
                    "To install MolScore, use: `pip install MolScore`"
                ) from MOLSCORE_ERR

            if cfg.molscore_mode == "single":
                task = MolScore(
                    model_name=cfg.agent_name,
                    task_config=cfg.molscore_task,
                    budget=cfg.total_smiles,
                    output_dir=os.path.abspath(save_dir),
                    add_run_dir=False,
                    score_invalids=False,
                    **cfg.get("molscore_kwargs", {}),
                )
                run_reinforce(cfg, task)

            if cfg.molscore_mode == "benchmark":
                MSB = MolScoreBenchmark(
                    model_name=cfg.agent_name,
                    model_parameters=dict(cfg),
                    benchmark=cfg.molscore_task,
                    budget=cfg.total_smiles,
                    output_dir=os.path.abspath(save_dir),
                    add_benchmark_dir=False,
                    score_invalids=True,
                    **cfg.get("molscore_kwargs", {}),
                )
                for task in MSB:
                    run_reinforce(cfg, task)
                    task.write_scores()

            if cfg.molscore_mode == "curriculum":
                task = MolScoreCurriculum(
                    model_name=cfg.agent_name,
                    model_parameters=dict(cfg),
                    benchmark=cfg.molscore_task,
                    budget=cfg.total_smiles,
                    output_dir=os.path.abspath(save_dir),
                    **cfg.get("molscore_kwargs", {}),
                )
                run_reinforce(cfg, task)

        elif cfg.get("custom_task", None):
            if cfg.custom_task not in custom_scoring_functions:
                register_custom_scoring_function(cfg.custom_task, cfg.custom_task)
            task = Task(
                name=cfg.custom_task,
                scoring_function=custom_scoring_functions[cfg.custom_task],
                budget=cfg.total_smiles,
                output_dir=save_dir,
            )
            run_reinforce(cfg, task)

        else:
            raise ValueError("No scoring function specified.")


def run_reinforce(cfg, task):

    # Get available device
    device = (
        torch.device("cuda:0") if torch.cuda.device_count() > 0 else torch.device("cpu")
    )

    # If custom model, register it
    if cfg.model not in models and cfg.get("custom_model_factory", None) is not None:
        register_model(cfg.model, cfg.model_factory)

    # Check if model is available
    if cfg.model not in models:
        raise ValueError(
            f"Model {cfg.model} not found. For custom models, define a model factory as explain in the tutorials."
        )

    # Get model
    (create_actor, _, _, voc_path, ckpt_path, tokenizer) = models[cfg.model]

    # Create vocabulary
    ####################################################################################################################

    vocabulary = Vocabulary.load(voc_path, tokenizer=tokenizer)

    # Create models
    ####################################################################################################################

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    actor_training, actor_inference = create_actor(vocabulary_size=len(vocabulary))
    actor_inference.load_state_dict(
        adapt_state_dict(deepcopy(ckpt), actor_inference.state_dict())
    )
    actor_inference = actor_inference.to(device)
    actor_training = actor_training.to(device)

    prior, _ = create_actor(vocabulary_size=len(vocabulary))
    prior.load_state_dict(adapt_state_dict(deepcopy(ckpt), prior.state_dict()))
    prior = prior.to(device)

    if cfg.get("rnd_coef", False):
        rnd_target, _ = create_gru_actor(vocabulary_size=len(vocabulary))
        rnd_pred, _ = create_gru_actor(vocabulary_size=len(vocabulary))
        rnd_target = rnd_target.to(device)
        reinitialize_model(rnd_target, seed=cfg.seed)
        rnd_pred = rnd_pred.to(device)
    else:
        rnd_target, rnd_pred = None, None

    # Create RL environment
    ####################################################################################################################

    env_kwargs = {
        "start_token": vocabulary.start_token_index,
        "end_token": vocabulary.end_token_index,
        "length_vocabulary": len(vocabulary),
        "batch_size": cfg.num_envs,
        "device": device,
    }

    def create_env_fn():
        """Create a single RL rl_env."""
        env = TokenEnv(**env_kwargs)
        env = TransformedEnv(env)
        env.append_transform(InitTracker())
        if primers := get_primers_from_module(actor_inference):
            env.append_transform(primers)
        return env

    env = create_env_fn()

    # Create replay buffer
    ####################################################################################################################

    storage = LazyTensorStorage(cfg.replay_buffer_size, device=device)
    if cfg.get("replay_sampler", "prioritized") in ["random", "uniform"]:
        replay_sampler = RandomSampler()
    elif cfg.get("replay_sampler", "prioritized") == "prioritized":
        replay_sampler = PrioritizedSampler(storage.max_size, alpha=1.0, beta=1.0)
    else:
        raise ValueError(f"Unknown replay sampler: {replay_sampler}")
    experience_replay_buffer = TensorDictReplayBuffer(
        storage=storage,
        sampler=replay_sampler,
        batch_size=cfg.replay_batch_size,
        writer=TensorDictMaxValueWriter(rank_key="priority"),
        priority_key="priority",
    )

    # Select baseline
    ####################################################################################################################

    # Select baseline
    baseline = None
    if cfg.get("baseline", False):
        if cfg.baseline == "loo":
            baseline = LeaveOneOutBaseline()
        elif cfg.baseline == "mab":
            baseline = MovingAverageBaseline(device=device)
        else:
            raise ValueError(f"Unknown baseline: {cfg.baseline}")

    if cfg.get("rnd_coef", False):
        rnd_baseline = MovingAverageBaseline(device=device)
    else:
        rnd_baseline = None

    # Create optimizer
    ####################################################################################################################

    optim = torch.optim.Adam(
        actor_training.parameters(),
        lr=cfg.lr,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )
    if cfg.get("lr_annealing", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=5000 // cfg.num_envs, eta_min=0.0001
        )

    if cfg.get("rnd_coef", False):
        rnd_optim = torch.optim.Adam(
            rnd_pred.parameters(),
            lr=0.001,
        )
    else:
        rnd_optim = None

    # Create logger
    ####################################################################################################################

    logger = None
    if cfg.logger_backend:
        experiment_name = f"{cfg.agent_name}"
        try:
            experiment_name += f"_{task.cfg.get('task')}"
        except AttributeError:
            experiment_name += "_custom_task"
        logger = get_logger(
            cfg.logger_backend,
            logger_name=cfg.save_dir,
            experiment_name=experiment_name,
            wandb_kwargs={
                "config": dict(cfg),
                "project": cfg.experiment_name,
                "group": cfg.agent_name,
                "reinit": True,
            },
        )

    # Training loop
    ####################################################################################################################

    total_done = 0
    pbar = tqdm.tqdm(total=cfg.total_smiles)

    while not task.finished:

        # Generate data
        data = generate_complete_smiles(
            policy_sample=actor_inference,
            policy_evaluate=actor_training,
            vocabulary=vocabulary,
            scoring_function=task,
            environment=env,
            prompt=cfg.get("prompt", None),
            promptsmiles=cfg.get("promptsmiles", None),
            promptsmiles_optimize=cfg.get("promptsmiles_optimize", True),
            promptsmiles_shuffle=cfg.get("promptsmiles_shuffle", True),
            promptsmiles_multi=cfg.get("promptsmiles_multi", False),
            promptsmiles_scan=cfg.get("promptsmiles_scan", False),
            remove_duplicates=True,
        )

        log_info = {}
        sample_n = data.size(0)
        data_next = data.get("next")
        done = data_next.get("done").squeeze(-1)
        total_done += cfg.num_envs
        pbar.update(done.sum().item())

        # Save info about smiles lengths and rewards
        episode_rewards = data_next["reward"][done]
        episode_length = (data_next["observation"] != 0.0).float().sum(-1).mean()
        if len(episode_rewards) > 0:
            log_info.update(
                {
                    "train/total_smiles": total_done,
                    "train/reward": episode_rewards.mean().item(),
                    "train/min_reward": episode_rewards.min().item(),
                    "train/max_reward": episode_rewards.max().item(),
                    "train/episode_length": episode_length.item(),
                }
            )

        # Apply Hill-Climb
        if cfg.get("topk", False) and cfg.get("topk", False) < 1:
            sscore, sscore_idxs = (
                data_next["reward"][done].squeeze(-1).sort(descending=True)
            )
            data = data[sscore_idxs.data[: int(cfg.num_envs * cfg.topk)]]

        # Apply experience replay
        if (
            cfg.replay_buffer_size
            and len(experience_replay_buffer) > cfg.replay_batch_size
        ):
            replay_batch = experience_replay_buffer.sample().exclude(
                "priority", "index", "prior_log_prob", "_weight", "SMILES"
            )
            data = torch.cat((data.exclude("SMILES"), replay_batch), 0)

        # Compute loss
        data, loss, agent_likelihood = compute_loss(
            data,
            actor_training,
            prior,
            alpha=cfg.get("alpha", 1),
            sigma=cfg.get("sigma", 0.0),
            baseline=baseline,
            entropy_coef=cfg.get("entropy_coef", 0.0),
            kl_coef=cfg.get("kl_coef", 0.0),
            rnd_coef=cfg.get("rnd_coef", 0.0),
            rnd_target=rnd_target,
            rnd_pred=rnd_pred,
            rnd_baseline=rnd_baseline,
        )
        # RND
        if cfg.get("rnd_coef", False):
            # Update RND
            RND_loss = update_rand(data, rnd_target, rnd_pred, rnd_optim)
            log_info.update({f"train/RND_loss": RND_loss.item()})

        # Average loss over the batch
        loss = loss.mean()

        # Add regularizer that penalizes high likelihood for the entire sequence
        if cfg.get("likely_penalty_coef", False):
            loss_p = -(1 / agent_likelihood).mean()
            loss += cfg.likely_penalty_coef * loss_p

        # Calculate gradients and make an update to the network weights
        optim.zero_grad()
        loss.backward()
        optim.step()
        log_info.update({f"train/loss": loss.item()})

        if cfg.get("lr_annealing", None) & (total_done < 5000):
            log_info.update(
                {
                    "train/lr": scheduler.get_last_lr()[0],
                }
            )
            scheduler.step()

        # Then add new experiences to the replay buffer
        if cfg.get("replay_buffer_size", 100):

            replay_data = data.clone()

            # MaxValueWriter is not compatible with storages of more than one dimension.
            replay_data.batch_size = [replay_data.batch_size[0]]

            # Remove SMILES that are already in the replay buffer
            if len(experience_replay_buffer) > 0:
                is_duplicated = isin(
                    input=replay_data,
                    key="action",
                    reference=experience_replay_buffer[:],
                )
                replay_data = replay_data[~is_duplicated]

            # Add data to the replay buffer
            if len(replay_data) > 0:
                reward = replay_data.get(("next", "reward"))
                replay_data.set("priority", reward)
                experience_replay_buffer.extend(replay_data)

        # Log info
        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, step=total_done)


def get_log_prob(data, model):
    mask = data.get("mask").squeeze(-1)
    actions = data.get("action")
    model_in = data.select(*model.in_keys, strict=False)
    dist = model.get_dist(model_in)
    log_prob = dist.log_prob(actions)
    log_prob = (log_prob * mask).sum(-1)
    return log_prob, dist


def compute_loss(
    data,
    model,
    prior,
    alpha=1,
    sigma=0.0,
    baseline=None,
    entropy_coef=0.0,
    kl_coef=0.0,
    rnd_coef=0.0,
    rnd_target=None,
    rnd_pred=None,
    rnd_baseline=None,
):

    mask = data.get("mask").squeeze(-1)

    # Get Prior LL
    with torch.no_grad():
        prior_likelihood, prior_dist = get_log_prob(data, prior)
    # Get Agent LL
    agent_likelihood, agent_dist = get_log_prob(data, model)

    # Reward reshaping
    reward = data.get(("next", "reward")).squeeze(-1).sum(-1)
    reward = torch.pow(torch.clamp(reward + sigma * prior_likelihood, min=0.0), alpha)

    # Subtract baselines
    if baseline:
        baseline.update(reward.detach())
        reward = reward - baseline.mean

    # Add intrinsic RND reward
    if rnd_coef:
        with torch.no_grad():
            target_likelihood, _ = get_log_prob(data, rnd_target)
            pred_likelihood, _ = get_log_prob(data, rnd_pred)
            rnd_reward = torch.pow((pred_likelihood - target_likelihood), 2).detach()
        # Normalize
        rnd_baseline.update(rnd_reward)
        rnd_reward = rnd_reward / rnd_baseline.std
        reward = reward + (rnd_coef * rnd_reward)

    # REINFORCE (negative as we minimize the negative log prob to maximize the policy)
    loss = -agent_likelihood * reward

    # Add KL loss term
    if kl_coef:
        kl_div = kl_divergence(prior_dist, agent_dist)
        kl_div = (kl_div * mask.squeeze()).sum(-1)
        loss += kl_coef * kl_div

    # Add Entropy loss term
    if entropy_coef:
        loss -= entropy_coef * (agent_dist.entropy() * mask.squeeze()).sum(-1)

    return data, loss, agent_likelihood


def update_rand(data, rnd_target, rnd_pred, rnd_optim):
    with torch.no_grad():
        target_likelihood, _ = get_log_prob(data, rnd_target)
    pred_likelihood, _ = get_log_prob(data, rnd_pred)

    # Loss
    loss = torch.pow((pred_likelihood - target_likelihood), 2).mean()

    # Update
    rnd_optim.zero_grad()
    loss.backward()
    rnd_optim.step()

    return loss


if __name__ == "__main__":
    main()

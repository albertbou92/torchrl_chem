import tqdm
import hydra
import torch
from pathlib import Path

from tensordict import TensorDict
from torchrl.envs import (
    Compose,
    ParallelEnv,
    SerialEnv,
    TransformedEnv,
    InitTracker,
    StepCounter,
    RewardSum,
    CatFrames,
    KLRewardTransform,
    UnsqueezeTransform,
)
from torchrl.envs.libs.gym import GymWrapper
from torchrl.record.loggers import get_logger
from torchrl.collectors import SyncDataCollector
from torchrl.objectives.value.advantages import GAE
from torchrl.objectives import ClipPPOLoss
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

from env import GenChemEnv, Monitor
from utils import create_shared_model
from reward_transform import SMILESReward
from scoring import WrapperScoringClass
# from writer import TensorDictMaxValueWriter


@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg: "DictConfig"):

    seed = 101

    # Set a seed for the random module
    import random
    random.seed(int(seed))

    # Set a seed for the numpy module
    import numpy as np
    np.random.seed(int(seed))

    # Set a seed for the torch module
    torch.manual_seed(int(seed))

    device = torch.device(cfg.device) if torch.cuda.is_available() else torch.device("cpu")

    scoring = WrapperScoringClass()
    vocabulary = torch.load(Path(__file__).resolve().parent / "priors" / "vocabulary.prior")
    env_kwargs = {"scoring_function": scoring.get_final_score, "vocabulary": vocabulary}

    # Models
    ####################################################################################################################

    test_env = GymWrapper(GenChemEnv(**env_kwargs))
    action_spec = test_env.action_spec
    actor_inference, actor_training, critic_inference, critic_training, rhs_transform = create_shared_model(vocabulary=vocabulary, output_size=action_spec.shape[-1])
    ckpt = torch.load(Path(__file__).resolve().parent / "priors" / "actor_critic.prior")
    actor_inference.load_state_dict(ckpt)
    actor_training.load_state_dict(ckpt)
    actor_inference = actor_inference.to(device)
    actor_training = actor_training.to(device)
    critic_inference = critic_inference.to(device)
    critic_training = critic_training.to(device)

    # Environment
    ####################################################################################################################

    def create_base_env():
        env = Monitor(GenChemEnv(**env_kwargs), log_dir=cfg.log_dir)
        env = GymWrapper(env, categorical_action_encoding=True, device=device)
        env = TransformedEnv(env)
        env.append_transform(rhs_transform)
        return env

    def create_env_fn(num_workers=cfg.num_env_workers):
        env = ParallelEnv(
            create_env_fn=create_base_env,
            num_workers=num_workers,
        )
        env = TransformedEnv(env)
        env.append_transform(StepCounter())
        env.append_transform(InitTracker())
        env.append_transform(UnsqueezeTransform(in_keys=["observation"], out_keys=["observation"], unsqueeze_dim=-1))
        env.append_transform(CatFrames(N=100, dim=-1, in_keys=["observation"], out_keys=["SMILES"]))
        env.append_transform(KLRewardTransform(actor_inference, coef=cfg.kl_coef, out_keys="reward-kl"))
        return env

    # Collector
    ####################################################################################################################

    collector = SyncDataCollector(
        create_env_fn=create_env_fn,
        policy=actor_inference,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        device=device,
        storing_device=device,
    )

    # Loss modules
    ####################################################################################################################

    adv_module = GAE(
        gamma=cfg.gamma,
        lmbda=cfg.lmbda,
        value_network=critic_inference,
        average_gae=False,
        shifted=True,
    )
    adv_module.set_keys(reward="reward-kl")
    adv_module = adv_module.to(device)
    loss_module = ClipPPOLoss(
        actor_training, critic_training,
        critic_coef=cfg.critic_coef,
        entropy_coef=cfg.entropy_coef,
        clip_epsilon=cfg.ppo_clip,
        loss_critic_type="l2",
        normalize_advantage=True,

    )
    loss_module = loss_module.to(device)
    # use end-of-life as done key
    loss_module.set_keys(reward="reward-kl")

    # Storage
    ####################################################################################################################

    sampler = SamplerWithoutReplacement()
    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(cfg.num_env_workers, device=device),
        sampler=sampler,
        batch_size=cfg.mini_batch_size,
        prefetch=2,
    )

    # topSMILESBuffer = TensorDictReplayBuffer(
    #     storage=LazyTensorStorage(100, device=device),
    #     sampler=sampler,
    #     batch_size=cfg.mini_batch_size,
    #     prefetch=2,
    # )

    diversity_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(100_000, device=device),
    )

    # Optimizer
    ####################################################################################################################

    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        eps=1e-5,
    )

    # Logger
    ####################################################################################################################

    logger = None
    if cfg.logger_backend:
        logger = get_logger(cfg.logger_backend, logger_name="ppo", experiment_name=cfg.experiment_name)

    # Training loop
    ####################################################################################################################

    total_done = 0
    collected_frames = 0
    repeated_smiles = 0
    pbar = tqdm.tqdm(total=cfg.total_frames)
    losses = TensorDict({}, batch_size=[cfg.ppo_epochs, cfg.num_env_workers // cfg.mini_batch_size])

    for data in collector:

        log_info = {}
        frames_in_batch = data.numel()
        total_done += data.get(("next", "done")).sum()
        collected_frames += frames_in_batch
        pbar.update(data.numel())

        # Log end-of-episode accumulated rewards for training
        episode_rewards = data["next", "reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            log_info.update({
                "train/reward": episode_rewards.mean().item(),
                "train/min_reward": episode_rewards.min().item(),
                "train/max_reward": episode_rewards.max().item(),
                "train/total_smiles": total_done,
                "train/repeated_smiles": repeated_smiles,
            })

        # # Penalize repeated SMILES
        # td = data.get("next")
        # done = td.get("done").squeeze(-1)
        # sub_td = td.get_sub_tensordict(done)
        # reward = sub_td.pop("reward-kl")
        # finished_smiles = sub_td.get("SMILES")
        # finished_smiles_td = sub_td.select("SMILES")
        # num_unique_smiles = len(diversity_buffer)
        # num_finished_smiles = len(finished_smiles_td)
        #
        # if num_finished_smiles > 0 and num_unique_smiles == 0:
        #     diversity_buffer.extend(finished_smiles_td.clone())
        #
        # elif num_finished_smiles > 0:
        #     for i, smi in enumerate(finished_smiles):
        #         td_smiles = diversity_buffer._storage._storage
        #         unique_smiles = td_smiles.get("_data").get("SMILES")[0:num_unique_smiles]
        #         repeated = (smi == unique_smiles).all(dim=-1).any()
        #         # TODO: also valid SMILES!
        #         if repeated:
        #             reward[i] = reward[i] * 0.5
        #             repeated_smiles += 1
        #         else:
        #             diversity_buffer.extend(finished_smiles_td[i:i+1].clone())  # TODO: is clone necessary?
        #             # diversity_buffer.add(finished_smiles_td[i].clone())  # TODO: is clone necessary?
        #             num_unique_smiles += 1
        # sub_td.set("reward-kl", reward, inplace=True)

        for j in range(cfg.ppo_epochs):

            with torch.no_grad():
                data = adv_module(data)

            # it is important to pass data that is not flattened
            buffer.extend(data)

            for i, batch in enumerate(buffer):

                loss = loss_module(batch)
                losses[j, i] = loss.select("loss_critic", "loss_entropy", "loss_objective").detach()
                loss_sum = loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]

                # Backward pass
                loss_sum.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=cfg.max_grad_norm)

                optim.step()
                optim.zero_grad()

        losses_mean = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses_mean.items():
            log_info.update({f"train/{key}": value.item()})

        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, collected_frames)
        collector.update_policy_weights_()

    collector.shutdown()


if __name__ == "__main__":
    main()

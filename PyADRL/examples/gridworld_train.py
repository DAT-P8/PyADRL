import random
import grpc
import ray
import os
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ..envs.ngw_env import NGWEnvironment
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ..envs.map_configs.square_map import SquareMapConfig
from ..envs.reward_functions.grid_world_rewards import GridWorldRewards
from ray.tune.registry import register_env
from ..logger.metricslogger import (
    MetricsCallback,
    build_train_iteration_data,
    build_train,
    metrics_path,
    write_metrics,
)

import matplotlib.pyplot as plt
from ..utils.model_save import restore_training, setup_checkpoint_dir
from ..utils.map_load import load_map_config

# Probability of sampling an old opponent policy
P_OLD = 0.3

# Number of alternating stages and PPO iterations per stage
N_STAGES = 4
ITERS_PER_STAGE = 20


def sample_opponent(pool: list[dict]) -> dict:
    """With prob P_OLD sample a random old policy, otherwise use the latest."""
    if len(pool) == 1:
        return pool[-1]  # most recent
    elif random.random() < P_OLD:
        return random.choice(pool[:-1])  # Sample from all but the last policy
    else:
        return pool[-1]


def gridworld_train(
    map: str,
    checkpoint: str | None = None,
    model_name: str | None = None,
):
    # If Ray is already initialized from a previous run, shut it down before starting a new one.
    ray.shutdown()
    ray.init()

    width, height, target_x, target_y, objects = load_map_config(map)

    register_env(
        "gridworld",
        lambda cfg: ParallelPettingZooEnv(
            NGWEnvironment(
                channel=grpc.insecure_channel("localhost:50051"),
                map_config=SquareMapConfig(width, height, target_x, target_y, objects),
                reward_function=GridWorldRewards(),
                n_pursuers=2,
                n_evaders=1,
            )
        ),
    )

    config: PPOConfig = (
        PPOConfig()
        .environment(
            "gridworld",
            env_config={
                "map_width": width,
                "map_height": height,
                "target_x": target_x,
                "target_y": target_y,
            },
        )
        .multi_agent(
            policies={"pursuer_policy", "evader_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: (
                "pursuer_policy" if "pursuer" in str(agent_id) else "evader_policy"
            ),
        )
        .learners(
            num_learners=2,  # Number of parallel learner processes for computing gradients
        )
        .env_runners(
            num_env_runners=4,  # Number of processes/threads that run the environment in parallel
            num_envs_per_env_runner=5,  # Number of environments per env_runner
        )
        .training(
            train_batch_size=10000,  # Number of timesteps before each gradient update. Larger batches = more stable gradients
            minibatch_size=512,  # Size of each mini batch for each SGD update
            num_epochs=10,  # Number of full passes over the train batch per learner. More epochs = more gradient updates per batch
            lr=3e-4,  # Learning rate for optimization
            gamma=0.99,  # Discount factor: future rewards are multiplied by gamma
            lambda_=0.95,  # Balances short-term, low-variance estimates against long-term, high-variance returns in GAE (General Advantage Estimation)
            clip_param=0.2,  # The PPO clip parameter: Limits how much the policy can change in one update
            vf_loss_coeff=0.5,  # Weight of the value function loss in the total loss
            entropy_coeff=0.01,  # Encourage exploration
        )
        .callbacks(MetricsCallback)
        .evaluation(
            evaluation_num_env_runners=0
        )  # No separate evaluation environments. >0 = parallel evaluation of policy while training
    )

    algo = config.build_algo()

    pursuer_pool: list[dict] = []
    evader_pool: list[dict] = []
    start_iteration = 0
    checkpoint_dir = ""

    train_metrics_path = metrics_path("train")
    if checkpoint is not None:
        checkpoint_dir, start_iteration = restore_training(
            algo, checkpoint, pursuer_pool, evader_pool, model_name=model_name
        )
    else:
        checkpoint_dir = setup_checkpoint_dir(model_name=model_name)
        print(f"No checkpoint specified. Starting new training run at {checkpoint_dir}")

    rewards = []
    episodes_data = []

    # try/finally ensures Ray always shuts down cleanly even if training crashes
    try:
        for k in range(start_iteration, start_iteration + N_STAGES):
            # for every stage the evader trains against a frozen pursuer, then the pursuer trains against a frozen evader
            for training_policy, frozen_policy, label, pool, opp_pool in [
                (
                    "evader_policy",
                    "pursuer_policy",
                    "evader",
                    evader_pool,
                    pursuer_pool,
                ),
                (
                    "pursuer_policy",
                    "evader_policy",
                    "pursuer",
                    pursuer_pool,
                    evader_pool,
                ),
            ]:
                print(f"\nStage {k + 1}: training {label}")

                assert algo.learner_group is not None

                # assert is needed because algo.config is typed as `AlgorithmConfig | None`
                assert algo.config is not None
                # Once the algorithm is built using build_algo(), RLlib locks the config as direct mutation is not intended.
                # Only solution (i found) is to unfreeze it, change the multi-agent config, then refreeze it
                algo.config._is_frozen = False
                # Update the learn_group config
                algo.learner_group.foreach_learner(
                    lambda learner, *args: learner.config.multi_agent(
                        policies_to_train=[training_policy]
                    )
                )
                algo.config._is_frozen = True

                # If pool has past policies, sample and load into frozen policy.
                if opp_pool:
                    opp_weights = sample_opponent(opp_pool)
                    algo.learner_group.set_weights({frozen_policy: opp_weights})

                for i in range(ITERS_PER_STAGE):
                    result = algo.train()
                    mean = result["env_runners"]["agent_episode_returns_mean"]
                    rewards.append(mean)
                    iteration_data = build_train_iteration_data(result, i + 1)
                    episodes_data.extend(iteration_data.get("episodes", []))

                    # Keep a live per-episode log while training is in progress.
                    write_metrics(train_metrics_path, {"episodes": episodes_data})

                if len(opp_pool) != 0:
                    algo.learner_group.set_weights({frozen_policy: opp_pool[-1]})

                assert algo.learner_group is not None
                # put the current training policy weights into the pool for future sampling
                updated_weights = algo.learner_group.get_weights()[training_policy]
                pool.append(updated_weights)

                # Save a checkpoint after each full stage (evader+pursuer training)
                if label == "pursuer":
                    # {k + 1:05d} for zero padding e.g. cp_00012 instead of cp_12
                    print(f"Saving stage {k + 1} at {checkpoint_dir}/cp_{k + 1:05d}")
                    check = os.path.abspath(f"{checkpoint_dir}/cp_{k + 1:05d}")
                    algo.save(checkpoint_dir=check)
    finally:
        algo.stop()
        ray.shutdown()

        # Add final aggregate summary after all episodes are complete.
        write_metrics(
            train_metrics_path,
            build_train(episodes_data, final_rewards=rewards[-1] if rewards else {}),
        )

        iterations = list(range(1, len(rewards) + 1))
        evader_rewards = [r["evader_2"] for r in rewards]
        pursuer_0_rewards = [r["pursuer_0"] for r in rewards]

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            iterations, evader_rewards, label="Evader", color="royalblue", linewidth=2
        )
        ax.plot(
            iterations,
            pursuer_0_rewards,
            label="Pursuers",
            color="seagreen",
            linewidth=2,
        )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Reward")
        ax.set_title("Mean reward per Iteration")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.show()

import grpc
import ray
import os
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ..envs.gridworld_env import GridWorldEnvironment
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ..logger.metricslogger import (
    MetricsCallback,
    build_eval,
    build_eval_data,
    build_train_iteration_data,
    build_train,
    metrics_path,
    print_eval_summary,
    write_metrics,
)

import matplotlib.pyplot as plt


def gridworld_train(checkpoint_path: str | None = None):
    ray.init()

    register_env(
        "gridworld",
        lambda cfg: ParallelPettingZooEnv(
            GridWorldEnvironment(channel=grpc.insecure_channel("localhost:50051"))
        ),
    )

    config: PPOConfig = (
        PPOConfig()
        .environment("gridworld")
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

    train_metrics_path = metrics_path("train")
    if checkpoint_path is not None:
        print("Restoring checkpoint from checkpoint:", checkpoint_path)
        algo.restore(checkpoint_path)

    rewards = []
    episodes_data = []

    for i in range(250):
        result = algo.train()

        checkpoint_dir = os.path.abspath(f"./checkpoints/iter_{i + 1}")
        algo.save(checkpoint_dir=checkpoint_dir)

        mean = result["env_runners"]["agent_episode_returns_mean"]
        rewards.append(mean)
        iteration_data = build_train_iteration_data(result, i + 1)
        episodes_data.extend(iteration_data.get("episodes", []))

        # Keep a live per-episode log while training is in progress.
        write_metrics(train_metrics_path, {"episodes": episodes_data})

    # Add final aggregate summary after all episodes are complete.
    write_metrics(
        train_metrics_path,
        build_train(episodes_data, final_rewards=rewards[-1] if rewards else {}),
    )

    iterations = list(range(1, len(rewards) + 1))
    evader_rewards = [r["evader"] for r in rewards]
    pursuer_0_rewards = [r["pursuer_0"] for r in rewards]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(iterations, evader_rewards, label="Evader", color="royalblue", linewidth=2)
    ax.plot(
        iterations, pursuer_0_rewards, label="Pursuers", color="seagreen", linewidth=2
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward")
    ax.set_title("Mean reward per Iteration")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


def gridworld_test(checkpoint_path: str):
    ray.init()

    register_env(
        "gridworld",
        lambda cfg: ParallelPettingZooEnv(
            GridWorldEnvironment(
                channel=grpc.insecure_channel("localhost:50051"), step_delay=0.2
            )
        ),
    )

    config: PPOConfig = (
        PPOConfig()
        .environment("gridworld")
        .multi_agent(
            policies={"pursuer_policy", "evader_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: (
                "pursuer_policy" if "pursuer" in str(agent_id) else "evader_policy"
            ),
        )
        .env_runners(
            num_env_runners=1,
        )
        .callbacks(MetricsCallback)
        .evaluation(evaluation_num_env_runners=1)
    )

    algo = config.build()
    algo.restore(checkpoint_path)

    results = algo.evaluate()
    eval_data = build_eval_data(results)
    eval_episodes = eval_data.get("episodes", [])
    eval_metrics_path = metrics_path("eval")
    write_metrics(
        eval_metrics_path,
        build_eval(eval_episodes, fallback_summary=eval_data.get("summary", {})),
    )
    print_eval_summary(eval_data, eval_metrics_path)

    algo.stop()

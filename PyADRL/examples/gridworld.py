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
    write_metrics,
)


def gridworld_train():
    ray.init(log_to_driver=False)

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
            num_learners=5,
        )
        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=5,
        )
        .training(
            train_batch_size=10000,  # Larger batches = more stable gradients
            minibatch_size=512,
            num_epochs=10,  # More SGD passes per batch
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,  # Encourage exploration
        )
        .callbacks(MetricsCallback)
        .evaluation(evaluation_num_env_runners=0)
    )

    algo = config.build_algo()

    train_metrics_path = metrics_path("train")
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

    import matplotlib.pyplot as plt

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

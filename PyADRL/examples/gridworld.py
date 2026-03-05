import grpc
import ray
import os
import json
from datetime import datetime
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ..envs.gridworld_env import GridWorldEnvironment
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.callbacks.callbacks import RLlibCallback


_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".results")
_EPISODES_JSONL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "_episodes.jsonl")


def _metrics_path(prefix: str) -> str:
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(_RESULTS_DIR, f"{prefix}_{ts}.json")


def _safe(val):
    """Convert numpy/Ray types to JSON-serializable Python types."""
    if val is None:
        return None
    if isinstance(val, dict):
        return {k: _safe(v) for k, v in val.items()}
    try:
        return float(val)
    except (TypeError, ValueError):
        return str(val)


def _read_episodes() -> list[dict]:
    episodes = []
    if os.path.exists(_EPISODES_JSONL_PATH):
        with open(_EPISODES_JSONL_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
    return episodes


def _clear_episodes():
    with open(_EPISODES_JSONL_PATH, "w"):
        pass


class MetricsCallback(RLlibCallback):
    def __init__(self):
        super().__init__()

    def on_episode_end(
        self,
        *,
        episode,
        env_runner,
        metrics_logger,
        env,
        env_index,
        rl_module,
        **kwargs,
    ) -> None:
        infos = episode.get_infos(-1)
        metrics = None

        if isinstance(infos, dict):
            for key, val in infos.items():
                if isinstance(val, dict) and "episode_metrics" in val:
                    metrics = val["episode_metrics"]
                    break
            if metrics is None and "episode_metrics" in infos:
                metrics = infos["episode_metrics"]

        if metrics is None:
            return

        metrics_logger.log_value("capture_rate", metrics["captured"], window=100)
        metrics_logger.log_value("avg_capture_step", metrics["capture_step"], window=100)
        metrics_logger.log_value("breach_rate", metrics["breached"], window=100)

        record = {
            "captured": bool(metrics["captured"]),
            "breached": bool(metrics["breached"]),
            "capture_step": float(metrics["capture_step"]),
            "episode_length": float(metrics["episode_length"]),
        }
        with open(_EPISODES_JSONL_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")


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

    metrics_path = _metrics_path("train")
    rewards = []
    iterations_data = []
    _clear_episodes()

    for i in range(250):
        _clear_episodes()

        result = algo.train()

        checkpoint_dir = os.path.abspath(f"./checkpoints/iter_{i + 1}")
        algo.save(checkpoint_dir=checkpoint_dir)

        mean = result["env_runners"]["agent_episode_returns_mean"]
        rewards.append(mean)

        episodes = _read_episodes()

        iteration_data = {
            "iteration": i + 1,
            "num_episodes": len(episodes),
            "episodes": episodes,
            "summary": {
                "capture_rate": _safe(result["env_runners"].get("capture_rate")),
                "avg_capture_step": _safe(result["env_runners"].get("avg_capture_step")),
                "breach_rate": _safe(result["env_runners"].get("breach_rate")),
            },
            "rewards": _safe(mean),
        }
        iterations_data.append(iteration_data)

        with open(metrics_path, "w") as f:
            json.dump({"iterations": iterations_data}, f, indent=2)

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

    _clear_episodes()
    results = algo.evaluate()

    env_runners = results.get("env_runners", {})
    episodes = _read_episodes()

    eval_data = {
        "num_episodes": len(episodes),
        "episodes": episodes,
        "summary": {
            "capture_rate": _safe(env_runners.get("capture_rate")),
            "avg_capture_step": _safe(env_runners.get("avg_capture_step")),
            "breach_rate": _safe(env_runners.get("breach_rate")),
        },
        "rewards": _safe(env_runners.get("agent_episode_returns_mean", {})),
    }

    metrics_path = _metrics_path("eval")
    with open(metrics_path, "w") as f:
        json.dump({"evaluation": eval_data}, f, indent=2)

    print(f"\n--- Evaluation Results ({len(episodes)} episodes) ---")
    for key, val in eval_data["summary"].items():
        print(f"  {key}: {val}")
    print(f"\nMetrics written to {metrics_path}")

    algo.stop()

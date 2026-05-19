from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from pathlib import Path


def _build_ppo_config(
    config: dict,
    callbacks: list[type[RLlibCallback]] | None = None,
    env_config: dict = {},
    n_pursuers: int = 2,
    n_evaders: int = 1,
    figure_path: Path | None = None,
    metrics_path: Path | None = None,
) -> PPOConfig:
    """Build a PPOConfig with the given hyperparameters."""
    if figure_path:
        env_config["figure_path"] = figure_path
    if metrics_path:
        env_config["metrics_path"] = metrics_path
    env_config["n_pursuers"] = n_pursuers
    env_config["n_evaders"] = n_evaders

    return (
        PPOConfig()
        .environment(
            "gridworld",
            env_config=env_config,
        )
        .multi_agent(
            policies={"pursuer_policy": PolicySpec(), "evader_policy": PolicySpec()},
            policy_mapping_fn=lambda agent_id, *_args, **_kwargs: (
                "pursuer_policy" if "pursuer" in str(agent_id) else "evader_policy"
            ),
        )
        .learners(
            num_learners=config.get("num_learners", 2),
        )
        .env_runners(
            num_env_runners=config.get("num_env_runners", 4),
            num_envs_per_env_runner=config.get("num_envs_per_env_runner", 5),
        )
        .training(
            train_batch_size=config.get("train_batch_size", 10000),
            minibatch_size=config.get("minibatch_size", 512),
            num_epochs=config.get("num_epochs", 10),
            lr=config.get("lr", 3e-4),
            gamma=config.get("gamma", 0.99),
            lambda_=config.get("lambda_", 0.95),
            clip_param=config.get("clip_param", 0.2),
            vf_loss_coeff=config.get("vf_loss_coeff", 0.5),
            entropy_coeff=config.get("entropy_coeff", 0.01),
        )
        .evaluation(
            evaluation_interval=None,
            evaluation_num_env_runners=config.get("evaluation_num_env_runners", 0),
            evaluation_duration=config.get("evaluation_duration", 200),

            # Override the training-time vectorization for evaluation. Without
            # this, eval workers inherit num_envs_per_env_runner from the
            # training config, which can be high (e.g. 128) and pile huge load
            # onto the simulation server during the post-training eval phase.
            evaluation_config={
                "num_envs_per_env_runner": config.get(
                    "evaluation_num_envs_per_env_runner", 4
                ),
            },
        )
        .callbacks(callbacks)
    )

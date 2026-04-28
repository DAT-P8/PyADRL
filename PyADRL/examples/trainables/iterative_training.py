from ray import tune
from ...utils.config_builder import _build_ppo_config
from ray.rllib.callbacks.callbacks import RLlibCallback


def iterative_trainable(
    config: dict,
    checkpoint_dir: str | None = None,
    iterations: int = 25,
    callbacks: list[RLlibCallback] = [],
) -> None:  # type: ignore[type-arg]
    ppo_config = _build_ppo_config(
        lr=config["lr"],
        gamma=config["gamma"],
        lambda_=config["lambda_"],
        clip_param=config["clip_param"],
        vf_loss_coeff=config["vf_loss_coeff"],
        entropy_coeff=config["entropy_coeff"],
        train_batch_size=config["train_batch_size"],
        minibatch_size=config["minibatch_size"],
        num_epochs=config["num_epochs"],
        # All in-process: no remote actors, no placement group conflicts
        num_learners=config.get("num_learners", 0),
        num_env_runners=config.get("num_env_runners", 0),
        num_envs_per_env_runner=config.get("num_envs_per_env_runner", 10),
        callbacks=callbacks,
    )

    # Configuration for evaluating policies after a stage
    ppo_config = ppo_config.evaluation(
        evaluation_interval=None,
        evaluation_num_env_runners=0,
        evaluation_duration=20,
    )

    algo = ppo_config.build_algo()
    try:
        _run_iterative_loop(
            algo,
            iterations=iterations,
            report_to_tune=True,
            checkpoint_dir=checkpoint_dir,
        )
    finally:
        algo.stop()


def _run_iterative_loop(
    algo, iterations: int, report_to_tune=False, checkpoint_dir: str | None = None
) -> list[dict]:
    rewards = []
    for i in range(iterations):
        print(f"Training iteration {i}")
        result = algo.train()
        mean = result["env_runners"]["agent_episode_returns_mean"]
        rewards.append(mean)

        eval_result = algo.evaluate()
        if report_to_tune:
            eval_mean = eval_result["env_runners"]["agent_episode_returns_mean"]
            total_mean_reward = (
                sum(eval_mean.values()) if isinstance(eval_mean, dict) else eval_mean
            )
            tune.report(metrics={"mean_reward": total_mean_reward})
    return rewards

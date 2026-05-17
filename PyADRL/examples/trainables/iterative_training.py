import torch
from ray import tune
from pathlib import Path
from ...utils.config_builder import _build_ppo_config
from ray.rllib.callbacks.callbacks import RLlibCallback


def iterative_trainable(
    config: dict,
    iterations: int = 25,
    callbacks: list[type[RLlibCallback]] | None = None,
    # model_path: Path | None = None,
) -> None:  # type: ignore[type-arg]
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Can only be set once per process; with reuse_actors=True
        # the second trial in the same actor will hit this. Safe to ignore.
        pass
    ppo_config = _build_ppo_config(
        config=config,
        callbacks=callbacks,
    )

    algo = ppo_config.build_algo()
    try:
        _run_iterative_loop(
            algo,
            iterations=iterations,
            report_to_tune=True,
            # model_path=model_path,
        )
    finally:
        algo.stop()


def _run_iterative_loop(
    algo,
    iterations: int,
    report_to_tune=False,
    model_path: Path | None = None,
) -> dict:
    result = {}
    for i in range(1, iterations + 1):
        print(f"Training iteration {i}")
        result = algo.train()

        if model_path:
            model_name = model_path / f"iteration_{i}"
            algo.save(str(model_name))

        eval_result = algo.evaluate()
        if report_to_tune:
            eval_mean = eval_result["env_runners"]["agent_episode_returns_mean"]
            total_mean_reward = (
                sum(eval_mean.values()) if isinstance(eval_mean, dict) else eval_mean
            )
            tune.report(metrics={"mean_reward": total_mean_reward})
    return result

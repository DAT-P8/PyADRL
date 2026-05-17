import torch
from ray import tune
from pathlib import Path
from ...utils.config_builder import _build_ppo_config
from ray.rllib.callbacks.callbacks import RLlibCallback
from ...logger.metrics import summarize_evaluation


def iterative_trainable(
    config: dict,
    iterations: int = 25,
    callbacks: list[type[RLlibCallback]] | None = None,
) -> None:  # type: ignore[type-arg]
    # Pin this trial's process to single-threaded PyTorch (see Tune concurrency notes)
    torch.set_num_threads(1)  # pyright: ignore[reportPrivateImportUsage]
    try:
        torch.set_num_interop_threads(1)  # pyright: ignore[reportPrivateImportUsage]
    except RuntimeError:
        # Already set in this process (reuse_actors=True). Safe to ignore.
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
        )
    finally:
        algo.stop()


def _run_iterative_loop(
    algo,
    iterations: int,
    report_to_tune=False,
    model_path: Path | None = None,
) -> dict:
    # Pull n_evaders once — needed to know what counts as a "full capture".
    n_evaders = 1
    if algo.config is not None and algo.config.env_config is not None:
        n_evaders = algo.config.env_config.get("n_evaders", 1)

    result = {}
    for i in range(1, iterations + 1):
        print(f"Training iteration {i}")
        result = algo.train()

        if model_path:
            model_name = model_path / f"iteration_{i}"
            algo.save(str(model_name))

        eval_result = algo.evaluate()
        if report_to_tune:
            metrics = summarize_evaluation(eval_result, n_evaders=n_evaders)
            tune.report(metrics=metrics)
    return result

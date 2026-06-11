import torch
from ray import tune
from pathlib import Path
from ...utils.config_builder import _build_ppo_config
from ray.rllib.callbacks.callbacks import RLlibCallback
from ...logger.metrics import summarize_evaluation, extract_entropies


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
            # During tune, evaluate every 5 iterations instead of every one.
            # (grace_period=50 → 10 reports before first cull).
            eval_interval=5,
        )
    finally:
        algo.stop()


def _run_iterative_loop(
    algo,
    iterations: int,
    report_to_tune=False,
    model_path: Path | None = None,
    eval_interval: int = 1,
) -> dict:
    # Pull n_evaders once for "full capture".
    n_evaders = 1
    time_limit = 100
    if algo.config is not None and algo.config.env_config is not None:
        n_evaders = algo.config.env_config.get("n_evaders", 1)
        time_limit = algo.config.env_config.get("time_limit", 100)

    result = {}
    # Last seen mean policy entropy per module (both policies train every
    # iteration in the simultaneous loop, so this is always fresh).
    last_entropy: dict[str, float] = {}
    for i in range(1, iterations + 1):
        print(f"Training iteration {i}")
        result = algo.train()
        last_entropy.update(extract_entropies(result))

        if model_path:
            model_name = model_path / f"iteration_{i}"
            algo.save(str(model_name))

        # Evaluate every eval_interval iterations (and always on the last
        # iteration so the final reported metric reflects the final model).
        if i % eval_interval == 0 or i == iterations:
            eval_result = algo.evaluate()
            if last_entropy:
                print(
                    "Policy entropy (max=ln(9)~2.20): "
                    + ", ".join(f"{p}={e:.3f}" for p, e in sorted(last_entropy.items()))
                )
            if report_to_tune:
                metrics = summarize_evaluation(
                    eval_result,
                    n_evaders=n_evaders,
                    time_limit=time_limit,
                    beta=1,
                    gamma=1,
                )
                # Include the actual algo.train() iteration count so ASHA can
                # use it as time_attr. Ray's automatic training_iteration only
                # counts tune.report() calls, which with eval_interval=5 ticks
                # 5x slower than real iterations — would break grace_period.
                metrics["algo_iteration"] = i
                # Per-policy mean action entropy (see extract_entropies).
                for pid, ent in last_entropy.items():
                    metrics[f"entropy_{pid}"] = ent
                tune.report(metrics=metrics)
    return result

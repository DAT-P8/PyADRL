import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ..envs.map_configs.square_map import SquareMapConfig
from ..envs.reward_functions.grid_world_rewards import GridWorldRewards
from ..utils.register_env import _register_gridworld_env
from .trainables.alternate_training import alternate_trainable

# Callbacks
from ..logger.metricslogger import MetricsCallback
from ..logger.heatmaps import HeatmapCallback

# Number of alternating stages and PPO iterations per stage
N_STAGES = 2
ITERS_PER_STAGE = 2

# Total training iterations across all stages (2 phases per stage: evader + pursuer)
TOTAL_TUNE_ITERATIONS = N_STAGES * ITERS_PER_STAGE * 2


def gridworld_tune(
    num_samples: int = 8, max_concurrent_trials: int = 4, checkpoint: str | None = None
) -> tune.ResultGrid:
    """Run a Ray Tune hyperparameter search over the alternating self-play loop.

    Args:
        num_samples: Number of hyperparameter configurations to try.
        max_concurrent_trials: Max trials running in parallel. Lower this if
            you're memory-constrained (each trial spins up learners + env runners).

    Returns:
        tune.ResultGrid with all trial results.
    """
    ray.shutdown()
    ray.init(log_to_driver=True)

    _register_gridworld_env(
        map_config=SquareMapConfig(11, 11, 5, 5),
        reward_function=GridWorldRewards(),
    )

    search_space = {
        # --- Training params ---
        "lr": tune.loguniform(1e-5, 1e-3),
        "gamma": tune.uniform(0.95, 0.999),
        "lambda_": tune.uniform(0.9, 1.0),
        "clip_param": tune.uniform(0.1, 0.3),
        "vf_loss_coeff": tune.uniform(0.25, 1.0),
        "entropy_coeff": tune.loguniform(0.001, 0.05),
        # --- Architecture params ---
        "train_batch_size": tune.choice([5000, 10000, 20000]),
        "minibatch_size": tune.choice([256, 512, 1024]),
        "num_epochs": tune.choice([5, 10, 15, 20]),
        # --- Resource params (all in-process to avoid placement group errors) ---
        "num_learners": 0,
        "num_env_runners": 0,
        "num_envs_per_env_runner": 10,
    }

    scheduler = ASHAScheduler(
        metric="mean_reward",
        mode="max",
        max_t=TOTAL_TUNE_ITERATIONS,
        # Let each trial finish at least one full phase
        grace_period=ITERS_PER_STAGE * 2,
        # Let top 1/3 trials continue to next stage (rung) while the rest are stopped
        reduction_factor=3,
    )

    callbacks = [MetricsCallback, HeatmapCallback]
    tuner = tune.Tuner(
        tune.with_parameters(
            alternate_trainable,
            checkpoint_dir=checkpoint,
            n_stages=N_STAGES,
            iters_per_stage=ITERS_PER_STAGE,
            callbacks=callbacks,
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            scheduler=scheduler,
            max_concurrent_trials=max_concurrent_trials,
        ),
        run_config=tune.RunConfig(
            name="gridworld_tune",
            storage_path=checkpoint,
            verbose=2,
        ),
    )

    results = tuner.fit()

    try:
        best = results.get_best_result(metric="mean_reward", mode="max")
        if best is not None:
            print("\n=== Best trial ===")
            print(f"Config:  {best.config}")
            best_reward = (
                best.metrics.get("mean_reward", "N/A") if best.metrics else "N/A"
            )
            print(f"Reward:  {best_reward}")
            print(f"Log dir: {best.path}")
    except RuntimeError:
        print("\nNo successful trials. Check error logs above for details.")

    ray.shutdown()
    return results

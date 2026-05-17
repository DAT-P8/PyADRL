import ray
from ray import tune
from ray.tune import ResultGrid
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.tune.schedulers import ASHAScheduler
from ..envs.reward_functions.grid_world_rewards import GridWorldRewards
from ..utils.register_env import _register_gridworld_env
from .trainables.alternate_training import alternate_trainable
from .trainables.iterative_training import iterative_trainable
from .gridworld_train import gridworld_train
from ..utils.save_info import save_info, make_strategy_dict
from ..utils.path_utils import setup_experiment_dirs
from ..utils.map_load import load_map_dict

# === Training specifications ===
# Training types
ALTERNATING = "alternate"
SIMULTANEOUS = "simultaneous"

# Alternating training specifications
N_STAGES = 8
ITERS_PER_STAGE = 10
ALTERNATING_GRACE = N_STAGES * ITERS_PER_STAGE * 2

# Simultaneous training specifications
ITERATIONS = 200
SIMULTANEOUS_GRACE = 20


# === Experiment Configurations ===
EXPERIMENT_NUM = 2

# Amount of hyperparameter configurations we pick for training
NUM_CONFIGS = 3

# How many models gets trained per config
TRAIN_PER_CONFIG = 3

# Training loop used
TRAINING_LOOP = SIMULTANEOUS

# Agents
N_PURSUERS = 2
N_EVADERS = 1

# Shielding
SHIELDING = False


def gridworld_tune(
    map: str,
    tuner_dir: str,
    num_samples: int = 56,
    max_concurrent_trials: int = 14,
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
    ray.init()

    map_dict = load_map_dict(map)
    _register_gridworld_env(
        map_dict=map_dict,
        reward_function=GridWorldRewards(),
        n_pursuers=N_PURSUERS,
        n_evaders=N_EVADERS,
        shielding=SHIELDING,
    )

    # Define what hyperparameter values to try
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
        "num_envs_per_env_runner": 128,
    }

    # Define scheduler
    grace = SIMULTANEOUS_GRACE
    max_time = ITERATIONS
    if TRAINING_LOOP == ALTERNATING:
        grace = ALTERNATING_GRACE
        max_time = N_STAGES * ITERS_PER_STAGE

    scheduler = ASHAScheduler(
        metric="mean_reward",
        mode="max",
        max_t=max_time,
        # Let each trial finish at least one full phase
        grace_period=grace,
        # Let top 1/3 trials continue to next stage (rung) while the rest are stopped
        reduction_factor=4,
    )

    # Setup tuner
    tuner = tune.Tuner(
        get_tune_trainable(),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=num_samples,
            scheduler=scheduler,
            max_concurrent_trials=max_concurrent_trials,
            reuse_actors=True,
        ),
        # We likely don't need this, but it's nice to have for safety
        run_config=tune.RunConfig(
            name="gridworld_tune",
            storage_path=tuner_dir,
            verbose=2,
        ),
    )

    # Find hyperparameters
    results = tuner.fit()

    try:
        n_best = get_best_n(results, 3)
        training_config = make_strategy_dict(
            training_strategy=TRAINING_LOOP,
            n_stages=N_STAGES,
            iters_per_stage=ITERS_PER_STAGE,
            n_iterations=ITERATIONS,
        )
        if n_best is not None:
            # Write n_best configs to file in case of error
            experiment_dirs = make_experiment_dirs(n_best, training_config, map)
            for i, (trial, config_dir) in enumerate(zip(n_best, experiment_dirs)):
                if trial.config is None or trial.metrics is None:
                    raise ValueError("Trial has missing data")

                print(f"\n=== Training trail {i + 1} ===")
                print(f"Reward:\n{trial.metrics.get('mean_reward')}")
                print(f"Config:\n{trial.config}")

                for model_num in range(1, TRAIN_PER_CONFIG + 1):
                    training_dir = config_dir / f"training_{model_num}"
                    gridworld_train(
                        map=map,
                        n_pursuers=N_PURSUERS,
                        n_evaders=N_EVADERS,
                        training_config=training_config,
                        model_config=trial.config,
                        training_path=training_dir,
                        shielding=SHIELDING,
                    )
    except RuntimeError:
        print("\nNo successful trials. Check error logs above for details.")

    ray.shutdown()
    return results


def get_best_n(results: ResultGrid, n: int):
    result_dataframe = results.get_dataframe()
    top_n_indices = result_dataframe.nlargest(n, "mean_reward").index
    top_n_results = [results[i] for i in top_n_indices]
    return top_n_results


def make_experiment_dirs(n_best, training_config, map):
    experiment_dir = setup_experiment_dirs(
        EXPERIMENT_NUM, NUM_CONFIGS, TRAIN_PER_CONFIG
    )

    dirs = []
    for i, experiment in enumerate(n_best):
        config_dir = experiment_dir / f"config_{i + 1}"
        save_info(
            config=experiment.config,
            training_config=training_config,
            map=map,
            n_pursuers=N_PURSUERS,
            n_evaders=N_EVADERS,
            config_dir=config_dir,
        )
        dirs.append(config_dir)
    return dirs


def get_tune_trainable(callbacks: list[type[RLlibCallback]] | None = None):
    if TRAINING_LOOP == ALTERNATING:
        return tune.with_parameters(
            alternate_trainable,
            n_stages=N_STAGES,
            iters_per_stage=ITERS_PER_STAGE,
            callbacks=callbacks,
        )
    else:
        return tune.with_parameters(
            iterative_trainable,
            iterations=ITERATIONS,
            callbacks=callbacks,
        )

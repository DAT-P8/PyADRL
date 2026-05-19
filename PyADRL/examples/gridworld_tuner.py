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
from ..logger.metrics import MetricsCallback

# === Training specifications ===
# Training types
ALTERNATING = "alternate"
SIMULTANEOUS = "simultaneous"

# Alternating training specifications
N_STAGES = 8
ITERS_PER_STAGE = 10
# Each stage has 2 halves (train evader, then pursuer), so total algo.train()
# calls = N_STAGES * ITERS_PER_STAGE * 2. Previous formula had max_t < total,
# which silently disabled ASHA culling because grace > max_t.
ALTERNATING_MAX_T = N_STAGES * ITERS_PER_STAGE * 2
ALTERNATING_GRACE = ALTERNATING_MAX_T // 2

# Simultaneous training specifications
# Convergence analysis showed trials reach peak ~70% through training;
# 250 iterations gives slow-learning configs room to actually converge.
ITERATIONS = 250
SIMULTANEOUS_GRACE = 50


# === Experiment Configurations ===
EXPERIMENT_NUM = 4

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

# === Metric selection ===
# Metric ASHA uses to cull trials during the search. "mean_reward" gives
# the earliest learning signal (works from iteration 1); capture/breach rates
# are often zero in early training and provide poor discrimination.
ASHA_METRIC = "mean_reward"

# Metric used to pick the top-N configs after the search finishes. This is
# where you encode what you actually care about — typically a task-grounded
# metric, not the noisy reward sum. Options exposed by summarize_evaluation:
#   "mean_reward"          - legacy sum-of-agent-rewards (matches ASHA default)
#   "pursuer_reward"       - sum of pursuer agents' returns only
#   "evader_reward"        - sum of evader agents' returns only
#   "full_capture_rate"    - fraction of episodes where all evaders were caught
#   "any_capture_rate"     - fraction of episodes with at least one capture
#   "breach_rate"          - fraction of episodes where an evader reached target
#   "mean_episode_length"  - average steps per episode
#   "pursuer_success"      - full_capture_rate - breach_rate
SELECTION_METRIC = "pursuer_success"


def gridworld_tune(
    map: str,
    tuner_dir: str,
    num_samples: int = 20,
    max_concurrent_trials: int = 12,
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

    search_space = {
        # --- Training params ---
        "lr": tune.loguniform(5e-5, 1e-3),
        "gamma": tune.uniform(0.95, 0.99),
        "lambda_": tune.uniform(0.9, 1.0),
        "clip_param": tune.uniform(0.15, 0.3),
        "vf_loss_coeff": tune.uniform(0.25, 1.0),
        "entropy_coeff": tune.loguniform(0.001, 0.05),
        # --- Architecture params (fixed/narrowed based on data) ---
        "train_batch_size": 10000,
        "minibatch_size": tune.choice([256, 1024]),
        "num_epochs": tune.choice([10, 15]),
        # --- Resource params (all in-process to avoid placement group errors) ---
        "num_learners": 0,
        "num_env_runners": 0,
        "num_envs_per_env_runner": 64,
    }

    # Define scheduler
    grace = SIMULTANEOUS_GRACE
    max_time = ITERATIONS
    if TRAINING_LOOP == ALTERNATING:
        grace = ALTERNATING_GRACE
        max_time = ALTERNATING_MAX_T

    scheduler = ASHAScheduler(
        time_attr="algo_iteration",
        metric=ASHA_METRIC,
        mode="max",
        max_t=max_time,
        grace_period=grace,
        reduction_factor=3,
    )

    # Setup tuner
    tuner = tune.Tuner(
        get_tune_trainable(callbacks=[MetricsCallback]),
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
                print(f"{SELECTION_METRIC}: {trial.metrics.get(SELECTION_METRIC)}")
                print(f"mean_reward: {trial.metrics.get('mean_reward')}")
                print(f"full_capture_rate: {trial.metrics.get('full_capture_rate')}")
                print(f"breach_rate: {trial.metrics.get('breach_rate')}")
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
    except (RuntimeError, FileExistsError, ValueError) as e:
        print(f"\nPost-tune step failed: {e}")
        print(f"Tune results saved at: {tuner_dir}")

    ray.shutdown()
    return results


def get_best_n(
    results: ResultGrid,
    n: int,
    metric: str = SELECTION_METRIC,
    window: int = 5,
    min_iters: int = 100,
):
    """Pick top-N trials by mean of the last `window` reports of `metric`,
    among trials that reached at least `min_iters` training iterations.

    Why a window: RL evaluation reward is noisy iteration-to-iteration.
    A trial that hit 0.9 once and dropped to 0.4 should rank below a trial
    that consistently stayed around 0.75. Averaging the last `window`
    reports captures stable performance instead of best-single-eval.

    Why a min_iters filter: ASHA culls poor trials early. Their final
    metric reflects a partially-trained policy, not converged performance.
    Picking the "best" trial from a mix of converged + half-trained
    trials systematically biases toward fast-but-shallow learners.
    With eval_interval=5 in the tune trainable, each report covers 5
    iterations, so window=5 ≈ 25 iterations of recent history.
    """
    import json
    from pathlib import Path

    def trial_stable_score(trial):
        """Returns (mean_score_over_window, final_algo_iteration) or None."""
        if trial.path is None:
            return None
        result_file = Path(trial.path) / "result.json"
        if not result_file.exists():
            return None
        with open(result_file) as f:
            history = [json.loads(line) for line in f if line.strip()]
        if not history:
            return None
        # algo_iteration is the actual algo.train() call count we emit in
        # the trainable. Ray's auto "training_iteration" only counts
        # tune.report() calls and lags by a factor of eval_interval.
        final_iter = history[-1].get("algo_iteration", 0)
        values = [h.get(metric) for h in history if h.get(metric) is not None]
        if not values:
            return None
        recent = values[-window:]
        return sum(recent) / len(recent), final_iter

    # Score every trial that produced any data
    scored = []
    for trial in results:
        s = trial_stable_score(trial)
        if s is None:
            continue
        scored.append((trial, s[0], s[1]))

    # Prefer trials that reached min_iters; if we don't have enough,
    # fall back to all scored trials rather than failing.
    long_trained = [t for t in scored if t[2] >= min_iters]
    if len(long_trained) >= n:
        candidates = long_trained
        print(
            f"Selecting top-{n} from {len(long_trained)} long-trained trials "
            f"(≥{min_iters} iters)"
        )
    else:
        print(
            f"Warning: only {len(long_trained)} trials reached {min_iters} iters; "
            f"falling back to all {len(scored)} trials"
        )
        candidates = scored

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [t[0] for t in candidates[:n]]


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

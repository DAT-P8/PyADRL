import ray
import torch
from pathlib import Path
from ray import tune

from PyADRL.envs.reward_functions.grid_world_rewards import GridWorldRewards
from PyADRL.examples.gridworld_train import gridworld_train
from PyADRL.utils.map_load import load_map_dict
from PyADRL.utils.paths import get_experiments_dir
from PyADRL.utils.register_env import _register_gridworld_env
from PyADRL.utils.save_info import save_info

hyperparameter1 = {
    "lr": 3e-4,
    "gamma": 0.99,
    "lambda_": 0.95,
    "clip_param": 0.2,
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.01,
    "train_batch_size": 10000,
    "minibatch_size": 10000,
    "num_epochs": 10,
    "num_learners": 0,
    "num_env_runners": 0,
    "num_envs_per_env_runner": 64,
}

hyperparameter2 = {
    "lr": 3e-3,
    "gamma": 0.99,
    "lambda_": 0.95,
    "clip_param": 0.2,
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.01,
    "train_batch_size": 10000,
    "minibatch_size": 10000,
    "num_epochs": 10,
    "num_learners": 0,
    "num_env_runners": 0,
    "num_envs_per_env_runner": 64,
}

hyperparameter3 = {
    "lr": 3e-4,
    "gamma": 0.98,
    "lambda_": 0.95,
    "clip_param": 0.2,
    "vf_loss_coeff": 0.5,
    "entropy_coeff": 0.01,
    "train_batch_size": 10000,
    "minibatch_size": 10000,
    "num_epochs": 10,
    "num_learners": 0,
    "num_env_runners": 0,
    "num_envs_per_env_runner": 64,
}

CONFIGS = [hyperparameter1, hyperparameter2, hyperparameter3]
SEEDS = [42, 67, 1337]
MAP = "map"
EXPERIMENT_NAME = "experiment_1"
N_PURSUERS = 2
N_EVADERS = 1
SHIELDING = False
TRAINING_CONFIG = {
    "name": "simultaneous",
    "n_iterations": 3,
}


def _trial(
    cfg: dict,
    configs: list[dict],
    seeds: list[int],
    experiment_dir: Path,
    training_config: dict,
    map_name: str,
) -> None:
    # Avoid CPU oversubscription when many trials run concurrently.
    torch.set_num_threads(1)  # pyright: ignore[reportPrivateImportUsage]
    try:
        torch.set_num_interop_threads(1)  # pyright: ignore[reportPrivateImportUsage]
    except RuntimeError:
        # Already set in this actor (reuse_actors=True). Safe to ignore.
        pass

    config_idx = cfg["config_idx"]
    seed_idx = cfg["seed_idx"]
    model_config = configs[config_idx]
    seed = seeds[seed_idx]

    training_path = (
        experiment_dir / f"config_{config_idx + 1}" / f"training_{seed_idx + 1}"
    )

    gridworld_train(
        map=map_name,
        n_pursuers=N_PURSUERS,
        n_evaders=N_EVADERS,
        shielding=SHIELDING,
        training_config=training_config,
        model_config=model_config,
        training_path=training_path,
        seed=seed,
    )


def main() -> None:
    experiment_dir = get_experiments_dir() / EXPERIMENT_NAME
    if experiment_dir.exists():
        raise FileExistsError(
            f"Experiment '{EXPERIMENT_NAME}' already exists at {experiment_dir}. Choose a different name."
        )
    experiment_dir.mkdir(parents=True)

    for config_idx, config in enumerate(CONFIGS):
        config_dir = experiment_dir / f"config_{config_idx + 1}"
        config_dir.mkdir()
        save_info(
            config=config,
            training_config=TRAINING_CONFIG,
            map=MAP,
            n_pursuers=N_PURSUERS,
            n_evaders=N_EVADERS,
            config_dir=config_dir,
        )

    ray.shutdown()
    ray.init()

    map_dict = load_map_dict(MAP)
    _register_gridworld_env(
        map_dict=map_dict,
        reward_function=GridWorldRewards(),
        n_pursuers=N_PURSUERS,
        n_evaders=N_EVADERS,
        shielding=SHIELDING,
    )

    n_trials = len(CONFIGS) * len(SEEDS)
    print(f"Launching {n_trials} trials via Ray Tune → {experiment_dir}")

    tuner = tune.Tuner(
        tune.with_parameters(
            _trial,
            configs=CONFIGS,
            seeds=SEEDS,
            experiment_dir=experiment_dir,
            training_config=TRAINING_CONFIG,
            map_name=MAP,
        ),
        param_space={
            "config_idx": tune.grid_search(list(range(len(CONFIGS)))),
            "seed_idx": tune.grid_search(list(range(len(SEEDS)))),
        },
        tune_config=tune.TuneConfig(
            num_samples=1,
            max_concurrent_trials=n_trials,
            reuse_actors=True,
        ),
        run_config=tune.RunConfig(
            name="gridworld_parallel",
            storage_path=str(experiment_dir / "_tune"),
            verbose=2,
        ),
    )

    tuner.fit()
    ray.shutdown()


if __name__ == "__main__":
    main()

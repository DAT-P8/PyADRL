import multiprocessing
from PyADRL.utils.paths import get_experiments_dir
from PyADRL.utils.save_info import save_info
from PyADRL.examples.gridworld_train import gridworld_train

# TODO: måske move them to another file
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


def main() -> None:
    experiment_dir = get_experiments_dir() / EXPERIMENT_NAME
    if experiment_dir.exists():
        raise FileExistsError(
            f"Experiment '{EXPERIMENT_NAME}' already exists at {experiment_dir}. Choose a different name."
        )
    experiment_dir.mkdir(parents=True)

    processes: list[multiprocessing.Process] = []
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
        for training_idx, seed in enumerate(SEEDS):
            training_path = config_dir / f"training_{training_idx + 1}"
            training_path.mkdir()
            p = multiprocessing.Process(
                target=gridworld_train,
                kwargs={
                    "map": MAP,
                    "n_pursuers": N_PURSUERS,
                    "n_evaders": N_EVADERS,
                    "shielding": SHIELDING,
                    "training_config": TRAINING_CONFIG,
                    "model_config": config,
                    "training_path": training_path,
                    "seed": seed,
                },
                name=f"config{config_idx + 1}_seed{seed}",
            )
            processes.append(p)

    print(f"Launching {len(processes)} runs in parallel → {experiment_dir}")
    for p in processes:
        p.start()

    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        print("\nInterrupted — terminating all runs")
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        print("All processes stopped.")
        return

    failed = [p.name for p in processes if p.exitcode != 0]
    if failed:
        print(f"Failed runs: {failed}")
    else:
        print("All 9 runs completed successfully.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()

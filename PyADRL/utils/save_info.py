from .path_utils import save_model_info
from pathlib import Path

ALTERNATING = "alternate"
SIMULTANEOUS = "simultaneous"


def save_info(
    config: dict,
    training_config: dict,
    map: str,
    n_pursuers: int,
    n_evaders: int,
    config_dir: Path,
):
    initial_info = {
        "map": map,
        "n_pursuers": n_pursuers,
        "n_evaders": n_evaders,
        "training_config": training_config,
        "hyperparameters": {
            "num_learners": config.get("num_learners"),
            "num_env_runners": config.get("num_env_runners"),
            "num_envs_per_env_runner": config.get("num_envs_per_env_runner"),
            "train_batch_size": config.get("train_batch_size"),
            "minibatch_size": config.get("minibatch_size"),
            "num_epochs": config.get("num_epochs"),
            "lr": config.get("lr"),
            "gamma": config.get("gamma"),
            "lambda_": config.get("lambda_"),
            "clip_param": config.get("clip_param"),
            "vf_loss_coeff": config.get("vf_loss_coeff"),
            "entropy_coeff": config.get("entropy_coeff"),
        },
    }
    save_model_info(config_dir, initial_info)


def make_strategy_dict(
    training_strategy: str,
    n_stages: int | None = None,
    iters_per_stage: int | None = None,
    n_iterations: int | None = None,
) -> dict:
    if training_strategy == ALTERNATING:
        if n_stages is None or iters_per_stage is None:
            raise ValueError(
                f"Parameters for training type {training_strategy} was not given"
            )
        return {
            "name": training_strategy,
            "n_stages": n_stages,
            "iters_per_stage": iters_per_stage,
        }
    elif training_strategy == SIMULTANEOUS:
        if n_iterations is None:
            raise ValueError(
                f"Parameters for training type {training_strategy} was not given"
            )
        return {
            "name": training_strategy,
            "n_iterations": n_iterations,
        }
    else:
        raise ValueError(f"Received unknown training strategy {training_strategy}")

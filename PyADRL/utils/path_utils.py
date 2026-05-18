import os
import re
import json
from datetime import datetime
from pathlib import Path
from .paths import get_checkpoints_dir, get_experiments_dir


def setup_checkpoints_dir(model_name: str | None = None) -> Path:
    """Create dir for saving checkpoints. If a model name is not provided, creates a model with the current time: YYMMDD_HHMM"""
    model_name = model_name or get_current_time_as_str()

    model_dir = get_experiments_dir() / model_name

    if os.path.isdir(model_dir):  # model_name provided with --name flag
        raise ValueError(
            f"Model name {model_name} already exists. Please choose a different name or delete the existing model."
        )

    return get_checkpoints_dir(model_name)

def setup_model_dir() -> Path:
    """Create dir for saving checkpoints. Creates a model with the current time: YYMMDD_HHMM"""
    model_name = get_current_time_as_str()

    model_dir = get_experiments_dir() / model_name

    if os.path.isdir(model_dir):  # model_name provided with --name flag
        raise ValueError(
            f"Model name {model_name} already exists. Please choose a different name or delete the existing model."
        )

    return model_dir


def setup_experiment_dirs(experiment_num, num_configs, train_per_config):
    experiment_dir = get_experiments_dir() / f"experiment_{experiment_num}"
    experiment_dir.mkdir()
    for i in range(1, num_configs + 1):
        config_dir = experiment_dir / f"config_{i}"
        config_dir.mkdir()
        for j in range(1, train_per_config + 1):
            train_dir = config_dir / f"training_{j}"
            train_dir.mkdir()
            models_dir = train_dir / "models"
            models_dir.mkdir()
            figures_dir = train_dir / "figures"
            figures_dir.mkdir()
    return experiment_dir


def setup_tuner_dir():
    """Create a directory for tuner to save stuff in"""
    tuner_dir = get_experiments_dir() / "tuner"
    tuner_dir.mkdir(exist_ok=True)
    run_dir = tuner_dir / get_current_time_as_str()
    run_dir.mkdir()
    return run_dir


def save_model_info(path: Path, info: dict):
    """Saves model info as a json file in the model directory."""
    info_file = path / "model-info.json"
    with open(info_file, "w") as f:
        json.dump(info, f, indent=2)


# === Utils ===
def checkpoint_exists(model_name: str) -> bool:
    """Checks if the model path has any checkpoints."""
    checkpoint_dir = get_checkpoints_dir(model_name)

    return checkpoint_dir.is_dir() and any(
        re.fullmatch(r"stage_\d{5}", p.name) for p in checkpoint_dir.iterdir()
    )


def get_current_time_as_str() -> str:
    current_date_time = datetime.now().strftime("%y%m%d_%H%M")
    return current_date_time


# === Model restoration ===
def restore_checkpoint(algo, model_name: str):
    """Restore model from latest checkpoint."""

    checkpoint_dir = get_checkpoints_dir(model_name)

    # Ensure file is in format stage_XXXXX
    existing_checkpoints = [
        d.name for d in checkpoint_dir.iterdir() if re.fullmatch(r"stage_\d{5}", d.name)
    ]
    if existing_checkpoints == []:
        raise ValueError(
            f"No checkpoints found in {checkpoint_dir} matching pattern stage_XXXXX"
        )

    latest = sorted(existing_checkpoints)[-1]
    latest_checkpoint = os.path.join(checkpoint_dir, latest)
    print("Restoring checkpoint from:", latest_checkpoint)
    algo.restore(latest_checkpoint)

    # return the checkpoint dir and the iteration number to continue from
    return checkpoint_dir, int(latest.split("_")[1])


def restore_testing(algo, model_name: str):
    """Restores testing from a checkpoint."""
    if not checkpoint_exists(model_name):
        raise ValueError(f"Checkpoint for {model_name} does not exist.")

    _, _ = restore_checkpoint(algo, model_name)

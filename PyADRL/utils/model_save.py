import os
import re
from datetime import datetime
from pathlib import Path
from .paths import get_checkpoints_dir, get_experiments_dir


def checkpoint_exists(model_name: str) -> bool:
    """Checks if the model path has any checkpoints."""
    checkpoint_dir = get_checkpoints_dir(model_name)

    return checkpoint_dir.is_dir() and any(
        re.fullmatch(r"stage_\d{5}", p.name) for p in checkpoint_dir.iterdir()
    )


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


def restore_training(
    algo,
    model_name: str,
    pursuer_pool=None,
    evader_pool=None,
) -> tuple[Path, int]:
    """Restores training from a specified models latest checkpoint. Returns the checkpoint dir and the iteration number to continue from."""
    if not checkpoint_exists(model_name):
        raise ValueError(f"Model {model_name} does not exist.")

    checkpoint_dir, iteration = restore_checkpoint(algo, model_name)
    weights = algo.learner_group.get_weights()
    if pursuer_pool and evader_pool:
        pursuer_pool.append(weights["pursuer_policy"])
        evader_pool.append(weights["evader_policy"])
    return checkpoint_dir, iteration


def restore_testing(algo, model_name: str):
    """Restores testing from a checkpoint."""
    if not checkpoint_exists(model_name):
        raise ValueError(f"Checkpoint for {model_name} does not exist.")

    _, _ = restore_checkpoint(algo, model_name)


def setup_checkpoints_dir(model_name: str | None = None) -> Path:
    """Create dir for saving checkpoints. If a model name is not provided, creates a model with the current time: YYMMDD_HHMM"""
    model_name = model_name or get_current_time_as_str()

    model_dir = get_experiments_dir() / model_name

    if os.path.isdir(model_dir):  # model_name provided with --name flag
        raise ValueError(
            f"Model name {model_name} already exists. Please choose a different name or delete the existing model."
        )

    return get_checkpoints_dir(model_name)


def get_current_time_as_str() -> str:
    current_date_time = datetime.now().strftime("%y%m%d_%H%M")
    return current_date_time

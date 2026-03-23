import os
import re


def checkpoint_exists(checkpoint_path: str) -> bool:
    if os.path.isdir(os.path.abspath(checkpoint_path)):
        return True

    checkpoint_dir = os.path.abspath(f"./checkpoints/{checkpoint_path}")
    return os.path.isdir(checkpoint_dir) and any(
        re.match(r"cp_\d{5}", e.name) for e in os.scandir(checkpoint_dir)
    )


def restore_checkpoint(algo, checkpoint_path: str, model_name: str | None = None):
    """ "Restore from checkpoint. If checkpoint_path is the model name e.g "model_001",
    it will restore newest CP from that model. If it's a path to a checkpoint folder,
    e.g. "./checkpoints/model_001/cp_00010", it will restore from that exact checkpoint."""

    if os.path.dirname(checkpoint_path):
        #  if checkpoint_path is a path to a specific checkpoint file, restore from that file
        print("Restoring checkpoint from file:", checkpoint_path)
        latest_checkpoint = os.path.abspath(checkpoint_path)
        algo.restore(latest_checkpoint)
        # still set up a new checkpoint dir for saving future checkpoints
        checkpoint_dir = setup_checkpoint_dir(model_name=model_name)
        return checkpoint_dir, 0  # start from iteration 0
    else:
        #  if checkpoint_path is a model name e.g. "model_001", restore the latest checkpoint
        checkpoint_dir = os.path.abspath(f"./checkpoints/{checkpoint_path}")

        # Ensure file is in format cp_XXXXX
        existing_checkpoints = [
            d for d in os.listdir(checkpoint_dir) if re.match(r"cp_\d{5}", d)
        ]
        if existing_checkpoints == []:
            raise ValueError(
                f"No checkpoints found in {checkpoint_dir} matching pattern cp_XXXXX"
            )

        latest = sorted(existing_checkpoints)[-1]
        latest_checkpoint = os.path.join(checkpoint_dir, latest)
        print("Restoring checkpoint from:", latest_checkpoint)
        algo.restore(latest_checkpoint)

        if model_name:
            # if model_name is provided, we save future checkpoints in that folder.
            return setup_checkpoint_dir(model_name=model_name), 0
        else:
            # return the checkpoint dir and the iteration number to continue from
            return checkpoint_dir, int(latest.split("_")[1])


def restore_training(
    algo,
    checkpoint_path: str,
    pursuer_pool=None,
    evader_pool=None,
    model_name: str | None = None,
) -> tuple[str, int]:
    """Restores training from a checkpoint. Returns the checkpoint dir and the iteration number to continue from."""
    if not checkpoint_exists(checkpoint_path):
        raise ValueError(f"Checkpoint {checkpoint_path} does not exist.")

    checkpoint_dir, iteration = restore_checkpoint(
        algo, checkpoint_path, model_name=model_name
    )
    weights = algo.learner_group.get_weights()
    if pursuer_pool and evader_pool:
        pursuer_pool.append(weights["pursuer_policy"])
        evader_pool.append(weights["evader_policy"])
    return checkpoint_dir, iteration


def restore_testing(algo, checkpoint_path: str):
    """Restores testing from a checkpoint."""
    if not checkpoint_exists(checkpoint_path):
        raise ValueError(f"Checkpoint {checkpoint_path} does not exist.")

    _, _ = restore_checkpoint(algo, checkpoint_path)


def setup_checkpoint_dir(model_name: str | None = None) -> str:
    """Create dir for saving checkpoints. If a model name is not provided, find the highest model_XXX and creates model_XXX+1"""
    checkpoint_dir = os.path.abspath("./checkpoints")
    if model_name:  # model_name provided with --name flag
        # Check if model_name already exists in checkpoint_dir
        existing_models = [
            d for d in os.listdir(checkpoint_dir) if re.match(rf"{model_name}", d)
        ]
        if existing_models:
            raise ValueError(
                f"Model name {model_name} already exists. Please choose a different name or delete the existing model."
            )
        return os.path.join(checkpoint_dir, model_name)
    else:
        # We match on "model_xxx" where xxx is a number
        existing_checkpoints = [
            d for d in os.listdir(checkpoint_dir) if re.match(r"model_\d{3}", d)
        ]
        if existing_checkpoints:
            latest = sorted(existing_checkpoints)[-1]
            latest_num = int(latest.split("_")[1])
            new_model_name = f"model_{latest_num + 1:03d}"
        else:
            new_model_name = "model_001"
        return os.path.join(checkpoint_dir, new_model_name)

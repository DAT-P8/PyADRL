import os
from pathlib import Path

# TODO: Error handling on file paths not found


# If this file is moved, remember to change the return Path accordingly.
def get_project_root() -> Path:
    """Gets the path of this file and traverses back to the root folder."""
    return Path(__file__).parent.parent.parent


def get_experiments_dir() -> Path:
    """Gets the path of the experiment folder. Creates the folder if it does not already exist."""
    if not os.path.exists(get_project_root() / "experiments"):
        os.mkdir(get_project_root() / "experiments")

    return get_project_root() / "experiments"


def get_env_maps_dir() -> Path:
    """Gets the path of the maps folder. Creates the folder if it does not already exist."""
    if not os.path.exists(get_project_root() / "PyADRL" / "examples" / "maps"):
        os.mkdir(get_project_root() / "PyADRL" / "examples" / "maps")

    return get_project_root() / "PyADRL" / "examples" / "maps"


def get_env_map(map_name: str) -> Path:
    """Gets a user specified environment map."""
    return get_env_maps_dir() / f"{map_name}.json"


def get_model_maps_dir(model_name: str) -> Path:
    if not os.path.exists(get_model_dir(model_name) / "maps"):
        os.mkdir(get_model_dir(model_name) / "maps")

    return get_model_dir(model_name) / "maps"


def get_model_dir(model_name: str) -> Path:
    """Gets the path of the model folder."""
    return get_experiments_dir() / model_name


def get_checkpoints_dir(model_name: str) -> Path:
    """Gets the path of the model checkpoints folder."""
    return get_model_dir(model_name) / "checkpoints"

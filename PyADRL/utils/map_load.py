import json

from PyADRL.envs.map_configs.square_map import SquareMapConfig
from PyADRL.utils.paths import get_env_map


def load_map_config(map_name: str) -> SquareMapConfig:
    """
    Load map config from file name.
    Expected format: map_{width}_{height}_{target_x}_{target_y} e.g. map_11_11_5_5
    """
    map_path = get_env_map(map_name)
    try:
        # load json file from PyADRL/examples/maps with name map_name and parse width, height, target_x, target_y
        with open(
            map_path,
            "r",
        ) as f:
            print(f"Loading map config from {map_path}.")
            map_config = json.load(f)
        return SquareMapConfig(
            map_config["width"],
            map_config["height"],
            map_config["target_x"],
            map_config["target_y"],
            [(obj["x"], obj["y"]) for obj in map_config.get("objects", [])],
        )
    except Exception:
        raise ValueError(f"Invalid map {map_name}")


def load_map_dict(map_name: str) -> dict:
    map_path = get_env_map(map_name)
    try:
        with open(
            map_path,
            "r",
        ) as f:
            map_config = json.load(f)
        return {
            "width": map_config["width"],
            "height": map_config["height"],
            "target_x": map_config["target_x"],
            "target_y": map_config["target_y"],
            "objects": [(obj["x"], obj["y"]) for obj in map_config.get("objects", [])],
        }
    except Exception:
        raise ValueError(f"Invalid map {map_name}")


def dict_to_map_config(map_dict: dict):
    return SquareMapConfig(
        map_dict["width"],
        map_dict["height"],
        map_dict["target_x"],
        map_dict["target_y"],
        map_dict["objects"],
    )

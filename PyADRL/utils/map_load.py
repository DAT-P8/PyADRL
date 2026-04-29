import json
import os

from PyADRL.envs.map_configs.square_map import SquareMapConfig

    
def load_map_config(map_name: str) -> SquareMapConfig:
    """Load map config from file name. Expected format: map_{width}_{height}_{target_x}_{target_y} e.g. map_11_11_5_5"""
    try:
        # load json file from PyADRL/examples/maps with name map_name and parse width, height, target_x, target_y
        with open(
            os.path.join("PyADRL", "examples", "maps", f"{map_name}.json"), "r"
        ) as f:
            print(
                f"Loading map config from {os.path.join('PyADRL', 'examples', 'maps', f'{map_name}.json')}"
            )
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

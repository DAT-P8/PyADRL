from .map_config import MapConfig
from ...ngw.v1.ngw2d_pb2 import SquareMap, MapSpec
import numpy as np


class SquareMapConfig(MapConfig):
    def __init__(self, width: int, height: int, target_x: int, target_y: int):
        self.width = width
        self.height = height
        self.target_x = target_x
        self.target_y = target_y
        self.map_spec = MapSpec(
            square_map=SquareMap(
                width=width, height=height, target_x=target_x, target_y=target_y
            )
        )

    def get_target_position(self) -> tuple[int, int]:
        return (self.target_x, self.target_y)

    def normalise_position(self, x: int, y: int) -> tuple[float, float]:
        return (x / self.width, y / self.height)

    def distance_to_target(self, x: int, y: int) -> float:
        return np.sqrt((x - self.target_x) ** 2 + (y - self.target_y) ** 2)

    def get_map_spec(self) -> MapSpec:
        return self.map_spec

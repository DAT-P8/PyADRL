from .map_config import MapConfig
from ...dtos.map_dtos import SquareMap, MapSpec, ObjectSpec, SquareObject
from ...utils.chebeshyv import chebyshev_distance


class SquareMapConfig(MapConfig):
    def __init__(
        self,
        width: int,
        height: int,
        target_x: int,
        target_y: int,
        objects: list[tuple[int, int]],
    ):
        self.width = width
        self.height = height
        self.target_x = target_x
        self.target_y = target_y
        self.objects = objects
        object_specs: list[ObjectSpec] = [
            ObjectSpec(square_object=SquareObject(x=obj_x, y=obj_y))
            for obj_x, obj_y in self.objects
        ]
        self.map_spec = MapSpec(
            square_map=SquareMap(
                width=width,
                height=height,
                target_x=target_x,
                target_y=target_y,
                objects=object_specs,
            )
        )

    def get_target_position(self) -> tuple[int, int]:
        return (self.target_x, self.target_y)

    def get_objects(self) -> list[tuple[int, int]]:
        return self.objects

    def normalise_position(self, x: int, y: int) -> tuple[float, float]:
        return (x / self.width, y / self.height)

    def distance_to_target(self, x: int, y: int) -> float:
        return chebyshev_distance(x, y, self.target_x, self.target_y)

    def get_map_spec(self) -> MapSpec:
        return self.map_spec

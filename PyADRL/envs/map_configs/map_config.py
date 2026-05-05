from abc import ABCMeta, abstractmethod
from ...dtos.map_dtos import MapSpec


class MapConfig(metaclass=ABCMeta):
    @abstractmethod
    def get_target_position(self) -> tuple[int, int]:
        raise NotImplementedError()

    @abstractmethod
    def get_objects(self) -> list[tuple[int, int]]:
        raise NotImplementedError()

    @abstractmethod
    def normalise_position(self, x: int, y: int) -> tuple[float, float]:
        raise NotImplementedError()

    @abstractmethod
    def distance_to_target(self, x: int, y: int) -> int:
        raise NotImplementedError()

    @abstractmethod
    def normalise_map_distance(self, distance: int) -> float:
        raise NotImplementedError()

    @abstractmethod
    def get_map_spec(self) -> MapSpec:
        raise NotImplementedError()

from abc import ABC, abstractmethod
from ...dtos.map_dtos import MapSpec, ObjectSpec


class MapConfig(ABC):
    @abstractmethod
    def get_target_position(self) -> tuple[int, int]:
        pass

    @abstractmethod
    def get_objects(self) -> list[tuple[int, int]]:
        pass

    @abstractmethod
    def normalise_position(self, x: int, y: int) -> tuple[float, float]:
        pass

    @abstractmethod
    def distance_to_target(self, x: int, y: int) -> float:
        pass

    @abstractmethod
    def get_map_spec(self) -> MapSpec:
        pass

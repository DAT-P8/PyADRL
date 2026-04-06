from abc import ABC, abstractmethod

class MapConfig(ABC):
    @abstractmethod
    def get_target_position():
        pass

    @abstractmethod
    def normalise_position(x: int, y: int):
        pass

    @abstractmethod
    def distance_to_target(x: int, y: int):
        pass

    @abstractmethod
    def get_map_spec():
        pass

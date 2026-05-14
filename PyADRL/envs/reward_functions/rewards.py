from abc import ABC, abstractmethod

from PyADRL.dtos.ngw_dtos import DroneState, Event
from ..map_configs.map_config import MapConfig


class RewardFunction(ABC):
    @abstractmethod
    def calculate_rewards(
        self,
        events: list[Event],
        drones: list[DroneState],
        map_config: MapConfig,
        time_limit_reached: bool,
    ) -> dict[int, float]:
        raise NotImplemented("this method is abstract")

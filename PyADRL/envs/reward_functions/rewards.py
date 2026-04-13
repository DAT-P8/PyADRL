from abc import ABC, abstractmethod
from ...ngw.v1.ngw2d_pb2 import State
from ..ngw_drone import NGW_Drone
from ..map_configs.map_config import MapConfig


class RewardFunction(ABC):
    @abstractmethod
    def calculate_rewards(
        self,
        new_state: State,
        agents: list[str],
        drones: dict[str, list[NGW_Drone]],
        map_config: MapConfig,
        time_limit_reached: bool,
    ) -> dict[str, float]:
        pass

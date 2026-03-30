from abc import abstractmethod
from ...grid_world_pb2 import GWActionResponse

class RewardFunction:
    @abstractmethod
    def calculate_rewards(response: GWActionResponse):
        pass

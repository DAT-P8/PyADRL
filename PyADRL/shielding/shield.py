# action, state is received from environment
# 
# outputs then to environment a safe action,
# and to agents the safe action and punishment

from abc import ABCMeta, abstractmethod
from ..dtos.ngw_dtos import (
    DroneAction,
    State
)

# The ABCMeta class makes sure that instances of this cannot be
# instantiated unless all abstract methods are overridden.
class Shield(metaclass=ABCMeta):
    @abstractmethod
    def shield(self, action: DroneAction, state: State) -> tuple[int, DroneAction]:
        raise NotImplemented("shield has not been implemented")

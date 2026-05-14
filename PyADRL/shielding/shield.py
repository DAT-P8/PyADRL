from abc import ABCMeta, abstractmethod

from PyADRL.dtos.ngw_dtos import DroneAction, State


class Shield(metaclass=ABCMeta):
    @abstractmethod
    def shield(self, actions: list[DroneAction], state: State) -> tuple[list[DroneAction], State | None]:
        raise NotImplemented("this method is abstract")

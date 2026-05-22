from abc import ABCMeta, abstractmethod
from logging import Logger
from pathlib import Path
from typing import override

from PyADRL.pooling.models.training import Training


class TrainingProvider(metaclass=ABCMeta):
    @abstractmethod
    def get_trainings(self, path: Path) -> list[Training]:
        raise NotImplementedError("abstract method")


class FSTrainingProvider(TrainingProvider):
    def __init__(self, logger: Logger) -> None:
        super().__init__()

    @override
    def get_trainings(self, path: Path) -> list[Training]:
        trainings: list[Training] = []

        for train_dir in path.iterdir():
            if not train_dir.is_dir():
                continue

            trainings.append(Training(train_dir.name, train_dir))

        return trainings

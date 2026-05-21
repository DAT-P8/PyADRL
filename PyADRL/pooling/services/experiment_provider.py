from abc import ABCMeta, abstractmethod
import logging
from typing import override

from PyADRL.pooling.services.config_provider import ExperimentConfigProvider
from ...utils import paths
from ..models.experiment import (
    Experiment
)

class ExperimentProvider(metaclass=ABCMeta):
    @abstractmethod
    def get_experiments(self) -> list[Experiment]:
        raise NotImplementedError("abstract method")

    @abstractmethod
    def get_experiment_by_name(self, name: str) -> Experiment | None:
        raise NotImplementedError("abstract method")

class FSExperimentProvider(ExperimentProvider):
    def __init__(self, config_provider: ExperimentConfigProvider, logger: logging.Logger) -> None:
        super().__init__()
        self.config_provider = config_provider
        self.folder = paths.get_experiments_dir()
        self.logger = logger

    @override
    def get_experiment_by_name(self, name: str) -> Experiment | None:
        for experiment_f in self.folder.iterdir():
            if not experiment_f.is_dir() or experiment_f.name != name:
                continue
            
            configs = self.config_provider.get_configs(experiment_f)
            return Experiment(experiment_f.name, configs)

        return None

    @override
    def get_experiments(self) -> list[Experiment]:
        experiments: list[Experiment] = []

        for experiment_f in self.folder.iterdir():
            if not experiment_f.is_dir():
                continue

            configs = self.config_provider.get_configs(experiment_f)
            exp = Experiment(experiment_f.name, configs)
            experiments.append(exp)

        return experiments

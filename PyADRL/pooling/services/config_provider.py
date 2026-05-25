from abc import ABCMeta, abstractmethod
from logging import Logger
from pathlib import Path
from typing import Any, override
import json

from PyADRL.pooling.services.training_provider import TrainingProvider

from ..models.experiment_config import ExperimentConfig


class ExperimentConfigProvider(metaclass=ABCMeta):
    @abstractmethod
    def get_configs(self, path: Path) -> list[ExperimentConfig]:
        raise NotImplementedError("abstract method")


class FSExperimentConfigProvider(ExperimentConfigProvider):
    def __init__(self, logger: Logger, training_provider: TrainingProvider) -> None:
        super().__init__()
        self.training_provider = training_provider
        self.logger = logger

    @override
    def get_configs(self, path: Path) -> list[ExperimentConfig]:
        configs: list[ExperimentConfig] = []

        for file in path.iterdir():
            if not file.is_dir():
                continue

            with open(file / "model-info.json", "r") as f:
                model_info: dict[str, Any] = json.load(f)

            trainings = self.training_provider.get_trainings(file)
            config = ExperimentConfig(file.name, trainings, model_info)
            configs.append(config)

        return configs

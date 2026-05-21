from typing import Any
from PyADRL.pooling.models.training import Training


class ExperimentConfig:
    def __init__(self, name: str, trainings: list[Training], raw_model_info: dict[str, Any]) -> None:
        self.name = name
        self.trainings = trainings
        self.model_info = raw_model_info

from PyADRL.pooling.models.experiment_config import ExperimentConfig


class Experiment:
    def __init__(self, name: str, configs: list[ExperimentConfig]) -> None:
        self.configs = configs
        self.name = name

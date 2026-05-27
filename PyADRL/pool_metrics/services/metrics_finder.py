import json
from logging import Logger

from PyADRL.pool_metrics.models.evaluation_result import (
    EvaluationPoolMetrics,
    EvaluationResult,
)
from PyADRL.pooling.services.experiment_provider import ExperimentProvider


class MetricsFinder:
    def __init__(self, experiment_provider: ExperimentProvider, logger: Logger) -> None:
        self.logger = logger
        self.experiment_provider = experiment_provider

    def scan_for_metrics(self) -> list[EvaluationPoolMetrics]:
        experiments = self.experiment_provider.get_experiments()
        ms: list[EvaluationPoolMetrics] = []

        for experiment in experiments:
            for config in experiment.configs:
                for training in config.trainings:
                    try:
                        for file in training.eval_pool_path.iterdir():
                            if not file.is_dir():
                                continue

                            metrics_fp = file / "evaluation_metrics.json"

                            with open(metrics_fp, "r", encoding="utf-8") as f:
                                data = json.load(f)

                            split_file_name = str(file.name).split("-")

                            results: list[EvaluationResult] = [
                                EvaluationResult(**d) for d in data
                            ]
                            metrics = EvaluationPoolMetrics(
                                experiment_name=experiment.name,
                                pursuer_config=split_file_name[0],
                                pursuer_training=split_file_name[1],
                                evader_config=config.name,
                                evader_training=training.name,
                                metrics=results,
                            )
                            ms.append(metrics)

                            self.logger.debug("Loaded metrics from: %s", metrics_fp)
                    except Exception as e:
                        self.logger.error(
                            "Failed to construct EvaluationResult: %s",
                            e,
                        )

        return ms

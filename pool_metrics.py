import logging

from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject

from PyADRL.pool_metrics.models.evaluation_result import combine
from PyADRL.pool_metrics.services.metrics_finder import MetricsFinder
from PyADRL.pooling.services.config_provider import FSExperimentConfigProvider
from PyADRL.pooling.services.experiment_provider import FSExperimentProvider
from PyADRL.pooling.services.training_provider import FSTrainingProvider


class PoolMetricsContainer(containers.DeclarativeContainer):
    config = providers.Configuration()
    logger = providers.Singleton(lambda: logging.getLogger("default"))

    training_provider = providers.Factory(FSTrainingProvider, logger=logger)
    experiment_config_provider = providers.Factory(
        FSExperimentConfigProvider, logger=logger, training_provider=training_provider
    )
    experiment_provider = providers.Factory(
        FSExperimentProvider, config_provider=experiment_config_provider, logger=logger
    )

    metrics_finder = providers.Factory(
        MetricsFinder, experiment_provider=experiment_provider, logger=logger
    )


@inject
def main(
    _: logging.Logger = Provide[PoolMetricsContainer.logger],
    metrics_finder: MetricsFinder = Provide[PoolMetricsContainer.metrics_finder],
) -> int:
    ms = metrics_finder.scan_for_metrics()

    combined_metrics = combine([metric for results in ms for metric in results.metrics])

    with open("combined_metrics.json", "w") as f:
        f.write(combined_metrics.model_dump_json(indent=2))

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    container = PoolMetricsContainer()
    container.wire(modules=[__name__])

    main()

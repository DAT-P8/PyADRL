import logging
from PyADRL.pool_metrics.services.metrics_finder import MetricsFinder
from PyADRL.pooling.services.map_service import MapService
from dependency_injector.providers import Factory
from dependency_injector import containers, providers
from PyADRL.pooling.models.pool_config import PoolConfig
from PyADRL.pooling.services.config_provider import (
    ExperimentConfigProvider,
    FSExperimentConfigProvider,
)
from PyADRL.pooling.services.evaluation_executor import (
    EvaluationExecutor,
    RayEvaluationExecutor,
)
from PyADRL.pooling.services.experiment_provider import (
    ExperimentProvider,
    FSExperimentProvider,
)
from PyADRL.pooling.services.training_provider import (
    FSTrainingProvider,
    TrainingProvider,
)


class DefaultContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    logger = providers.Singleton(lambda: logging.getLogger("default"))
    pool_config = providers.Factory(
        PoolConfig,
        experiment=config.experiment,
        n_pursuers=config.n_pursuers,
        n_evaders=config.n_evaders,
    )
    training_provider: Factory[TrainingProvider] = providers.Factory(
        FSTrainingProvider, logger=logger
    )
    config_provider: Factory[ExperimentConfigProvider] = providers.Factory(
        FSExperimentConfigProvider, logger=logger, training_provider=training_provider
    )
    experiment_provider: Factory[ExperimentProvider] = providers.Factory(
        FSExperimentProvider, config_provider=config_provider, logger=logger
    )
    map_service: Factory[MapService] = providers.Factory(MapService, logger=logger)
    evaluation_executor: Factory[EvaluationExecutor] = providers.Factory(
        RayEvaluationExecutor,
        map_service=map_service,
        logger=logger,
        evader_key="evader",
        pursuer_key="pursuer",
    )
    metrics_finder = providers.Factory(
        MetricsFinder, experiment_provider=experiment_provider, logger=logger
    )

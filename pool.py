import logging
from PyADRL.pooling.services.evaluation_executor import EvaluationExecutor
from PyADRL.pooling.models.pool_config import PoolConfig
import argparse
import ray
from dependency_injector.wiring import Provide, inject
from PyADRL.pooling.service_configuration import (
    Container,
)
from PyADRL.pooling.services.experiment_provider import ExperimentProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example CLI parser")

    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        required=True,
        default=None,
        help="Please specify experiment to run. E.g. 'experiment_6'",
    )

    return parser.parse_args()


@inject
def main(
    pool_config: PoolConfig = Provide[Container.pool_config],
    experiment_provider: ExperimentProvider = Provide[Container.experiment_provider],
    evaluation_executor: EvaluationExecutor = Provide[Container.evaluation_executor],
    logger: logging.Logger = Provide[Container.logger],
) -> int:
    ray.shutdown()
    ray.init()

    experiment = experiment_provider.get_experiment_by_name(pool_config.experiment)
    if experiment is None:
        raise Exception(f"Did not find experiment with name: {pool_config.experiment}")

    confs = [
        (config, training)
        for config in experiment.configs
        for training in config.trainings
    ]
    combinations = [(c1, t1, c2, t2) for c1, t1 in confs for c2, t2 in confs]

    for c1, t1, c2, t2 in combinations:
        evaluation_executor.evaluate_alternating(c1, t1, c2, t2)

    logger.info("read experiment: %s", experiment.name)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    args = parse_args()

    container = Container()
    container.config.experiment.from_value(args.experiment)
    container.wire(modules=[__name__])

    main()

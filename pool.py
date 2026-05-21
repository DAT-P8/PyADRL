import logging
from PyADRL.pooling.services.evaluation_executor import EvaluationExecutor
from PyADRL.utils.register_env import _register_gridworld_env
from ray.tune.registry import _global_registry, ENV_CREATOR
from PyADRL.envs.reward_functions.grid_world_rewards import GridWorldRewards
from PyADRL.pooling.models.pool_config import PoolConfig
from parser import get_maps
import argparse
import ray
from dependency_injector.wiring import Provide, inject
from PyADRL.pooling.service_configuration import (
    Container,
)
from PyADRL.utils import map_load
from PyADRL.pooling.services.experiment_provider import (
    ExperimentProvider
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example CLI parser")

    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        required=True,
        default=None,
        help="Please specify experiment to run. E.g. 'experiment_6'"
    )


    parser.add_argument(
        "--map",
        type=str,
        default="map",
        required=False,
        help=f"Map name, maps are found in PyADRL/examples/maps. Maps: {get_maps()}",
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

    map_dict = map_load.load_map_dict(pool_config.map)

    # only register the environment if it hasn't been registered
    if not _global_registry.contains(ENV_CREATOR, "gridworld"):
        _register_gridworld_env(
            map_dict=map_dict,
            reward_function=GridWorldRewards(),
            n_pursuers=pool_config.n_pursuers,
            n_evaders=pool_config.n_evaders,
            shielding=False,
        )

    experiment = experiment_provider.get_experiment_by_name(pool_config.experiment)
    if experiment is None:
        raise Exception(f"Did not find experiment with name: {pool_config.experiment}")

    confs = [(config, training) for config in experiment.configs for training in config.trainings]
    combinations = [(c1, t1, c2, t2) for c1, t1 in confs for c2, t2 in confs]

    for c1, t1, c2, t2 in combinations:
        evaluation_executor.evaluate_alternating(c1, t1, c2, t2)

    logger.info("read experiment: %s", experiment.name)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    args = parse_args()

    container = Container()

    container.config.map.from_value(args.map)
    container.config.experiment.from_value(args.experiment)
    container.config.n_pursuers.from_value(2)
    container.config.n_evaders.from_value(1)

    container.wire(modules=[__name__])

    main()

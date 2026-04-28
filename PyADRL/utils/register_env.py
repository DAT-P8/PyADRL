import grpc
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ..envs.reward_functions.rewards import RewardFunction
from ..envs.map_configs.map_config import MapConfig
from ..envs.ngw_env import NGWEnvironment

from ..envs.map_configs.square_map import SquareMapConfig


def _register_gridworld_env(
    map_config: MapConfig, reward_function: RewardFunction, n_pursuers=2, n_evaders=1
) -> None:
    """Register the gridworld environment with Ray. Safe to call multiple times."""
    # We need to figure out a way so we can say map_config=map_config
    register_env(
        "gridworld",
        lambda _cfg: ParallelPettingZooEnv(
            NGWEnvironment(
                channel=grpc.insecure_channel("localhost:50051"),
                map_config=SquareMapConfig(11, 11, 5, 5),
                reward_function=reward_function,
                n_pursuers=n_pursuers,
                n_evaders=n_evaders,
            )
        ),
    )

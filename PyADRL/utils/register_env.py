import grpc
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ..envs.reward_functions.rewards import RewardFunction
from ..envs.ngw_env import NGWEnvironment
from .map_load import load_map_config


def _register_gridworld_env(
    map: str, reward_function: RewardFunction, n_pursuers=2, n_evaders=1
) -> None:
    """Register the gridworld environment with Ray. Safe to call multiple times."""
    map_config = load_map_config(map)
    register_env(
        "gridworld",
        lambda _cfg: ParallelPettingZooEnv(
            NGWEnvironment(
                channel=grpc.insecure_channel("localhost:50051"),
                map_config=map_config,
                reward_function=reward_function,
                n_pursuers=n_pursuers,
                n_evaders=n_evaders,
            )
        ),
    )

import grpc
import ray
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ..envs.ngw_env import NGWEnvironment
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ..envs.reward_functions.grid_world_rewards import GridWorldRewards
from ray.tune.registry import register_env
from ..logger.metrics import MetricsCallback
from ..logger.heatmaps import HeatmapCallback

from ..utils.path_utils import restore_testing
from ..utils.map_load import load_map_config


def gridworld_eval(
    map: str,
    checkpoint_path: str,
    delay: float,
):
    ray.init()

    map_config = load_map_config(map)

    register_env(
        "gridworld",
        lambda cfg: ParallelPettingZooEnv(
            NGWEnvironment(
                channel=grpc.insecure_channel("localhost:50051"),
                map_config=map_config,
                reward_function=GridWorldRewards(),
                n_pursuers=2,
                n_evaders=1,
                step_delay=delay,
                shielded=True,
            )
        ),
    )

    config: PPOConfig = (
        PPOConfig()
        .environment(
            "gridworld",
            env_config={
                "map_width": map_config.width,
                "map_height": map_config.height,
                "target_x": map_config.target_x,
                "target_y": map_config.target_y,
                "model_name": checkpoint_path,
            },
        )
        .multi_agent(
            policies={"pursuer_policy", "evader_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: (
                "pursuer_policy" if "pursuer" in str(agent_id) else "evader_policy"
            ),
        )
        .env_runners(
            num_env_runners=1,
        )
        .callbacks(callbacks_class=[MetricsCallback, HeatmapCallback])
        .evaluation(evaluation_num_env_runners=1)
    )

    algo = config.build()
    restore_testing(algo, checkpoint_path)

    algo.evaluate()

    algo.stop()

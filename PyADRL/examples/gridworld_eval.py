import grpc
import ray
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ..envs.ngw_env import NGWEnvironment
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ..envs.map_configs.square_map import SquareMapConfig
from ..envs.reward_functions.grid_world_rewards import GridWorldRewards
from ray.tune.registry import register_env
from ..logger.metricslogger import (
    MetricsCallback,
    build_eval,
    build_eval_data,
    metrics_path,
    print_eval_summary,
    write_metrics,
)
from ..logger.heatmaps import HeatmapCallback

from ..utils.model_save import restore_testing


def gridworld_eval(
    checkpoint_path: str, width: int, height: int, target_x: int, target_y: int
):
    ray.init()

    register_env(
        "gridworld",
        lambda cfg: ParallelPettingZooEnv(
            NGWEnvironment(
                channel=grpc.insecure_channel("localhost:50051"),
                map_config=SquareMapConfig(width, height, target_x, target_y),
                reward_function=GridWorldRewards(),
                n_pursuers=2,
                n_evaders=1,
            )
        ),
    )

    config: PPOConfig = (
        PPOConfig()
        .environment(
            "gridworld",
            env_config={
                "map_width": width,
                "map_height": height,
                "target_x": target_x,
                "target_y": target_y,
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

    results = algo.evaluate()
    eval_data = build_eval_data(results)
    eval_episodes = eval_data.get("episodes", [])
    eval_metrics_path = metrics_path("eval")
    write_metrics(
        eval_metrics_path,
        build_eval(eval_episodes, fallback_summary=eval_data.get("summary", {})),
    )
    print_eval_summary(eval_data, eval_metrics_path)

    algo.stop()

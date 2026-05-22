from abc import ABCMeta, abstractmethod
from PyADRL.envs.reward_functions.grid_world_rewards import GridWorldRewards
from PyADRL.utils.register_env import _register_gridworld_env
from logging import Logger
from typing import override

from PyADRL.logger.heatmaps import HeatmapCallback
from PyADRL.logger.metrics import MetricsCallback
from PyADRL.pooling.models.experiment_config import ExperimentConfig
from PyADRL.pooling.models.training import Training
from PyADRL.pooling.services.map_service import MapService
from PyADRL.utils import config_builder, map_load


class EvaluationExecutor(metaclass=ABCMeta):
    @abstractmethod
    def evaluate_alternating(
        self, c1: ExperimentConfig, t1: Training, c2: ExperimentConfig, t2: Training
    ):
        raise NotImplementedError("abstract method")


class RayEvaluationExecutor(EvaluationExecutor):
    def __init__(
        self, map_service: MapService, logger: Logger, evader_key: str, pursuer_key: str
    ) -> None:
        super().__init__()
        self.evader_key = evader_key
        self.pursuer_key = pursuer_key
        self.logger = logger
        self.map_service = map_service

    @override
    def evaluate_alternating(
        self, c1: ExperimentConfig, t1: Training, c2: ExperimentConfig, t2: Training
    ):
        callbacks = [MetricsCallback, HeatmapCallback]

        self.logger.debug(
            "Building algo1 to extract evader weights: %s", c1.name + "-" + t1.name
        )
        map1 = self.map_service.get_from_name(c1.model_info["map"])
        map2 = self.map_service.get_from_name(c2.model_info["map"])
        assert map1 is not None, f"did not find map of: {c1.model_info['map']}"
        assert map2 is not None, f"did not find map of: {c2.model_info['map']}"

        n_pursuers = int(c1.model_info["n_pursuers"])
        n_evaders = int(c1.model_info["n_evaders"])

        assert n_pursuers == int(c2.model_info["n_pursuers"])
        assert n_evaders == int(c2.model_info["n_evaders"])

        map_dict = map_load.load_map_dict(c1.model_info["map"])
        # only register the environment if it hasn't been registered
        _register_gridworld_env(
            map_dict=map_dict,
            reward_function=GridWorldRewards(),
            n_pursuers=n_pursuers,
            n_evaders=n_evaders,
            shielding=False,
        )

        o1 = "[" + str.join(", ", [f"({o.x}, {o.y})" for o in map1.objects]) + "]"
        o2 = "[" + str.join(", ", [f"({o.x}, {o.y})" for o in map2.objects]) + "]"
        self.logger.debug(
            "map1 D: (%s, %s), T: (%s, %s), O: %s",
            map1.width,
            map1.height,
            map1.target_x,
            map1.target_y,
            o1,
        )
        self.logger.debug(
            "map2 D: (%s, %s), T: (%s, %s), O: %s",
            map2.width,
            map2.height,
            map2.target_x,
            map2.target_y,
            o2,
        )
        self.logger.debug("map1 %s", c1.model_info["map"])
        self.logger.debug("map2 %s", c2.model_info["map"])

        fig_path = t1.eval_pool_path / (c2.name + "-" + t2.name)

        ppo_config1 = config_builder._build_ppo_config(
            config=c1.model_info,
            callbacks=callbacks,
            env_config={
                "width": map1.width,
                "height": map1.height,
                "target_x": map1.target_x,
                "target_y": map1.target_y,
                "objects": map1.objects,
                "figure_path": fig_path,
                "n_evaders": n_evaders,
            },
            n_pursuers=n_pursuers,
            n_evaders=n_evaders,
            figure_path=fig_path,
            metrics_path=fig_path,
        )
        algo1 = ppo_config1.build_algo()
        algo1.restore(str(t1.final_model_path))

        assert algo1.learner_group is not None, (
            "algo1 learner_group is None after restore"
        )
        evader_weights = algo1.learner_group.get_weights()[f"{self.evader_key}_policy"]
        self.logger.debug(
            "Extracted evader weights from algo1 %s, stopping it",
            c1.name + "-" + t1.name,
        )

        # need to stop this one before loading algo2
        algo1.stop()

        # --- Build and restore algo2 (pursuer side) ---
        self.logger.debug("Building algo2: %s", c2.name + "-" + t2.name)

        shallow_c_dict = {k: v for k, v in c2.model_info.items()}
        shallow_c_dict["evaluation_duration"] = 1000  # extra iterations
        ppo_config2 = config_builder._build_ppo_config(
            config=c2.model_info,
            callbacks=callbacks,
            env_config={
                "width": map2.width,
                "height": map2.height,
                "target_x": map2.target_x,
                "target_y": map2.target_y,
                "objects": map2.objects,
                "figure_path": fig_path,
                "n_evaders": n_evaders,
            },
            n_pursuers=n_pursuers,
            n_evaders=n_evaders,
            figure_path=fig_path,
            metrics_path=fig_path,
        )
        algo2 = ppo_config2.build_algo()
        algo2.restore(str(t2.final_model_path))

        assert algo2.env_runner is not None
        assert algo2.learner_group is not None, (
            "algo2 learner_group is None after restore"
        )

        # some issue with env not being registered, so we call this to make sure
        algo2.env_runner.make_env()

        # put algo1's evader weights into algo2
        self.logger.debug(
            "Transplanting algo1 evader weights into algo2 %s", c2.name + "-" + t2.name
        )
        algo2.learner_group.set_weights({f"{self.evader_key}_policy": evader_weights})

        assert algo2.config is not None
        assert algo2.config.num_env_runners is not None

        # make sure to sync weights with remote workers
        if algo2.env_runner_group is not None and algo2.config.num_env_runners > 0:
            algo2.env_runner_group.sync_weights(
                from_worker_or_learner_group=algo2.learner_group,
                policies=[f"{self.evader_key}_policy"],
            )

        # this is *magic*
        algo2.config._is_frozen = False
        algo2.config.multi_agent(policies_to_train=[f"{self.pursuer_key}_policy"])
        algo2.config._is_frozen = True
        algo2.learner_group.foreach_learner(
            lambda learner, *_args: learner.config.multi_agent(
                policies_to_train=[f"{self.pursuer_key}_policy"]
            )
        )

        eval_result = algo2.evaluate()

        algo2.stop()
        return eval_result

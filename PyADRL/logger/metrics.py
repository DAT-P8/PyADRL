from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import numpy as np

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger


@dataclass
class EpisodeOutcome:
    capture_steps: list[int] = field(default_factory=list)
    breached: bool = False
    episode_length: int = 0
    evader_drone_collision_rate: float = 0.0
    pursuer_drone_collision_rate: float = 0.0
    evader_obstacle_collision_rate: float = 0.0
    pursuer_obstacle_collision_rate: float = 0.0
    pursuer_entered_target_count: int = 0
    pursuer_out_of_bounds_rate: float = 0.0
    evader_out_of_bounds_rate: float = 0.0
    pursuer_shield_intervention_rate: float = 0.0
    evader_shield_intervention_rate: float = 0.0


def extract_episode_metrics(infos) -> dict | None:
    if not isinstance(infos, dict):
        return None

    for val in infos.values():
        if isinstance(val, dict) and "episode_metrics" in val:
            return val["episode_metrics"]

    if "episode_metrics" in infos:
        return infos["episode_metrics"]
    return None


def mean_capture_steps(episode_outcomes: list[dict]) -> list[float]:
    capture_step_sequences = [
        outcome.get("capture_steps", []) for outcome in episode_outcomes
    ]
    max_length = max((len(sequence) for sequence in capture_step_sequences), default=0)

    mean_steps: list[float] = []
    for index in range(max_length):
        values = [
            sequence[index]
            for sequence in capture_step_sequences
            if len(sequence) > index
        ]
        if values:
            mean_steps.append(float(np.mean(values)))

    return mean_steps


def mean_capture(episode_outcomes: list[dict]):
    capture_steps = [outcome.get("capture_steps", []) for outcome in episode_outcomes]

    # filter out episodes with no captures
    capture_steps = [np.mean(steps) for steps in capture_steps if len(steps) > 0]
    return np.mean(capture_steps) if capture_steps else -1.0


def capture_rate_at_k(episode_outcomes: list[dict]) -> dict[int, float]:
    capture_step_sequences = [
        outcome.get("capture_steps", []) for outcome in episode_outcomes
    ]

    # Count how many episodes had k captures for each k e.g. {0: 30, 1: 5, 2: 15}
    capture_rates = {}
    for captures in capture_step_sequences:
        capture_rates[len(captures)] = capture_rates.get(len(captures), 0) + 1

    # normalize from the number of episodes to get a rate e.g. {0: 0.6, 1: 0.1, 2: 0.3}
    total_episodes = len(episode_outcomes)
    for k in capture_rates:
        capture_rates[k] /= total_episodes

    return capture_rates


def rate(ids, role_ids, n):
    return len([i for i in ids if i in role_ids]) / n if n > 0 else 0.0


class MetricsCallback(RLlibCallback):
    def __init__(self):
        super().__init__()
        self.metrics_path = None
        self.n_pursuers = 0
        self.n_evaders = 0
        self.episode_outcomes: list[EpisodeOutcome] = []
        self.capture_steps: list[int] = []  # Track steps at which evaders are captured
        self.captured_evader_ids: set[int] = set()  # evaders that have been captured
        self.target_reached_ids: set[int] = set()
        self.pursuer_entered_target_count: int = 0
        self.drone_object_collision_ids: set[int] = set()
        self.drone_out_of_bounds_ids: set[int] = set()
        self.collision_ids: set[int] = set()
        self.evader_ids: set[int] = set()
        self.pursuer_ids: set[int] = set()
        self.evader_shield_interventions: int = 0
        self.pursuer_shield_interventions: int = 0
        self.timestep: int = 0

    def on_algorithm_init(
        self, *, algorithm, metrics_logger: MetricsLogger | None = None, **kwargs
    ) -> None:
        if algorithm.config is None or algorithm.config.env_config is None:
            raise ValueError(
                "MetricsCallback requires env_config to be set in the algorithm config"
            )
        self.metrics_path = algorithm.config.env_config.get("metrics_path")
        self.n_pursuers = algorithm.config.env_config.get("n_pursuers", 0)
        self.n_evaders = algorithm.config.env_config.get("n_evaders", 0)

    def on_episode_end(
        self,
        *,
        episode,
        env_runner=None,
        metrics_logger=None,
        env=None,
        env_index,
        rl_module=None,
        **kwargs,
    ) -> None:
        n_evaders = len(self.evader_ids)
        n_pursuers = len(self.pursuer_ids)

        outcome_obj = EpisodeOutcome(
            capture_steps=list(self.capture_steps),
            breached=len(self.target_reached_ids) > 0,
            episode_length=self.timestep,
            evader_drone_collision_rate=rate(
                self.collision_ids, self.evader_ids, n_evaders
            ),
            pursuer_drone_collision_rate=rate(
                self.collision_ids, self.pursuer_ids, n_pursuers
            ),
            evader_obstacle_collision_rate=rate(
                self.drone_object_collision_ids, self.evader_ids, n_evaders
            ),
            pursuer_obstacle_collision_rate=rate(
                self.drone_object_collision_ids, self.pursuer_ids, n_pursuers
            ),
            pursuer_entered_target_count=self.pursuer_entered_target_count,
            evader_out_of_bounds_rate=rate(
                self.drone_out_of_bounds_ids, self.evader_ids, n_evaders
            ),
            pursuer_out_of_bounds_rate=rate(
                self.drone_out_of_bounds_ids, self.pursuer_ids, n_pursuers
            ),
            evader_shield_intervention_rate=self.evader_shield_interventions
            / n_evaders
            / self.timestep
            if self.timestep > 0 and n_evaders > 0
            else 0.0,
            pursuer_shield_intervention_rate=self.pursuer_shield_interventions
            / n_pursuers
            / self.timestep
            if self.timestep > 0 and n_pursuers > 0
            else 0.0,
        )
        self.episode_outcomes.append(outcome_obj)
        if metrics_logger is not None:
            metrics_logger.log_value(
                "episode_outcomes", asdict(outcome_obj), reduce="item_series"
            )

    def _get_episode_info(self, env, env_index: int):
        if not env or not getattr(env, "_infos", None):
            return None

        if isinstance(env._infos, (list, tuple)):
            if 0 <= env_index < len(env._infos):
                return env._infos[env_index]
            return env._infos[0] if env._infos else None

        return env._infos

    def on_episode_start(
        self,
        *,
        episode,
        env_runner=None,
        metrics_logger=None,
        env=None,
        env_index,
        rl_module=None,
        **kwargs,
    ):
        # Reset episode-specific tracking variables
        self.capture_steps = []
        self.captured_evader_ids = set()
        self.target_reached_ids = set()
        self.pursuer_entered_target_count = 0
        self.evader_shield_interventions = 0
        self.pursuer_shield_interventions = 0
        self.drone_object_collision_ids = set()
        self.drone_out_of_bounds_ids = set()
        self.collision_ids = set()
        self.timestep = 0
        self.evader_ids = set()
        self.pursuer_ids = set()
        infos = self._get_episode_info(env, env_index)
        if infos is not None:
            self.evader_ids.update(
                [d["drone"].id for d in infos.values() if d["drone"].is_evader]
            )
            self.pursuer_ids.update(
                [d["drone"].id for d in infos.values() if not d["drone"].is_evader]
            )

    def on_episode_step(
        self,
        *,
        episode,
        env_runner=None,
        metrics_logger=None,
        env=None,
        env_index,
        rl_module=None,
        **kwargs,
    ):
        episode_info = self._get_episode_info(env, env_index)
        if episode_info is None:
            return
        self.timestep += 1
        agent_ids = list(episode_info.keys())

        shield_drone_collision_ids = set()

        # count shield interventions by iterating alt_state events
        for event in episode_info[agent_ids[0]].get("shield_events", []):
            if event.drone_object_collision_event is not None:
                self.evader_shield_interventions += len(
                    [
                        id
                        for id in event.drone_object_collision_event.drone_ids
                        if id in self.evader_ids
                    ]
                )
                self.pursuer_shield_interventions += len(
                    [
                        id
                        for id in event.drone_object_collision_event.drone_ids
                        if id in self.pursuer_ids
                    ]
                )
            if event.out_of_bounds_event is not None:
                self.evader_shield_interventions += len(
                    [
                        id
                        for id in event.out_of_bounds_event.drone_ids
                        if id in self.evader_ids
                    ]
                )
                self.pursuer_shield_interventions += len(
                    [
                        id
                        for id in event.out_of_bounds_event.drone_ids
                        if id in self.pursuer_ids
                    ]
                )
            if event.collision_event is not None:
                shield_drone_collision_ids.update(event.collision_event.drone_ids)

        collision_events: list[set[int]] = []
        actual_drone_collision_ids = set()

        for event in episode_info[agent_ids[0]].get("events", []):
            if event.drone_object_collision_event is not None:
                self.drone_object_collision_ids.update(
                    event.drone_object_collision_event.drone_ids
                )
            elif event.pursuer_entered_target_event is not None:
                self.pursuer_entered_target_count += len(
                    event.pursuer_entered_target_event.drone_ids
                )
            elif event.target_reached_event is not None:
                self.target_reached_ids.update(event.target_reached_event.drone_ids)
            elif event.collision_event is not None:
                collision_events.append(set(event.collision_event.drone_ids))
                actual_drone_collision_ids.update(event.collision_event.drone_ids)
            elif event.out_of_bounds_event is not None:
                self.drone_out_of_bounds_ids.update(event.out_of_bounds_event.drone_ids)

        # Count shield interventions for collisions, but only for drones "saved" by the shield
        shield_drone_collision_ids = shield_drone_collision_ids.difference(
            actual_drone_collision_ids
        )
        self.evader_shield_interventions += len(
            [id for id in shield_drone_collision_ids if id in self.evader_ids]
        )
        self.pursuer_shield_interventions += len(
            [id for id in shield_drone_collision_ids if id in self.pursuer_ids]
        )

        # Check for captures:
        evaders_caught: set[int] = set()
        for coll_set in collision_events:
            evaders_in_coll = [x for x in coll_set if x in self.evader_ids]
            pursuers_in_coll = [x for x in coll_set if x in self.pursuer_ids]

            # any collision event that includes at least one evader and one pursuer
            if evaders_in_coll and pursuers_in_coll:
                evaders_caught.update(evaders_in_coll)
            # Count collisions only for events with one team.
            # Mixed-team events are capture events and are excluded from collision_ids.
            elif evaders_in_coll and not pursuers_in_coll:
                self.collision_ids.update(evaders_in_coll)
            elif pursuers_in_coll and not evaders_in_coll:
                self.collision_ids.update(pursuers_in_coll)

        for evader_id in evaders_caught:
            if evader_id not in self.captured_evader_ids:
                self.capture_steps.append(self.timestep)
                self.captured_evader_ids.add(evader_id)

    def on_evaluate_start(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger: MetricsLogger | None = None,
        **kwargs,
    ) -> None:
        # Reset for next evaluation
        self.episode_outcomes = []

    def on_evaluate_end(
        self,
        *,
        algorithm,
        metrics_logger=None,
        evaluation_metrics: dict | None = None,
        **kwargs,
    ):
        if evaluation_metrics is None:
            evaluation_metrics = kwargs.get("result")
        if evaluation_metrics is None:
            raise ValueError("No evaluation results to process.")

        env_runners = evaluation_metrics.get("env_runners", {})
        rewards_dict = env_runners.get("agent_episode_returns_mean", {})
        episode_outcomes = env_runners.get("episode_outcomes")
        if not episode_outcomes:
            episode_outcomes = [asdict(outcome) for outcome in self.episode_outcomes]

        if not episode_outcomes:
            print("No episode outcomes to evaluate.")
            return

        outcomes_array = np.array(
            [
                [
                    float(outcome["breached"]),
                    outcome["episode_length"],
                    outcome["evader_drone_collision_rate"],
                    outcome["pursuer_drone_collision_rate"],
                    outcome["evader_obstacle_collision_rate"],
                    outcome["pursuer_obstacle_collision_rate"],
                    outcome["evader_out_of_bounds_rate"],
                    outcome["pursuer_out_of_bounds_rate"],
                    outcome["evader_shield_intervention_rate"],
                    outcome["pursuer_shield_intervention_rate"],
                ]
                for outcome in episode_outcomes
            ]
        )

        means = np.mean(outcomes_array, axis=0)
        mean_steps = mean_capture_steps(episode_outcomes)
        capture_rates = capture_rate_at_k(episode_outcomes)
        average_capture_step = mean_capture(episode_outcomes)

        mean_summary = {
            "timestamp": datetime.now().isoformat(),
            "capture_rate_at_k": capture_rates,
            "mean_capture_step_at_k": mean_steps,
            "mean_capture_step": average_capture_step,
            "mean_pursuer_entered_target_count": float(
                np.mean(
                    [
                        outcome.get("pursuer_entered_target_count", 0)
                        for outcome in episode_outcomes
                    ]
                )
            ),
            "breach_rate": float(means[0]),
            "mean_episode_length": float(means[1]),
            "mean_evader_drone_collision_rate": float(means[2]),
            "mean_pursuer_drone_collision_rate": float(means[3]),
            "mean_evader_obstacle_collision_rate": float(means[4]),
            "mean_pursuer_obstacle_collision_rate": float(means[5]),
            "mean_evader_out_of_bounds_rate": float(means[6]),
            "mean_pursuer_out_of_bounds_rate": float(means[7]),
            "evader_shield_intervention_rate": float(means[8]),
            "pursuer_shield_intervention_rate": float(means[9]),
            "mean_rewards": rewards_dict,
        }

        if self.metrics_path:
            results_dir = self.metrics_path / "evaluation_metrics.json"

            # Load existing metrics or create new list
            metrics_list = []
            if results_dir.exists():
                with open(results_dir, "r") as f:
                    metrics_list = json.load(f)

            # Append new metrics
            metrics_list.append(mean_summary)

            # Write updated metrics
            with open(results_dir, "w") as f:
                json.dump(metrics_list, f, indent=4)
            print(f"Saved evaluation metrics to {results_dir}")

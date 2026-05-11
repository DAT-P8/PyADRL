from dataclasses import asdict, dataclass, field

import numpy as np

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

EVADERS = "evaders"
PURSUERS = "pursuers"


@dataclass
class EpisodeOutcome:
    capture_rate: float = 0.0
    capture_steps: list[int] = field(default_factory=list)
    breached: bool = False
    episode_length: int = 0
    evader_drone_collision_rate: float = 0.0
    pursuer_drone_collision_rate: float = 0.0
    evader_obstacle_collision_rate: float = 0.0
    pursuer_obstacle_collision_rate: float = 0.0
    pursuer_entered_target_count: int = 0
    n_evaders: int = 0
    n_pursuers: int = 0
    evader_ids: list[int] = field(default_factory=list)
    pursuer_ids: list[int] = field(default_factory=list)
    collision_ids: list[int] = field(default_factory=list)
    drone_object_collision_ids: list[int] = field(default_factory=list)


def compute_episode_metrics(
    state,
    drones,
    timestep: int,
    n_evaders: int,
    n_pursuers: int,
    capture_steps: list[int],
    captured_evader_ids: set[int],
    target_reached_ids: set[int] | None = None,
    pursuer_entered_target_count: int = 0,
    drone_object_collision_ids: set[int] | None = None,
    cumulative_collision_ids: set[int] | None = None,
) -> EpisodeOutcome:
    target_reached_set: set[int] = (
        target_reached_ids if target_reached_ids is not None else set()
    )
    drone_object_collision_set: set[int] = (
        drone_object_collision_ids if drone_object_collision_ids is not None else set()
    )
    collision_id_set: set[int] = (
        cumulative_collision_ids if cumulative_collision_ids is not None else set()
    )
    collision_events: list[set[int]] = []

    for event in state.events:
        if event.drone_object_collision_event is not None:
            drone_object_collision_set.update(
                event.drone_object_collision_event.drone_ids
            )
        elif event.pursuer_entered_target_event is not None:
            pursuer_entered_target_count += len(
                event.pursuer_entered_target_event.drone_ids
            )
        elif event.target_reached_event is not None:
            target_reached_set.update(event.target_reached_event.drone_ids)
        elif event.collision_event is not None:
            collision_events.append(set(event.collision_event.drone_ids))

    evader_ids = {d.id for d in drones[EVADERS]}
    pursuer_ids = {d.id for d in drones[PURSUERS]}

    # Check for captures:
    # any collision event that includes at least one evader and one pursuer
    evaders_caught: set[int] = set()
    for coll_set in collision_events:
        evaders_in_coll = [x for x in coll_set if x in evader_ids]
        pursuers_in_coll = [x for x in coll_set if x in pursuer_ids]
        if evaders_in_coll and pursuers_in_coll:
            evaders_caught.update(evaders_in_coll)

    for evader_id in evaders_caught:
        if evader_id not in captured_evader_ids:
            capture_steps.append(timestep)
            captured_evader_ids.add(evader_id)

    # Check for collisions
    for coll_set in collision_events:
        evaders_in_coll = [x for x in coll_set if x in evader_ids]
        pursuers_in_coll = [x for x in coll_set if x in pursuer_ids]

        # Count collisions only for events with one team.
        # Mixed-team events are capture events and are excluded from collision_ids.
        if evaders_in_coll and not pursuers_in_coll:
            collision_id_set.update(evaders_in_coll)
        elif pursuers_in_coll and not evaders_in_coll:
            collision_id_set.update(pursuers_in_coll)

    return EpisodeOutcome(
        capture_steps=list(capture_steps),
        breached=len(target_reached_set) > 0,
        episode_length=int(timestep),
        pursuer_entered_target_count=int(pursuer_entered_target_count),
        n_evaders=int(n_evaders),
        n_pursuers=int(n_pursuers),
        evader_ids=sorted(evader_ids),
        pursuer_ids=sorted(pursuer_ids),
        collision_ids=sorted(collision_id_set),
        drone_object_collision_ids=sorted(drone_object_collision_set),
    )


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

def capture_rate_at_k(episode_outcomes: list[dict]) -> dict[int, float]:
    capture_step_sequences = [
        outcome.get("capture_steps", []) for outcome in episode_outcomes
    ]

    capture_rates = {}
    for captures in capture_step_sequences:
        capture_rates[len(captures)] = capture_rates.get(len(captures), 0) + 1

    total_episodes = len(episode_outcomes)
    print(f"Total episodes: {total_episodes} Capture rates: {capture_rates}")

    # normalize to get capture rate at each k
    for k in capture_rates:
        capture_rates[k] /= total_episodes

    return capture_rates


class MetricsCallback(RLlibCallback):
    def __init__(self):
        super().__init__()
        self.episode_outcomes: list[EpisodeOutcome] = []

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
        metrics = extract_episode_metrics(episode.get_infos(-1))
        print(f"Episode ended. Extracted metrics: {metrics}")
        if metrics is None:
            return

        evader_ids = set(metrics.get("evader_ids", []))
        pursuer_ids = set(metrics.get("pursuer_ids", []))
        collision_id_set = set(metrics.get("collision_ids", []))
        drone_object_collision_set = set(metrics.get("drone_object_collision_ids", []))

        n_evaders = int(metrics.get("n_evaders", 0))
        n_pursuers = int(metrics.get("n_pursuers", 0))
        capture_steps = list(metrics.get("capture_steps", []))

        evader_collided = len([i for i in collision_id_set if i in evader_ids])
        pursuer_collided = len([i for i in collision_id_set if i in pursuer_ids])

        evader_obj_collided = len(
            [i for i in drone_object_collision_set if i in evader_ids]
        )
        pursuer_obj_collided = len(
            [i for i in drone_object_collision_set if i in pursuer_ids]
        )

        evader_drone_collision_rate = (
            evader_collided / n_evaders if n_evaders > 0 else 0.0
        )
        pursuer_drone_collision_rate = (
            pursuer_collided / n_pursuers if n_pursuers > 0 else 0.0
        )

        evader_obstacle_collision_rate = (
            evader_obj_collided / n_evaders if n_evaders > 0 else 0.0
        )
        pursuer_obstacle_collision_rate = (
            pursuer_obj_collided / n_pursuers if n_pursuers > 0 else 0.0
        )

        capture_rate = len(capture_steps) / n_evaders if n_evaders > 0 else 0.0

        metrics["capture_rate"] = float(capture_rate)
        metrics["evader_drone_collision_rate"] = float(evader_drone_collision_rate)
        metrics["pursuer_drone_collision_rate"] = float(pursuer_drone_collision_rate)
        metrics["evader_obstacle_collision_rate"] = float(
            evader_obstacle_collision_rate
        )
        metrics["pursuer_obstacle_collision_rate"] = float(
            pursuer_obstacle_collision_rate
        )

        outcome_obj = EpisodeOutcome(**metrics)
        self.episode_outcomes.append(outcome_obj)
        if metrics_logger is not None:
            metrics_logger.log_value(
                "episode_outcomes",
                asdict(outcome_obj),
                reduce="item_series",
            )

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
                ]
                for outcome in episode_outcomes
            ]
        )

        means = np.mean(outcomes_array, axis=0)
        mean_steps = mean_capture_steps(episode_outcomes)
        capture_rates = capture_rate_at_k(episode_outcomes)

        mean_summary = {
            "capture_rate_at_k": capture_rates,
            "mean_capture_steps": mean_steps,
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
            "mean_rewards": rewards_dict,
        }

        print("Eval summary:")
        for key, value in mean_summary.items():
            print(f"{key}: {value}")

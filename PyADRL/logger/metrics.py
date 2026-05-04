from dataclasses import asdict, dataclass, field
import os

import numpy as np

from ray.rllib.callbacks.callbacks import RLlibCallback

_RESULTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
)

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


def compute_episode_metrics(
    state,
    drones,
    timestep: int,
    n_evaders: int,
    n_pursuers: int,
    capture_steps: list[int],
    captured_evader_ids: set[int],
) -> EpisodeOutcome:
    target_reached_set: set[int] = set()
    drone_object_collision_set: set[int] = set()
    collision_dict: dict[int, set[int]] = {}

    for event in state.events:
        if event.drone_object_collision_event is not None:
            drone_object_collision_set.update(event.drone_object_collision_event.drone_ids)
        elif event.target_reached_event is not None:
            target_reached_set.update(event.target_reached_event.drone_ids)
        elif event.collision_event is not None:
            for id_key in event.collision_event.drone_ids:
                if id_key in collision_dict:
                    collision_dict[id_key].update(event.collision_event.drone_ids)
                else:
                    collision_dict[id_key] = set(event.collision_event.drone_ids)

    evader_ids = {d.id for d in drones[EVADERS]}
    pursuer_ids = {d.id for d in drones[PURSUERS]}

    evaders_caught: set[int] = set()
    for _, coll_set in collision_dict.items():
        evaders_in_coll = [x for x in coll_set if x in evader_ids]
        pursuers_in_coll = [x for x in coll_set if x in pursuer_ids]
        if evaders_in_coll and pursuers_in_coll:
            evaders_caught.update(evaders_in_coll)

    for evader_id in evaders_caught:
        if evader_id not in captured_evader_ids:
            capture_steps.append(timestep)
            captured_evader_ids.add(evader_id)

    collision_ids: set[int] = set()
    for key, collision_set in collision_dict.items():
        collision_ids.add(key)
        collision_ids.update(collision_set)

    evader_collided = len([i for i in collision_ids if i in evader_ids])
    pursuer_collided = len([i for i in collision_ids if i in pursuer_ids])

    evader_obj_collided = len([i for i in drone_object_collision_set if i in evader_ids])
    pursuer_obj_collided = len([i for i in drone_object_collision_set if i in pursuer_ids])

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

    return EpisodeOutcome(
        capture_rate=float(capture_rate),
        capture_steps=list(capture_steps),
        breached=len(target_reached_set) > 0,
        episode_length=int(timestep),
        evader_drone_collision_rate=float(evader_drone_collision_rate),
        pursuer_drone_collision_rate=float(pursuer_drone_collision_rate),
        evader_obstacle_collision_rate=float(evader_obstacle_collision_rate),
        pursuer_obstacle_collision_rate=float(pursuer_obstacle_collision_rate),
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
    capture_step_sequences = [outcome.get("capture_steps", []) for outcome in episode_outcomes]
    max_length = max((len(sequence) for sequence in capture_step_sequences), default=0)

    mean_steps: list[float] = []
    for index in range(max_length):
        values = [sequence[index] for sequence in capture_step_sequences if len(sequence) > index]
        if values:
            mean_steps.append(float(np.mean(values)))

    return mean_steps


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
        outcome_obj = EpisodeOutcome(**metrics)
        self.episode_outcomes.append(outcome_obj)
        if metrics_logger is not None:
            metrics_logger.log_value(
                "episode_outcomes",
                asdict(outcome_obj),
                reduce="item_series",
            )


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

        outcomes_array = np.array([
            [
                outcome["capture_rate"],
                float(outcome["breached"]),
                outcome["episode_length"],
                outcome["evader_drone_collision_rate"],
                outcome["pursuer_drone_collision_rate"],
                outcome["evader_obstacle_collision_rate"],
                outcome["pursuer_obstacle_collision_rate"],
            ]
            for outcome in episode_outcomes
        ])

        means = np.mean(outcomes_array, axis=0)
        mean_steps = mean_capture_steps(episode_outcomes)

        mean_summary = {
            "mean_capture_rate": float(means[0]),
            "mean_capture_steps": mean_steps,
            "mean_breached": float(means[1]),
            "mean_episode_length": float(means[2]),
            "mean_evader_drone_collision_rate": float(means[3]),
            "mean_pursuer_drone_collision_rate": float(means[4]),
            "mean_evader_obstacle_collision_rate": float(means[5]),
            "mean_pursuer_obstacle_collision_rate": float(means[6]),
            "mean_rewards": rewards_dict,
        }
        
        print("Eval summary:")
        for key, value in mean_summary.items():
            print(f"{key}: {value}")

        # Reset for next evaluation
        self.episode_outcomes = []
        
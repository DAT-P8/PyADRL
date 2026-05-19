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


def summarize_evaluation(eval_result: dict, n_evaders: int) -> dict:
    """Flatten an algo.evaluate() result into a dict of scalar metrics for tune.report.

    Pulls the per-episode outcomes that MetricsCallback logs via
    metrics_logger.log_value("episode_outcomes", ..., reduce="item_series")
    and aggregates them into single numbers suitable for Tune's metric tracking
    and ASHA scheduling.

    Args:
        eval_result: Return value of algo.evaluate().
        n_evaders: Number of evaders per episode. Used to determine what counts
            as a "full" capture (all evaders caught).

    Returns:
        Flat dict of scalar metrics. Always contains 'mean_reward' so existing
        ASHA configs keep working unchanged.
    """
    env_runners = eval_result.get("env_runners", {}) or {}
    rewards_dict = env_runners.get("agent_episode_returns_mean", {}) or {}
    episode_outcomes = env_runners.get("episode_outcomes") or []

    if isinstance(rewards_dict, dict):
        mean_reward = float(sum(rewards_dict.values()))
        pursuer_reward = float(
            sum(v for k, v in rewards_dict.items() if "pursuer" in str(k))
        )
        evader_reward = float(
            sum(v for k, v in rewards_dict.items() if "evader" in str(k))
        )
    else:
        mean_reward = float(rewards_dict)
        pursuer_reward = 0.0
        evader_reward = 0.0

    # Defaults for the case where no episodes completed in this eval window.
    full_capture_rate = 0.0
    any_capture_rate = 0.0
    breach_rate = 0.0
    mean_episode_length = 0.0
    mean_capture_step = -1.0

    if episode_outcomes:
        capture_rates = capture_rate_at_k(episode_outcomes)
        # Fraction of episodes where every evader was caught.
        full_capture_rate = float(capture_rates.get(n_evaders, 0.0))
        # Fraction of episodes where at least one capture happened.
        any_capture_rate = float(sum(v for k, v in capture_rates.items() if k > 0))

        breach_rate = float(
            np.mean([float(o.get("breached", False)) for o in episode_outcomes])
        )
        mean_episode_length = float(
            np.mean([o.get("episode_length", 0) for o in episode_outcomes])
        )
        mean_capture_step = float(mean_capture(episode_outcomes))

    return {
        "mean_reward": mean_reward,
        # Per-side rewards — useful when you care about one role specifically.
        "pursuer_reward": pursuer_reward,
        "evader_reward": evader_reward,
        # Task-grounded metrics — bounded in [0, 1], reward-shaping-invariant.
        "full_capture_rate": full_capture_rate,
        "any_capture_rate": any_capture_rate,
        "breach_rate": breach_rate,
        "mean_episode_length": mean_episode_length,
        "mean_capture_step": mean_capture_step,
        # Composite: positive means pursuers winning, negative means evaders winning.
        # Useful as an ASHA metric when you care about pursuer success specifically.
        "pursuer_success": full_capture_rate - breach_rate,
    }


class MetricsCallback(RLlibCallback):
    # Per-episode state lives in episode.custom_data["state"], NOT on self.
    # With vectorized environments (num_envs_per_env_runner > 1), many
    # episodes run concurrently on the same callback instance. Storing
    # per-episode tracking on self.X would mix events across episodes.
    # Cross-episode state (the accumulated episode_outcomes buffer, the
    # config values from env_config) is correctly kept on self.

    def __init__(self):
        super().__init__()
        # Config — set once at on_algorithm_init, immutable thereafter.
        self.metrics_path = None
        self.n_pursuers = 0
        self.n_evaders = 0
        # Cross-episode buffer — appended on each on_episode_end across all
        # concurrent envs, drained on on_evaluate_start.
        self.episode_outcomes: list[EpisodeOutcome] = []

    @staticmethod
    def _init_episode_state(episode) -> dict:
        """Create a fresh per-episode state dict on episode.custom_data."""
        state = {
            "capture_steps": [],
            "captured_evader_ids": set(),
            "target_reached_ids": set(),
            "pursuer_entered_target_count": 0,
            "evader_shield_interventions": 0,
            "pursuer_shield_interventions": 0,
            "drone_object_collision_ids": set(),
            "drone_out_of_bounds_ids": set(),
            "collision_ids": set(),
            "timestep": 0,
            "evader_ids": set(),
            "pursuer_ids": set(),
        }
        episode.custom_data["metrics_state"] = state
        return state

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
        state = episode.custom_data.get("metrics_state")
        if state is None:
            # No state ever tracked for this episode — skip.
            return

        evader_ids = state["evader_ids"]
        pursuer_ids = state["pursuer_ids"]
        n_evaders = len(evader_ids)
        n_pursuers = len(pursuer_ids)
        timestep = state["timestep"]

        outcome_obj = EpisodeOutcome(
            capture_steps=list(state["capture_steps"]),
            breached=len(state["target_reached_ids"]) > 0,
            episode_length=timestep,
            evader_drone_collision_rate=rate(
                state["collision_ids"], evader_ids, n_evaders
            ),
            pursuer_drone_collision_rate=rate(
                state["collision_ids"], pursuer_ids, n_pursuers
            ),
            evader_obstacle_collision_rate=rate(
                state["drone_object_collision_ids"], evader_ids, n_evaders
            ),
            pursuer_obstacle_collision_rate=rate(
                state["drone_object_collision_ids"], pursuer_ids, n_pursuers
            ),
            pursuer_entered_target_count=state["pursuer_entered_target_count"],
            evader_out_of_bounds_rate=rate(
                state["drone_out_of_bounds_ids"], evader_ids, n_evaders
            ),
            pursuer_out_of_bounds_rate=rate(
                state["drone_out_of_bounds_ids"], pursuer_ids, n_pursuers
            ),
            evader_shield_intervention_rate=state["evader_shield_interventions"]
            / n_evaders
            / timestep
            if timestep > 0 and n_evaders > 0
            else 0.0,
            pursuer_shield_intervention_rate=state["pursuer_shield_interventions"]
            / n_pursuers
            / timestep
            if timestep > 0 and n_pursuers > 0
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
        # Fresh per-episode tracking dict on the episode object itself.
        # Avoids contamination across concurrent vectorized episodes.
        state = self._init_episode_state(episode)
        infos = self._get_episode_info(env, env_index)
        if infos is not None:
            state["evader_ids"].update(
                [d["drone"].id for d in infos.values() if d["drone"].is_evader]
            )
            state["pursuer_ids"].update(
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
        # Read the per-episode state for THIS episode (not shared).
        state = episode.custom_data.get("metrics_state")
        if state is None:
            # Defensive: on_episode_step fired before on_episode_start landed,
            # or state was lost somehow. Init now to avoid crashing.
            state = self._init_episode_state(episode)
        state["timestep"] += 1
        agent_ids = list(episode_info.keys())

        shield_drone_collision_ids = set()
        evader_ids = state["evader_ids"]
        pursuer_ids = state["pursuer_ids"]

        # count shield interventions by iterating alt_state events
        for event in episode_info[agent_ids[0]].get("shield_events", []):
            if event.drone_object_collision_event is not None:
                state["evader_shield_interventions"] += len(
                    [
                        id
                        for id in event.drone_object_collision_event.drone_ids
                        if id in evader_ids
                    ]
                )
                state["pursuer_shield_interventions"] += len(
                    [
                        id
                        for id in event.drone_object_collision_event.drone_ids
                        if id in pursuer_ids
                    ]
                )
            if event.out_of_bounds_event is not None:
                state["evader_shield_interventions"] += len(
                    [
                        id
                        for id in event.out_of_bounds_event.drone_ids
                        if id in evader_ids
                    ]
                )
                state["pursuer_shield_interventions"] += len(
                    [
                        id
                        for id in event.out_of_bounds_event.drone_ids
                        if id in pursuer_ids
                    ]
                )
            if event.collision_event is not None:
                shield_drone_collision_ids.update(event.collision_event.drone_ids)
                state["evader_shield_interventions"] += len(
                    [id for id in event.collision_event.drone_ids if id in evader_ids]
                )
                state["pursuer_shield_interventions"] += len(
                    [id for id in event.collision_event.drone_ids if id in pursuer_ids]
                )

        actual_drone_collision_ids = set()
        evaders_caught: set[int] = set()

        for event in episode_info[agent_ids[0]].get("events", []):
            if event.drone_object_collision_event is not None:
                state["drone_object_collision_ids"].update(
                    event.drone_object_collision_event.drone_ids
                )
            elif event.pursuer_entered_target_event is not None:
                state["pursuer_entered_target_count"] += len(
                    event.pursuer_entered_target_event.drone_ids
                )
            elif event.target_reached_event is not None:
                state["target_reached_ids"].update(event.target_reached_event.drone_ids)
            elif event.out_of_bounds_event is not None:
                state["drone_out_of_bounds_ids"].update(
                    event.out_of_bounds_event.drone_ids
                )
            elif event.collision_event is not None:
                evaders_in_coll: set[int] = set([x for x in event.collision_event.drone_ids if x in evader_ids])
                pursuers_in_coll = set([x for x in event.collision_event.drone_ids if x in pursuer_ids])

                n_caught = 0

                # any collision event that includes at least one evader and one pursuer
                if evaders_in_coll and pursuers_in_coll:
                    n_caught += len(evaders_in_coll) - len(evaders_caught.intersection(evaders_in_coll))
                    evaders_caught.update(evaders_in_coll)
                # Count collisions only for events with one team.
                # Mixed-team events are capture events and are excluded from collision_ids.
                elif evaders_in_coll and not pursuers_in_coll:
                    state["collision_ids"].update(evaders_in_coll)
                elif pursuers_in_coll and not evaders_in_coll:
                    state["collision_ids"].update(pursuers_in_coll)

                for _ in range(n_caught):
                    state["capture_steps"].append(state["timestep"])

                actual_drone_collision_ids.update(event.collision_event.drone_ids)

        # Count shield interventions for collisions, but only for drones "saved" by the shield
        shield_drone_collision_ids = shield_drone_collision_ids.difference(
            actual_drone_collision_ids
        )

        for evader_id in evaders_caught:
            if evader_id not in state["captured_evader_ids"]:
                state["capture_steps"].append(state["timestep"])
                state["captured_evader_ids"].add(evader_id)

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
                    loaded = json.load(f)
                    metrics_list = loaded if isinstance(loaded, list) else [loaded]

            # Append new metrics
            metrics_list.append(mean_summary)

            # Write updated metrics
            with open(results_dir, "w") as f:
                json.dump(metrics_list, f, indent=4)
            print(f"Saved evaluation metrics to {results_dir}")

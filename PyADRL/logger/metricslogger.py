"""Episode-level metric definitions and logging helpers for GridWorld."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os

from ray.rllib.callbacks.callbacks import RLlibCallback


WINDOW_SIZE = 100

_RESULTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
)


@dataclass
class EpisodeOutcome:
    captured: bool = False
    breached: bool = False
    capture_step: int | None = None
    episode_length: int = 0


def metrics_path(prefix: str) -> str:
    """Create a timestamped JSON path under the shared results directory."""
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(_RESULTS_DIR, f"{prefix}_{ts}.json")


def safe_json_value(val):
    """Convert numpy/Ray values into JSON-serializable Python types."""
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, list):
        return [safe_json_value(v) for v in val]
    if isinstance(val, dict):
        return {k: safe_json_value(v) for k, v in val.items()}
    try:
        return float(val)
    except (TypeError, ValueError):
        return str(val)


def extract_episode_metrics(infos) -> dict | None:
    if not isinstance(infos, dict):
        return None

    for val in infos.values():
        if isinstance(val, dict) and "episode_metrics" in val:
            return val["episode_metrics"]

    if "episode_metrics" in infos:
        return infos["episode_metrics"]
    return None


class MetricsCallback(RLlibCallback):
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
        if metrics_logger is None:
            return
        metrics = extract_episode_metrics(episode.get_infos(-1))
        if metrics is None:
            return

        captured = bool(metrics.get("captured", False))
        breached = bool(metrics.get("breached", False))
        capture_step = safe_json_value(metrics.get("capture_step"))
        episode_length = safe_json_value(metrics.get("episode_length"))

        metrics_logger.log_value("capture_rate", captured, window=WINDOW_SIZE)
        metrics_logger.log_value("avg_capture_step", capture_step, window=WINDOW_SIZE)
        metrics_logger.log_value("breach_rate", breached, window=WINDOW_SIZE)

        metrics_logger.log_value(
            "episode_logs",
            {
                "captured": captured,
                "breached": breached,
                "capture_step": capture_step,
                "episode_length": episode_length,
            },
            reduce="item_series",
        )


def build_train_iteration_data(result: dict, iteration: int) -> dict:
    env_runners = result.get("env_runners", {})
    mean_rewards = env_runners.get("agent_episode_returns_mean", {})
    episodes_value = safe_json_value(env_runners.get("episode_logs", []))
    episodes = episodes_value if isinstance(episodes_value, list) else []
    return {
        "iteration": iteration,
        "num_episodes": len(episodes),
        "summary": {
            "capture_rate": safe_json_value(env_runners.get("capture_rate")),
            "avg_capture_step": safe_json_value(env_runners.get("avg_capture_step")),
            "breach_rate": safe_json_value(env_runners.get("breach_rate")),
        },
        "rewards": safe_json_value(mean_rewards),
        "episodes": episodes,
    }


def build_eval_data(results: dict) -> dict:
    """Build a JSON-safe evaluation metrics object from RLlib results."""
    env_runners = results.get("env_runners", {})
    episodes_value = safe_json_value(env_runners.get("episode_logs", []))
    episodes = episodes_value if isinstance(episodes_value, list) else []
    return {
        "num_episodes": len(episodes),
        "summary": {
            "capture_rate": safe_json_value(env_runners.get("capture_rate")),
            "avg_capture_step": safe_json_value(env_runners.get("avg_capture_step")),
            "breach_rate": safe_json_value(env_runners.get("breach_rate")),
        },
        "rewards": safe_json_value(env_runners.get("agent_episode_returns_mean", {})),
        "episodes": episodes,
    }


def build_episode_summary(episodes: list[dict]) -> dict:
    """Build final aggregate summary across all recorded episodes."""
    if not episodes:
        return {
            "total_episodes": 0,
            "capture_rate": None,
            "breach_rate": None,
            "avg_capture_step": None,
            "avg_episode_length": None,
        }

    total_episodes = len(episodes)
    captured_values = [1.0 if ep.get("captured") else 0.0 for ep in episodes]
    breached_values = [1.0 if ep.get("breached") else 0.0 for ep in episodes]
    capture_steps = [ep.get("capture_step") for ep in episodes if ep.get("captured")]
    episode_lengths = [ep.get("episode_length") for ep in episodes]

    def _mean(values: list) -> float | None:
        numeric_values = [
            float(v)
            for v in values
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        if not numeric_values:
            return None
        return sum(numeric_values) / len(numeric_values)

    return {
        "total_episodes": total_episodes,
        "capture_rate": _mean(captured_values),
        "breach_rate": _mean(breached_values),
        "avg_capture_step": _mean(capture_steps),
        "avg_episode_length": _mean(episode_lengths),
    }


def build_train(episodes: list[dict], final_rewards: dict | None = None) -> dict:
    """Return the full training payload with per-episode data and final summary."""
    payload = {
        "episodes": episodes,
        "summary": build_episode_summary(episodes),
    }
    if final_rewards is not None:
        payload["summary"]["final_rewards"] = safe_json_value(final_rewards)

    return payload


def build_eval(episodes: list[dict], fallback_summary: dict | None = None) -> dict:
    """Return the full evaluation payload with episode logs and final summary."""
    summary = build_episode_summary(episodes)
    if summary["total_episodes"] == 0 and fallback_summary is not None:
        summary = safe_json_value(fallback_summary)

    return {
        "episodes": episodes,
        "summary": summary,
    }


def write_metrics(file_path: str, payload: dict) -> None:
    with open(file_path, "w") as f:
        json.dump(payload, f, indent=2)


def print_eval_summary(eval_data: dict, file_path: str) -> None:
    print(f"\n--- Evaluation Results ({eval_data['num_episodes']} episodes) ---")
    for key, val in eval_data["summary"].items():
        print(f"  {key}: {val}")
    print(f"\nMetrics written to {file_path}")

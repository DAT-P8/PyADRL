"""Metrics for evaluating pursuit-evasion episodes.

Inspired by Ray RLlib's custom metrics pattern
(ray/rllib/examples/metrics/custom_metrics_in_env_runners.py).

Each metric class accumulates per-episode outcomes over a sliding window
and exposes a ``compute()`` method that returns the current aggregate value.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
#  Episode outcome — produced by the environment at the end of each episode
# ---------------------------------------------------------------------------

@dataclass
class EpisodeOutcome:
    """Lightweight record of how a single episode ended."""

    captured: bool = False
    """True if the evader was captured by a pursuer."""

    breached: bool = False
    """True if the evader reached the target."""

    capture_step: int | None = None
    """Timestep at which the evader was captured (None if not captured)."""

    episode_length: int = 0
    """Total number of timesteps in the episode."""


# ---------------------------------------------------------------------------
#  Capture Rate
# ---------------------------------------------------------------------------

class CaptureRate:
    """Ratio of episodes where the evader is captured before the
    maximum episode length.

    Higher values indicate better pursuer performance.

    Args:
        window: Number of most-recent episodes to consider.
    """

    def __init__(self, window: int = 100):
        self._outcomes: deque[bool] = deque(maxlen=window)

    def update(self, outcome: EpisodeOutcome) -> None:
        self._outcomes.append(outcome.captured)

    def compute(self) -> float:
        """Return capture rate in [0, 1]. Returns 0.0 when no data."""
        if not self._outcomes:
            return 0.0
        return sum(self._outcomes) / len(self._outcomes)

    def reset(self) -> None:
        self._outcomes.clear()

    def __repr__(self) -> str:
        return f"CaptureRate(value={self.compute():.3f}, episodes={len(self._outcomes)})"


# ---------------------------------------------------------------------------
#  Capture Step
# ---------------------------------------------------------------------------

class CaptureStep:
    """Average timestep of first capture.

    If the evader was not captured in an episode the maximum episode
    length is recorded instead, penalising slow / failed pursuits.

    Lower values indicate quicker capture.

    Args:
        window: Number of most-recent episodes to consider.
        max_episode_length: Fallback value when capture does not occur.
    """

    def __init__(self, window: int = 100, max_episode_length: int = 100):
        self._steps: deque[int] = deque(maxlen=window)
        self.max_episode_length = max_episode_length

    def update(self, outcome: EpisodeOutcome) -> None:
        if outcome.captured and outcome.capture_step is not None:
            self._steps.append(outcome.capture_step)
        else:
            self._steps.append(self.max_episode_length)

    def compute(self) -> float:
        """Return mean capture step. Returns max_episode_length when no data."""
        if not self._steps:
            return float(self.max_episode_length)
        return sum(self._steps) / len(self._steps)

    def reset(self) -> None:
        self._steps.clear()

    def __repr__(self) -> str:
        return f"CaptureStep(value={self.compute():.1f}, episodes={len(self._steps)})"


# ---------------------------------------------------------------------------
#  Breach Rate
# ---------------------------------------------------------------------------

class BreachRate:
    """Ratio of episodes where the evader reaches the target.

    Higher values indicate worse pursuer performance (the evader is
    succeeding).

    Args:
        window: Number of most-recent episodes to consider.
    """

    def __init__(self, window: int = 100):
        self._outcomes: deque[bool] = deque(maxlen=window)

    def update(self, outcome: EpisodeOutcome) -> None:
        self._outcomes.append(outcome.breached)

    def compute(self) -> float:
        """Return breach rate in [0, 1]. Returns 0.0 when no data."""
        if not self._outcomes:
            return 0.0
        return sum(self._outcomes) / len(self._outcomes)

    def reset(self) -> None:
        self._outcomes.clear()

    def __repr__(self) -> str:
        return f"BreachRate(value={self.compute():.3f}, episodes={len(self._outcomes)})"


# ---------------------------------------------------------------------------
#  Metrics Logger — convenience aggregator (one per environment)
# ---------------------------------------------------------------------------

class MetricsLogger:
    """Aggregates all episode-level metrics in one place.

    Usage::

        logger = MetricsLogger(window=100, max_episode_length=100)
        # … at the end of every episode …
        logger.log(outcome)
        print(logger.report())
    """

    def __init__(self, window: int = 100, max_episode_length: int = 100):
        self.capture_rate = CaptureRate(window=window)
        self.capture_step = CaptureStep(window=window, max_episode_length=max_episode_length)
        self.breach_rate = BreachRate(window=window)
        self._episode_count: int = 0

    def log(self, outcome: EpisodeOutcome) -> None:
        """Record the outcome of a finished episode."""
        self.capture_rate.update(outcome)
        self.capture_step.update(outcome)
        self.breach_rate.update(outcome)
        self._episode_count += 1

    def report(self) -> dict[str, float]:
        """Return a dict of current metric values."""
        return {
            "episodes": self._episode_count,
            "capture_rate": self.capture_rate.compute(),
            "avg_capture_step": self.capture_step.compute(),
            "breach_rate": self.breach_rate.compute(),
        }

    def reset(self) -> None:
        """Clear all accumulated data."""
        self.capture_rate.reset()
        self.capture_step.reset()
        self.breach_rate.reset()
        self._episode_count = 0

    def __repr__(self) -> str:
        r = self.report()
        return (
            f"MetricsLogger(episodes={r['episodes']}, "
            f"capture_rate={r['capture_rate']:.3f}, "
            f"avg_capture_step={r['avg_capture_step']:.1f}, "
            f"breach_rate={r['breach_rate']:.3f})"
        )
"""Evaluation metrics for the pursuer-evader drone project."""

from abc import ABC, abstractmethod


class EpisodeResult:
    """Outcome of a single episode."""

    def __init__(
        self,
        captured: bool = False,
        breached: bool = False,
        out_of_bounds: bool = False,
        timesteps: int = 0,
        max_timesteps: int = 100,
        total_pursuer_reward: float = 0.0,
        total_evader_reward: float = 0.0,
    ):
        self.captured = captured
        self.breached = breached
        self.out_of_bounds = out_of_bounds
        self.timesteps = timesteps
        self.max_timesteps = max_timesteps
        self.total_pursuer_reward = total_pursuer_reward
        self.total_evader_reward = total_evader_reward


class Metric(ABC):
    """Abstract base class for evaluation metrics.

    Attributes:
        smaller: True if lower values are better, False if higher is better.
    """

    smaller: bool = False

    def __init__(self) -> None:
        self.reset()

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def update(self, episode: EpisodeResult) -> None: ...

    @abstractmethod
    def compute(self) -> float: ...

    def __repr__(self) -> str:
        arrow = "↓" if self.smaller else "↑"
        return f"{self.__class__.__name__}={self.compute():.4f} {arrow}"


class CaptureRate(Metric):
    """Ratio of episodes where the evader is captured. Higher is better."""

    smaller = False

    def reset(self) -> None:
        self.total = 0
        self.captures = 0

    def update(self, episode: EpisodeResult) -> None:
        self.total += 1
        if episode.captured:
            self.captures += 1

    def compute(self) -> float:
        return self.captures / self.total if self.total else 0.0


class CaptureStep(Metric):
    """Average timestep of capture (max_timesteps if not captured). Lower is better."""

    smaller = True

    def reset(self) -> None:
        self.total = 0
        self.steps = 0

    def update(self, episode: EpisodeResult) -> None:
        self.total += 1
        self.steps += episode.timesteps if episode.captured else episode.max_timesteps

    def compute(self) -> float:
        return self.steps / self.total if self.total else 0.0


class BreachRate(Metric):
    """Ratio of episodes where the evader reaches the target. Lower is better."""

    smaller = True

    def reset(self) -> None:
        self.total = 0
        self.breaches = 0

    def update(self, episode: EpisodeResult) -> None:
        self.total += 1
        if episode.breached:
            self.breaches += 1

    def compute(self) -> float:
        return self.breaches / self.total if self.total else 0.0

"""Episode-level metric definitions for the area defense environment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpisodeOutcome:
    """Record of how a single episode ended."""

    captured: bool = False
    breached: bool = False
    capture_step: int | None = None
    episode_length: int = 0
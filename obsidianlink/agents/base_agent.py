"""Minimal Agent interface: Observation in, Action out.

Evaluator-only truth (hidden_state, reward, biome) must never be an
argument here. Pose is already on ``Observation``. Keep this module
small — no planner, no framework.
"""

from __future__ import annotations

from obsidianlink.env.actions import Action
from obsidianlink.env.environment import Observation


class BaseAgent:
    """Unified agent API for the benchmark loop."""

    def reset(self) -> None:
        """Prepare for a new episode. Default is a no-op."""

    def act(self, observation: Observation) -> Action:
        raise NotImplementedError


__all__ = ["BaseAgent"]

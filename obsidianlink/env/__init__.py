"""Environment adapters.

Phase 1 / Step 1 introduces :class:`MineRLEnvironment` as the first real
adapter; the in-process simulated adapter will be added in a later
sub-step. New adapters must implement the :class:`Environment` protocol
from :mod:`obsidianlink.env.environment`.
"""

from obsidianlink.env.environment import Environment, Observation
from obsidianlink.env.minerl import MineRLEnvironment

__all__ = ["Environment", "Observation", "MineRLEnvironment"]

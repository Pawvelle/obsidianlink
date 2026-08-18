"""Agent protocol: observation in, action out."""

from typing import Protocol

from obsidianlink.env.actions import Action
from obsidianlink.env.environment import Observation


class Agent(Protocol):
    def act(self, observation: Observation) -> Action:
        """Return the next action. Must not receive evaluator-only truth."""
        ...

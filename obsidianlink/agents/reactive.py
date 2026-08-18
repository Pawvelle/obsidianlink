"""Reactive agent skeleton. No memory, planner, or reflection."""

from obsidianlink.agents.model_client import ModelClient
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Observation


class ReactiveAgent:
    def __init__(self, model: ModelClient) -> None:
        self._model = model
        self.model_calls = 0

    def act(self, observation: Observation) -> Action:
        del observation
        self.model_calls += 1
        self._model.complete("act")
        return Action(type=ActionType.WAIT)

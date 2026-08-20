"""LLM agent: Observation → prompt → model text → legal Action.

No memory, RAG, planner, tools, or multi-agent loop.
"""

from __future__ import annotations

from obsidianlink.agents.base_agent import BaseAgent
from obsidianlink.agents.prompt import build_prompt, parse_action
from obsidianlink.env.actions import Action
from obsidianlink.env.environment import Observation
from obsidianlink.models.base_client import BaseLLMClient


class LLMAgent(BaseAgent):
    """External LLM controller behind the unified Agent API."""

    def __init__(
        self,
        client: BaseLLMClient,
        *,
        goal: str | None = None,
    ) -> None:
        self._client = client
        self._goal = goal
        self.last_prompt: str | None = None
        self.last_raw_response: str | None = None
        self.last_parsed_ok: bool | None = None
        self.invalid_actions = 0
        self.model_calls = 0

    def reset(self) -> None:
        self.last_prompt = None
        self.last_raw_response = None
        self.last_parsed_ok = None
        self.invalid_actions = 0
        self.model_calls = 0

    def act(self, observation: Observation) -> Action:
        prompt = build_prompt(observation, goal=self._goal)
        self.last_prompt = prompt
        raw = self._client.generate(prompt)
        self.model_calls += 1
        self.last_raw_response = raw
        action, ok = parse_action(raw)
        self.last_parsed_ok = ok
        if not ok:
            self.invalid_actions += 1
        return action


__all__ = ["LLMAgent"]

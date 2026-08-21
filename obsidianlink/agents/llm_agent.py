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
        use_vision: bool = False,
    ) -> None:
        self._client = client
        self._goal = goal
        self.use_vision = bool(use_vision)
        self.last_prompt: str | None = None
        self.last_raw_response: str | None = None
        self.last_parsed_ok: bool | None = None
        self.last_used_vision = False
        self.last_fallback_reason: str | None = None
        self.invalid_actions = 0
        self.model_calls = 0
        self.vision_calls = 0

    def reset(self) -> None:
        self.last_prompt = None
        self.last_raw_response = None
        self.last_parsed_ok = None
        self.last_used_vision = False
        self.last_fallback_reason = None
        self.invalid_actions = 0
        self.model_calls = 0
        self.vision_calls = 0

    def act(self, observation: Observation) -> Action:
        vision_attached = self.use_vision and observation.frame is not None
        prompt = build_prompt(
            observation,
            goal=self._goal,
            vision_attached=vision_attached,
        )
        self.last_prompt = prompt
        raw = self._generate(prompt, observation)
        self.model_calls += 1
        self.last_raw_response = raw
        action, ok = parse_action(raw)
        self.last_parsed_ok = ok
        if not ok:
            self.invalid_actions += 1
        return action

    def _generate(self, prompt: str, observation: Observation) -> str:
        vision_fn = getattr(self._client, "generate_with_vision", None)
        if not self.use_vision:
            self.last_used_vision = False
            self.last_fallback_reason = None
            return self._client.generate(prompt)
        if observation.frame is None:
            self.last_used_vision = False
            self.last_fallback_reason = "no_frame"
            return self._client.generate(prompt)
        if not callable(vision_fn):
            self.last_used_vision = False
            self.last_fallback_reason = "text_only_model"
            return self._client.generate(prompt)
        try:
            text = vision_fn(prompt, frame=observation.frame)
        except (TypeError, ValueError) as exc:
            self.last_used_vision = False
            self.last_fallback_reason = f"frame_encode_failed:{type(exc).__name__}"
            return self._client.generate(prompt)
        self.last_used_vision = True
        self.last_fallback_reason = None
        self.vision_calls += 1
        return text


__all__ = ["LLMAgent"]

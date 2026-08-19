"""Reactive agent: Observation → prompt + frame → model → Action."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from obsidianlink.agents.model_client import ModelClient, call_model
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Observation

PromptBuilder = Callable[[Observation], str]


class ReactiveAgent:
    """No planner, no memory, no reflection. One observation, one action."""

    def __init__(
        self,
        model: ModelClient,
        *,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self._model = model
        self._prompt_builder = prompt_builder or _default_prompt
        self.model_calls = 0
        self.vision_calls = 0
        self.text_calls = 0
        self.invalid_actions = 0
        self.last_raw_response: str | None = None
        self.last_used_vision: bool = False
        self.last_fallback_reason: str | None = None
        self.last_report: Any = None

    def act(self, observation: Observation) -> Action:
        self.model_calls += 1
        prompt = self._prompt_builder(observation)
        call = call_model(self._model, prompt, observation=observation)
        self.last_raw_response = call.text
        self.last_used_vision = call.used_vision
        self.last_fallback_reason = call.fallback_reason
        if call.used_vision:
            self.vision_calls += 1
        else:
            self.text_calls += 1
        action, parsed = parse_model_response(call.text)
        if not parsed:
            self.invalid_actions += 1
        return action


def _default_prompt(observation: Observation) -> str:
    frame = observation.frame
    if frame is None:
        frame_summary = "frame: <none>"
    else:
        shape = getattr(frame, "shape", None)
        dtype = getattr(frame, "dtype", None)
        frame_summary = f"frame: ndarray shape={shape} dtype={dtype}"
    inventory = observation.inventory or {}
    if not inventory:
        inv_summary = "inventory: <empty>"
    else:
        items = ", ".join(f"{name}={qty}" for name, qty in list(inventory.items())[:5])
        inv_summary = f"inventory: {{{items}}}"
    return (
        "You are an agent in a Minecraft environment. "
        f"{frame_summary}; {inv_summary}; "
        f"selected_item={observation.selected_item!r}. "
        "Respond with a single JSON object describing the next action, "
        'e.g. {"action": "move", "dx": 1, "dz": 0} or {"action": "wait"}.'
    )


def parse_model_response(response: str) -> tuple[Action, bool]:
    """Parse JSON into an Action.

    Returns ``(action, parsed)``. ``parsed`` is False on empty / invalid
    JSON / unknown action verb. Those cases become WAIT.
    """
    data = extract_json_object(response)
    if data is None:
        return Action(type=ActionType.WAIT), False
    type_raw = data.get("action", "wait")
    if not isinstance(type_raw, str):
        return Action(type=ActionType.WAIT), False
    try:
        action_type = ActionType(type_raw.strip().lower())
    except ValueError:
        return Action(type=ActionType.WAIT), False

    def _int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    return (
        Action(
            type=action_type,
            dx=_int(data.get("dx", 0)),
            dz=_int(data.get("dz", 0)),
            yaw=_float(data.get("yaw", 0.0)),
            pitch=_float(data.get("pitch", 0.0)),
            target=str(data.get("target", "") or ""),
        ),
        True,
    )


def extract_json_object(response: str) -> dict[str, Any] | None:
    if not isinstance(response, str) or not response.strip():
        return None
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


__all__ = ["ReactiveAgent", "extract_json_object", "parse_model_response"]

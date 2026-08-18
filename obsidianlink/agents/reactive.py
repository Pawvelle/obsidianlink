"""Reactive agent: observation -> model -> action.

Phase 1 realisation of the dev plan's ReactiveAgent step:

1. Build a text prompt from the agent-visible :class:`Observation`
   (frame shape / dtype, inventory snapshot, selected hotbar item).
2. Call the injected :class:`ModelClient`.
3. Parse the response as a small JSON object describing the next
   :class:`obsidianlink.env.actions.Action`.
4. Fall back to a no-op WAIT when the response is malformed, so a
   misbehaving model cannot break the env loop.

The frame is summarised by shape and dtype only; raw pixels are NOT
embedded in the prompt. A future vision-capable model would receive
the frame through a separate channel (or the prompt would carry a
reference, not the data). This keeps the contract identical whether
the model is a heuristic or a real LLM, and keeps the prompt bounded
in size.

The agent is reactive on purpose: it has no memory of past
observations, no planner, no reflection. Those land in Phase 3.
"""

from __future__ import annotations

import json
from typing import Any

from obsidianlink.agents.model_client import ModelClient
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Observation


class ReactiveAgent:
    def __init__(self, model: ModelClient) -> None:
        self._model = model
        self.model_calls = 0

    def act(self, observation: Observation) -> Action:
        self.model_calls += 1
        prompt = self._build_prompt(observation)
        response = self._model.complete(prompt)
        return parse_model_response(response)

    @staticmethod
    def _build_prompt(observation: Observation) -> str:
        frame = observation.frame
        if frame is None:
            frame_summary = "frame: <none>"
        else:
            try:
                shape = getattr(frame, "shape", None)
                dtype = getattr(frame, "dtype", None)
            except Exception:  # defensive
                shape, dtype = None, None
            frame_summary = f"frame: ndarray shape={shape} dtype={dtype}"

        inventory: Any = observation.inventory
        if not inventory:
            inv_summary = "inventory: <empty>"
        else:
            items = ", ".join(
                f"{name}={qty}"
                for name, qty in list(inventory.items())[:5]
            )
            more = "" if len(inventory) <= 5 else ", ..."
            inv_summary = f"inventory: {{{items}{more}}}"

        return (
            "You are an agent in a Minecraft environment. "
            f"{frame_summary}; {inv_summary}; "
            f"selected_item={observation.selected_item!r}. "
            "Respond with a single JSON object describing the next action, "
            'e.g. {"action": "MOVE", "dx": 1, "dz": 0}, '
            '{"action": "ATTACK"}, {"action": "CAMERA", "yaw": 10.0}, '
            'or {"action": "WAIT"}.'
        )


def parse_model_response(response: str) -> Action:
    """Parse a model response string into an :class:`Action`.

    Returns a WAIT action on any parsing failure so the env loop can
    continue even when the model misbehaves.
    """
    if not isinstance(response, str) or not response.strip():
        return Action(type=ActionType.WAIT)
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return Action(type=ActionType.WAIT)
    if not isinstance(data, dict):
        return Action(type=ActionType.WAIT)

    type_raw = data.get("action", "wait")
    if not isinstance(type_raw, str):
        return Action(type=ActionType.WAIT)
    try:
        action_type = ActionType(type_raw.strip().lower())
    except ValueError:
        action_type = ActionType.WAIT

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

    return Action(
        type=action_type,
        dx=_int(data.get("dx", 0)),
        dz=_int(data.get("dz", 0)),
        yaw=_float(data.get("yaw", 0.0)),
        pitch=_float(data.get("pitch", 0.0)),
        target=str(data.get("target", "") or ""),
        slot=_int(data.get("slot", 0)),
    )


__all__ = ["ReactiveAgent", "parse_model_response"]

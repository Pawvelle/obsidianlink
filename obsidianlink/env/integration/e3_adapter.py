"""Adapter from backend public Observation onto the P1 E3 protocol.

Observed state comes only from ``Observation.selected_item`` returned by
backend reset. This module never parses raw ``equipped_items``, inventory,
hotbar mappings, expected config, action history, or evaluator truth.
Importing it never imports MineRL or creates a production backend.
"""

from __future__ import annotations

from typing import Mapping

from obsidianlink.env.integration.e0_adapter import MineRLE0LifecycleAdapter
from obsidianlink.env.integration.e3_config import build_e3_compatibility_task
from obsidianlink.env.validation.selected_item import PublicSelectedItemObservation


def _identity_field(value: object, field_name: str) -> object:
    if isinstance(value, Mapping) and field_name in value:
        return value[field_name]
    return getattr(value, field_name, None)


def _extract_selected_item(value: object) -> tuple[bool, object]:
    if isinstance(value, PublicSelectedItemObservation):
        return True, value.selected_item
    if isinstance(value, Mapping):
        if "selected_item" in value:
            return True, value["selected_item"]
        return False, None
    if hasattr(value, "selected_item"):
        return True, getattr(value, "selected_item")
    return False, None


def public_selected_item_observation(
    reset_result: object, *, episode_id: str
) -> dict[str, dict[str, object]]:
    """Project backend reset output to exact E3 public fields only."""

    if not isinstance(reset_result, Mapping) or not reset_result:
        return {}
    projected: dict[str, dict[str, object]] = {}
    for agent_id, value in reset_result.items():
        if not isinstance(agent_id, str) or not agent_id.strip() or value is None:
            continue
        observed_episode = _identity_field(value, "episode_id")
        observed_step = _identity_field(value, "step_id")
        observed_agent = _identity_field(value, "agent_id")
        payload: dict[str, object] = {
            "agent_id": (
                observed_agent
                if isinstance(observed_agent, str) and observed_agent.strip()
                else agent_id
            ),
            "episode_id": (
                observed_episode
                if isinstance(observed_episode, str) and observed_episode.strip()
                else episode_id
            ),
            "step_id": observed_step if type(observed_step) is int else 0,
        }
        found, selected_item = _extract_selected_item(value)
        if found:
            payload["selected_item"] = selected_item
        projected[agent_id] = payload
    return projected


class MineRLE3SelectedItemAdapter(MineRLE0LifecycleAdapter):
    """Translate backend reset output into E3 selected-item payloads."""

    @staticmethod
    def _build_compatibility_task(episode_id: str) -> object:
        return build_e3_compatibility_task(episode_id)

    def reset(self) -> Mapping[str, dict[str, object]]:
        if not self._opened:
            self.open()
        backend = self._ensure_backend()
        reset = getattr(backend, "reset", None)
        if not callable(reset):
            raise RuntimeError("MineRL backend reset is not callable")
        raw = reset(self._compatibility_task)
        return public_selected_item_observation(raw, episode_id=self.episode_id)

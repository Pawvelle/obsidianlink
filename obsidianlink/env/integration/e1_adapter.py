"""Adapter from MineRLEnvironmentBackend onto the P1 E1 RGB protocol.

The validation runner sees only ``reset()`` / ``close()``. This adapter
projects raw MineRL/backend reset output onto a public RGB observation:
identity fields plus ``rgb``. It discards inventory, selected item,
workflow, and evaluator-only fields so E1 cannot become E2--E12.

This adapter does not adopt the legacy ``Observation`` type as the E1
contract. Importing this module never starts MineRL.
"""

from __future__ import annotations

from typing import Mapping

from obsidianlink.env.integration.e0_adapter import MineRLE0LifecycleAdapter
from obsidianlink.env.validation.rgb import PublicRGBObservation


def _identity_field(value: object, field_name: str) -> object:
    if isinstance(value, Mapping) and field_name in value:
        return value[field_name]
    return getattr(value, field_name, None)


def _extract_rgb(value: object) -> tuple[bool, object]:
    if isinstance(value, PublicRGBObservation):
        return True, value.rgb
    if isinstance(value, Mapping):
        for key in ("rgb", "pov", "frame"):
            if key in value:
                return True, value[key]
        return False, None
    if hasattr(value, "rgb"):
        return True, getattr(value, "rgb")
    if hasattr(value, "frame"):
        return True, getattr(value, "frame")
    if hasattr(value, "pov"):
        return True, getattr(value, "pov")
    return False, None


def public_rgb_observation(
    reset_result: object,
    *,
    episode_id: str,
) -> dict[str, dict[str, object]]:
    """Project reset output to public RGB fields only.

    Inventory, selected item, workflow, evaluator grids, and other
    payloads are discarded. Missing RGB is represented by omitting the
    ``rgb`` key; an explicit ``None`` is preserved so E1 can fail closed
    on a None POV.
    """

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
        found, rgb = _extract_rgb(value)
        if found:
            payload["rgb"] = rgb
        projected[agent_id] = payload
    return projected


class MineRLE1RGBAdapter(MineRLE0LifecycleAdapter):
    """Translate MineRL open/reset(task)/close into E1 public RGB reset/close."""

    def reset(self) -> Mapping[str, dict[str, object]]:
        if not self._opened:
            self.open()
        backend = self._ensure_backend()
        reset = getattr(backend, "reset", None)
        if not callable(reset):
            raise RuntimeError("MineRL backend reset is not callable")
        raw = reset(self._compatibility_task)
        return public_rgb_observation(raw, episode_id=self.episode_id)

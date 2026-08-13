"""Adapter from MineRLEnvironmentBackend onto the P1 E2 inventory protocol.

Production observations are projected from the backend's public
``Observation.visible_inventory`` field. Expected calibration inventory and
observed reset inventory remain independent data paths. Importing this module
does not import MineRL or create a production backend.
"""

from __future__ import annotations

from typing import Mapping

from obsidianlink.env.integration.e0_adapter import MineRLE0LifecycleAdapter
from obsidianlink.env.integration.e2_config import build_e2_compatibility_task
from obsidianlink.env.validation.inventory import PublicInventoryObservation


def _identity_field(value: object, field_name: str) -> object:
    if isinstance(value, Mapping) and field_name in value:
        return value[field_name]
    return getattr(value, field_name, None)


def _extract_inventory(value: object) -> tuple[bool, object]:
    if isinstance(value, PublicInventoryObservation):
        return True, value.inventory
    if isinstance(value, Mapping):
        if "visible_inventory" in value:
            return True, value["visible_inventory"]
        if "inventory" in value:
            return True, value["inventory"]
        return False, None
    if hasattr(value, "visible_inventory"):
        return True, getattr(value, "visible_inventory")
    return False, None


def _detach_if_mapping(value: object) -> object:
    if isinstance(value, Mapping):
        return dict(value)
    return value


def public_inventory_observation(
    reset_result: object,
    *,
    episode_id: str,
) -> dict[str, dict[str, object]]:
    """Project backend reset output to E2 identity plus inventory only.

    No item names or quantities are coerced. Invalid values are preserved so
    the existing E2 public contract can fail closed. RGB, selected item,
    workflow state, evaluator truth, and arbitrary extra fields are dropped.
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
        found, inventory = _extract_inventory(value)
        if found:
            payload["inventory"] = _detach_if_mapping(inventory)
        projected[agent_id] = payload
    return projected


class MineRLE2InventoryAdapter(MineRLE0LifecycleAdapter):
    """Translate backend reset output into E2 public inventory payloads."""

    @staticmethod
    def _build_compatibility_task(episode_id: str) -> object:
        return build_e2_compatibility_task(episode_id)

    def reset(self) -> Mapping[str, dict[str, object]]:
        if not self._opened:
            self.open()
        backend = self._ensure_backend()
        reset = getattr(backend, "reset", None)
        if not callable(reset):
            raise RuntimeError("MineRL backend reset is not callable")
        raw = reset(self._compatibility_task)
        return public_inventory_observation(raw, episode_id=self.episode_id)

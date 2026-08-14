"""Narrow MineRL bridge for the P1 E5 movement calibration.

Importing this module does not import MineRL or construct a production
backend. Position and yaw truth come only from retained MineRL FullStats;
neither the requested move nor configured spawn is used as observed truth.
"""

from __future__ import annotations

from typing import Any, Mapping

from obsidianlink.core.types import BackendStep, MacroAction
from obsidianlink.env.integration.e0_adapter import MineRLE0LifecycleAdapter, public_initial_state
from obsidianlink.env.integration.e5_config import build_e5_compatibility_task
from obsidianlink.env.validation.movement import (
    MovementActionExecution,
    MovementOrientationSnapshot,
    PlayerPositionSnapshot,
)


def _scalar(value: object) -> object:
    shape = getattr(value, "shape", None)
    item = getattr(value, "item", None)
    if shape == () and callable(item):
        return item()
    return value


def player_position_snapshot(value: object) -> PlayerPositionSnapshot:
    """Project an exact backend mapping to typed evaluator-only E5 truth."""

    if not isinstance(value, Mapping):
        raise ValueError("player position truth is missing")
    required = {"episode_id", "agent_id", "step_id", "x", "y", "z"}
    if set(value) != required:
        raise ValueError("player position truth fields are missing or unknown")
    return PlayerPositionSnapshot(
        value["episode_id"], value["agent_id"], value["step_id"],
        _scalar(value["x"]), _scalar(value["y"]), _scalar(value["z"]),
    )


def movement_orientation_snapshot(value: object) -> MovementOrientationSnapshot:
    """Project only reset yaw from the established E4 FullStats surface."""

    if not isinstance(value, Mapping):
        raise ValueError("movement orientation truth is missing")
    required = {"episode_id", "agent_id", "step_id", "yaw", "pitch"}
    if set(value) != required:
        raise ValueError("movement orientation truth fields are missing or unknown")
    return MovementOrientationSnapshot(
        value["episode_id"], value["agent_id"], value["step_id"], _scalar(value["yaw"])
    )


class MineRLE5MovementAdapter(MineRLE0LifecycleAdapter):
    """Own one backend and expose exactly the action/truth E5 requires."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tested_action_count = 0

    @staticmethod
    def _build_compatibility_task(episode_id: str) -> object:
        return build_e5_compatibility_task(episode_id)

    def reset(self) -> Mapping[str, dict[str, object]]:
        if not self._opened:
            self.open()
        backend = self._ensure_backend()
        reset = getattr(backend, "reset", None)
        if not callable(reset):
            raise RuntimeError("MineRL backend reset is not callable")
        self._tested_action_count = 0
        raw = reset(self._compatibility_task)
        return public_initial_state(raw, episode_id=self.episode_id)

    def player_position_truth(self) -> PlayerPositionSnapshot | None:
        backend = self._ensure_backend()
        getter = getattr(backend, "get_player_position_truth", None)
        if not callable(getter):
            raise RuntimeError("MineRL backend position truth is not callable")
        value = getter()
        return None if value is None else player_position_snapshot(value)

    def movement_orientation_truth(self) -> MovementOrientationSnapshot | None:
        backend = self._ensure_backend()
        getter = getattr(backend, "get_camera_orientation_truth", None)
        if not callable(getter):
            raise RuntimeError("MineRL backend orientation truth is not callable")
        value = getter()
        return None if value is None else movement_orientation_snapshot(value)

    def reset_failure_audit(self) -> dict[str, int]:
        """Project backend reset counters without exposing MineRL state."""

        backend = self._ensure_backend()
        getter = getattr(backend, "get_reset_audit", None)
        if not callable(getter):
            raise RuntimeError("MineRL backend reset audit is not callable")
        value = getter()
        required = {"reset_attempt_count", "environment_launch_count"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("backend reset audit fields are missing or unknown")
        result: dict[str, int] = {}
        for field_name in sorted(required):
            field_value = value[field_name]
            if type(field_value) is not int or field_value < 0:
                raise ValueError(f"{field_name} must be a non-negative int")
            result[field_name] = field_value
        return result

    def execute_movement_action(self, action: MacroAction) -> MovementActionExecution:
        if not isinstance(action, MacroAction) or action.action_type != "move":
            raise ValueError("E5 tested action must be MacroAction('move')")
        expected = {"forward": 1.0, "strafe": 0.0, "sprint": False, "jump": False}
        if action.duration_ticks != 1 or dict(action.parameters) != expected:
            raise ValueError("E5 tested move differs from frozen calibration")
        self._tested_action_count += 1
        if self._tested_action_count != 1:
            raise RuntimeError("E5 permits exactly one tested movement action")
        backend = self._ensure_backend()
        step = getattr(backend, "step", None)
        if not callable(step):
            raise RuntimeError("MineRL backend step is not callable")
        result = step({"agent_1": action})
        if not isinstance(result, BackendStep):
            raise TypeError("MineRL backend step must return BackendStep")
        return MovementActionExecution(
            episode_id=result.episode_id,
            agent_id="agent_1",
            step_id=result.step_id,
            action_type=action.action_type,
            forward=action.parameters.get("forward"),
            strafe=action.parameters.get("strafe"),
            sprint=action.parameters.get("sprint"),
            jump=action.parameters.get("jump"),
            duration_ticks=action.duration_ticks,
            translated_action_accepted=result.info.get("translation_accepted"),
            tested_action_count=self._tested_action_count,
        )

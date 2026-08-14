"""Narrow MineRL bridge for the P1 E4 camera calibration.

Importing this module does not import MineRL or construct a production
backend. Orientation truth comes only from the backend's retained MineRL
FullStats info; action intent is never used as observed truth.
"""

from __future__ import annotations

from typing import Any, Mapping

from obsidianlink.core.types import BackendStep, MacroAction
from obsidianlink.env.integration.e0_adapter import MineRLE0LifecycleAdapter, public_initial_state
from obsidianlink.env.integration.e4_config import build_e4_compatibility_task
from obsidianlink.env.validation.camera import CameraActionExecution, CameraOrientationSnapshot


def _scalar(value: object) -> object:
    shape = getattr(value, "shape", None)
    item = getattr(value, "item", None)
    if shape == () and callable(item):
        return item()
    return value


def camera_orientation_snapshot(value: object) -> CameraOrientationSnapshot:
    """Project an evaluator-only backend mapping to the typed E4 truth."""

    if not isinstance(value, Mapping):
        raise ValueError("camera orientation truth is missing")
    required = {"episode_id", "agent_id", "step_id", "yaw", "pitch"}
    if set(value) != required:
        raise ValueError("camera orientation truth fields are missing or unknown")
    return CameraOrientationSnapshot(
        episode_id=value["episode_id"],
        agent_id=value["agent_id"],
        step_id=value["step_id"],
        yaw=_scalar(value["yaw"]),
        pitch=_scalar(value["pitch"]),
    )


class MineRLE4CameraAdapter(MineRLE0LifecycleAdapter):
    """Own one backend and expose the exact lifecycle/action/truth E4 needs."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tested_action_count = 0

    @staticmethod
    def _build_compatibility_task(episode_id: str) -> object:
        return build_e4_compatibility_task(episode_id)

    def reset(self) -> Mapping[str, dict[str, object]]:
        if not self._opened:
            self.open()
        backend = self._ensure_backend()
        reset = getattr(backend, "reset", None)
        if not callable(reset):
            raise RuntimeError("MineRL backend reset is not callable")
        raw = reset(self._compatibility_task)
        self._tested_action_count = 0
        return public_initial_state(raw, episode_id=self.episode_id)

    def camera_orientation_truth(self) -> CameraOrientationSnapshot | None:
        backend = self._ensure_backend()
        getter = getattr(backend, "get_camera_orientation_truth", None)
        if not callable(getter):
            raise RuntimeError("MineRL backend camera truth is not callable")
        value = getter()
        if value is None:
            return None
        return camera_orientation_snapshot(value)

    def execute_camera_action(self, action: MacroAction) -> CameraActionExecution:
        if not isinstance(action, MacroAction) or action.action_type != "look":
            raise ValueError("E4 tested action must be MacroAction('look')")
        if action.duration_ticks != 1:
            raise ValueError("E4 tested look must have duration_ticks=1")
        self._tested_action_count += 1
        if self._tested_action_count != 1:
            raise RuntimeError("E4 permits exactly one tested camera action")
        backend = self._ensure_backend()
        step = getattr(backend, "step", None)
        if not callable(step):
            raise RuntimeError("MineRL backend step is not callable")
        result = step({"agent_1": action})
        if not isinstance(result, BackendStep):
            raise TypeError("MineRL backend step must return BackendStep")
        accepted = result.info.get("translation_accepted")
        return CameraActionExecution(
            episode_id=result.episode_id,
            agent_id="agent_1",
            step_id=result.step_id,
            action_type=action.action_type,
            requested_yaw=action.parameters.get("yaw"),
            requested_pitch=action.parameters.get("pitch"),
            translated_action_accepted=accepted,
            tested_action_count=self._tested_action_count,
        )

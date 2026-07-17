"""Bounded recent-action detector for repeated rotation without forward progress."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from mc_agent.actions import MacroAction


@dataclass(frozen=True)
class TurningLoopState:
    active: bool
    rotation_actions: int
    cumulative_abs_yaw: float

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


class TurningLoopDetector:
    def __init__(self, window_size: int = 3, yaw_threshold: float = 30.0):
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        if yaw_threshold <= 0:
            raise ValueError("yaw_threshold must be positive")
        self.window_size = window_size
        self.yaw_threshold = yaw_threshold
        self._history: deque[float | None] = deque(maxlen=window_size)

    def reset(self) -> None:
        self._history.clear()

    @staticmethod
    def is_rotation_only(action: MacroAction) -> bool:
        return action.action in {"look", "turn"} and action.camera_yaw != 0.0

    def observe(self, action: MacroAction) -> TurningLoopState:
        if self.is_rotation_only(action):
            self._history.append(abs(float(action.camera_yaw)))
        else:
            # Forward motion, waiting, zero-angle looks, or vertical-only looks all
            # break a run of yaw-only rotation.
            self._history.append(None)
        return self.snapshot()

    def snapshot(self) -> TurningLoopState:
        rotations = [yaw for yaw in self._history if yaw is not None]
        consecutive_rotations = (
            len(self._history) == self.window_size
            and len(rotations) == self.window_size
        )
        cumulative_yaw = float(sum(rotations))
        return TurningLoopState(
            active=consecutive_rotations and cumulative_yaw >= self.yaw_threshold,
            rotation_actions=len(rotations),
            cumulative_abs_yaw=cumulative_yaw,
        )

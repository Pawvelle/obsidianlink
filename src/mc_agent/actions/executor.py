"""Deterministic translation from macro-actions to MineRL ticks."""

from __future__ import annotations

from typing import Any

import numpy as np

from .schema import MacroAction, limit_macro_action
from .watchdog import Watchdog


class MacroExecutor:
    def __init__(self, action_space: Any, watchdog: Watchdog | None = None):
        self.action_space = action_space
        self.watchdog = watchdog
        self.current = MacroAction.no_op("initial")
        self.elapsed_ticks = self.current.duration_ticks

    @property
    def needs_action(self) -> bool:
        return self.elapsed_ticks >= self.current.duration_ticks

    def submit(self, action: MacroAction) -> None:
        self.current = limit_macro_action(action)
        self.elapsed_ticks = 0

    def interrupt(self, reason: str = "interrupt") -> None:
        self.current = MacroAction.no_op(reason)
        self.elapsed_ticks = self.current.duration_ticks

    def next_tick(self) -> dict[str, Any]:
        if self.watchdog is not None and self.watchdog.should_stop:
            self.interrupt(self.watchdog.reason or "watchdog")
            return self._no_op()
        if self.needs_action:
            return self._no_op()

        tick = self._no_op()
        action = self.current
        first_tick = self.elapsed_ticks == 0
        if action.action == "move_forward":
            tick["forward"] = 1
        if first_tick and action.action in {"look", "turn", "move_forward"}:
            tick["camera"] = np.asarray(
                [action.camera_pitch, action.camera_yaw], dtype=np.float32
            )
        tick["attack"] = int(action.attack)
        tick["jump"] = int(action.jump)
        tick["sprint"] = int(action.sprint and action.action == "move_forward")
        tick["ESC"] = 0
        self.elapsed_ticks += 1
        return tick

    def _no_op(self) -> dict[str, Any]:
        tick = self.action_space.no_op()
        tick["ESC"] = 0
        return tick

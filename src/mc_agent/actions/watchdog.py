"""Immediate, thread-safe stop signal for the environment loop."""

from __future__ import annotations

import threading


class Watchdog:
    def __init__(self, max_ticks: int | None = None):
        self.max_ticks = max_ticks
        self.ticks = 0
        self._stop = threading.Event()
        self.reason: str | None = None

    @property
    def should_stop(self) -> bool:
        return self._stop.is_set() or (
            self.max_ticks is not None and self.ticks >= self.max_ticks
        )

    def request_stop(self, reason: str = "requested") -> None:
        self.reason = reason
        self._stop.set()

    def after_tick(self) -> None:
        self.ticks += 1
        if self.max_ticks is not None and self.ticks >= self.max_ticks:
            self.reason = self.reason or "max_ticks"

"""Capacity-one latest-action mailbox for future planner isolation."""

from __future__ import annotations

import queue

from .schema import MacroAction


class LatestActionMailbox:
    def __init__(self):
        self._queue: queue.Queue[MacroAction] = queue.Queue(maxsize=1)

    def publish(self, action: MacroAction) -> None:
        try:
            self._queue.put_nowait(action)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        self._queue.put_nowait(action)

    def take_latest(self) -> MacroAction | None:
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

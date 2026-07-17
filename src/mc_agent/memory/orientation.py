"""Bounded relative-orientation memory for short-horizon visual exploration."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

from mc_agent.actions import MacroAction


@dataclass(frozen=True)
class OrientationView:
    heading: int
    low_change: bool


@dataclass(frozen=True)
class OrientationState:
    active: bool
    relative_yaw: float
    heading: int
    suggested_yaw: int
    recent_views: tuple[OrientationView, ...]
    unique_headings: int
    revisit_samples: int
    total_samples: int

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


class OrientationMemory:
    def __init__(self, max_recent_views: int = 3, bucket_degrees: int = 20):
        if max_recent_views < 1:
            raise ValueError("max_recent_views must be positive")
        if bucket_degrees < 1 or 360 % bucket_degrees != 0:
            raise ValueError("bucket_degrees must divide 360")
        self.max_recent_views = max_recent_views
        self.bucket_degrees = bucket_degrees
        self._recent: deque[OrientationView] = deque(maxlen=max_recent_views)
        self._visited: set[int] = set()
        self._visit_counts: dict[int, int] = {}
        self._relative_yaw = 0.0
        self._revisit_samples = 0
        self._total_samples = 0

    def reset(self) -> None:
        self._recent.clear()
        self._visited.clear()
        self._visit_counts.clear()
        self._relative_yaw = 0.0
        self._revisit_samples = 0
        self._total_samples = 0

    @staticmethod
    def _wrap_yaw(yaw: float) -> float:
        return (yaw + 180.0) % 360.0 - 180.0

    def _heading_bucket(self) -> int:
        half = self.bucket_degrees / 2.0
        bucket = (
            math.floor((self._relative_yaw + half) / self.bucket_degrees)
            * self.bucket_degrees
        )
        return int(self._wrap_yaw(float(bucket)))

    def observe_action(self, action: MacroAction) -> OrientationState:
        self._relative_yaw = self._wrap_yaw(
            self._relative_yaw + float(action.camera_yaw)
        )
        return self.snapshot()

    def observe_view(self, low_change: bool) -> OrientationState:
        if type(low_change) is not bool:
            raise ValueError("low_change must be boolean")
        heading = self._heading_bucket()
        self._revisit_samples += int(heading in self._visited)
        self._visited.add(heading)
        self._visit_counts[heading] = self._visit_counts.get(heading, 0) + 1
        self._total_samples += 1
        self._recent.append(OrientationView(heading=heading, low_change=low_change))
        return self.snapshot()

    def snapshot(self) -> OrientationState:
        heading = self._heading_bucket()
        right = int(self._wrap_yaw(heading + self.bucket_degrees))
        left = int(self._wrap_yaw(heading - self.bucket_degrees))
        suggested_yaw = (
            self.bucket_degrees
            if self._visit_counts.get(right, 0) <= self._visit_counts.get(left, 0)
            else -self.bucket_degrees
        )
        return OrientationState(
            active=bool(self._recent),
            relative_yaw=self._relative_yaw,
            heading=heading,
            suggested_yaw=suggested_yaw,
            recent_views=tuple(self._recent),
            unique_headings=len(self._visited),
            revisit_samples=self._revisit_samples,
            total_samples=self._total_samples,
        )

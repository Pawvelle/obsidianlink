"""Bounded relative-orientation memory for short-horizon visual exploration."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

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


@dataclass(frozen=True)
class FrameChange:
    mean_absolute_difference: float
    changed_pixel_fraction: float
    low_change: bool

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


class FrameChangeDetector:
    """Compare sparse grayscale POV features while ignoring the HUD strip."""

    def __init__(
        self,
        mean_difference_threshold: float = 0.005,
        changed_fraction_threshold: float = 0.01,
        pixel_difference_threshold: float = 20.0,
    ):
        if not 0.0 <= mean_difference_threshold <= 1.0:
            raise ValueError("mean_difference_threshold must be within [0, 1]")
        if not 0.0 <= changed_fraction_threshold <= 1.0:
            raise ValueError("changed_fraction_threshold must be within [0, 1]")
        if not 0.0 <= pixel_difference_threshold <= 255.0:
            raise ValueError("pixel_difference_threshold must be within [0, 255]")
        self.mean_difference_threshold = mean_difference_threshold
        self.changed_fraction_threshold = changed_fraction_threshold
        self.pixel_difference_threshold = pixel_difference_threshold
        self._reference: np.ndarray | None = None

    def reset(self, pov: np.ndarray) -> None:
        self._reference = self._feature(pov)

    def compare_and_update(self, pov: np.ndarray) -> FrameChange:
        if self._reference is None:
            raise RuntimeError("frame change detector must be reset first")
        current = self._feature(pov)
        difference = np.abs(current - self._reference)
        mean_difference = float(difference.mean() / 255.0)
        changed_fraction = float(
            np.count_nonzero(difference >= self.pixel_difference_threshold)
            / difference.size
        )
        result = FrameChange(
            mean_absolute_difference=mean_difference,
            changed_pixel_fraction=changed_fraction,
            low_change=(
                mean_difference < self.mean_difference_threshold
                and changed_fraction < self.changed_fraction_threshold
            ),
        )
        self._reference = current
        return result

    @staticmethod
    def _feature(pov: np.ndarray) -> np.ndarray:
        if not isinstance(pov, np.ndarray) or pov.dtype != np.uint8:
            raise ValueError("pov must be a uint8 numpy array")
        if pov.shape != (360, 640, 3):
            raise ValueError(f"unexpected pov shape: {pov.shape}")
        sampled = pov[:300:8, ::8].astype(np.float32)
        return (
            sampled[..., 0] * 0.299
            + sampled[..., 1] * 0.587
            + sampled[..., 2] * 0.114
        )


__all__ = [
    "FrameChange",
    "FrameChangeDetector",
    "OrientationMemory",
    "OrientationState",
    "OrientationView",
]

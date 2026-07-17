"""Deterministic low-cost visual change measurement for Phase 5."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrameChange:
    mean_absolute_difference: float
    changed_pixel_fraction: float
    low_change: bool

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


class FrameChangeDetector:
    """Compare sparse grayscale POV features while ignoring the bottom HUD strip."""

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
        # Keep rows 0..299 and sample every eighth pixel. The bottom 60 rows
        # contain the HUD, whose heart/hunger animation is not world movement.
        sampled = pov[:300:8, ::8].astype(np.float32)
        return (
            sampled[..., 0] * 0.299
            + sampled[..., 1] * 0.587
            + sampled[..., 2] * 0.114
        )

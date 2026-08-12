"""Small, stable v2 metric vocabulary."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class MetricName(str, Enum):
    SUCCESS_RATE = "success_rate"
    COMPLETION_RATE = "completion_rate"
    ENVIRONMENT_STEPS = "environment_steps"
    GAME_TIME = "game_time"
    MODEL_CALLS = "model_calls"
    INVALID_ACTION_RATE = "invalid_action_rate"
    RECOVERY_RATE = "recovery_rate"
    EVIDENCE_COMPLETENESS = "evidence_completeness"


@dataclass(frozen=True)
class MetricRecord:
    name: MetricName
    value: float
    numerator: int | None = None
    denominator: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, MetricName):
            raise ValueError("name must be MetricName")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ValueError("value must be numeric")
        if not math.isfinite(float(self.value)):
            raise ValueError("value must be finite")
        if (self.numerator is None) != (self.denominator is None):
            raise ValueError("numerator and denominator must be provided together")
        if self.numerator is not None:
            if type(self.numerator) is not int or self.numerator < 0:
                raise ValueError("numerator must be a non-negative int")
            if type(self.denominator) is not int or self.denominator < 1:
                raise ValueError("denominator must be a positive int")
            if self.numerator > self.denominator:
                raise ValueError("numerator cannot exceed denominator")

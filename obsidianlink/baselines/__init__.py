"""Baseline categories; scripted policies remain calibration/oracle assets."""

from enum import Enum


class BaselineKind(str, Enum):
    SCRIPTED_ORACLE = "scripted_oracle"
    REACTIVE = "reactive"
    PLANNER = "planner"


__all__ = ["BaselineKind"]

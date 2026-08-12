"""Stable v2 task identity vocabulary.

This module intentionally defines taxonomy only. Concrete L1--L4 and D1--D6
task instances are frozen in later roadmap phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


TASK_FAMILY = "nether_portal_construction"
DIAGNOSTIC_LEVELS = frozenset({f"D{index}" for index in range(1, 7)})
PORTAL_LEVELS = frozenset({f"L{index}" for index in range(1, 5)})


class BenchmarkSuite(str, Enum):
    DIAGNOSTIC = "diagnostic"
    END_TO_END = "end_to_end"
    GENERALIZATION_RECOVERY = "generalization_recovery"


class ExecutionMode(str, Enum):
    SINGLE = "single"
    MULTI = "multi"


class LayoutType(str, Enum):
    CONTROLLED = "controlled"
    RANDOMIZED = "randomized"
    HIDDEN = "hidden"
    CHALLENGE = "challenge"


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class TaskIdentity:
    """Taxonomic identity for a future v2 task instance."""

    task_instance_id: str
    suite: BenchmarkSuite
    mode: ExecutionMode
    level: str
    layout: LayoutType
    family: str = TASK_FAMILY

    def __post_init__(self) -> None:
        _identifier(self.task_instance_id, "task_instance_id")
        if self.family != TASK_FAMILY:
            raise ValueError(f"family must be {TASK_FAMILY!r}")
        if not isinstance(self.suite, BenchmarkSuite):
            raise ValueError("suite must be BenchmarkSuite")
        if not isinstance(self.mode, ExecutionMode):
            raise ValueError("mode must be ExecutionMode")
        if not isinstance(self.layout, LayoutType):
            raise ValueError("layout must be LayoutType")
        allowed = (
            DIAGNOSTIC_LEVELS
            if self.suite is BenchmarkSuite.DIAGNOSTIC
            else PORTAL_LEVELS
        )
        if self.level not in allowed:
            raise ValueError(
                f"level {self.level!r} is invalid for suite {self.suite.value!r}"
            )

    @property
    def canonical_name(self) -> str:
        mode = "s" if self.mode is ExecutionMode.SINGLE else "m"
        return "_".join(
            (
                self.family,
                mode,
                self.suite.value,
                self.level.lower(),
                self.layout.value,
            )
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "task_instance_id": self.task_instance_id,
            "family": self.family,
            "suite": self.suite.value,
            "mode": self.mode.value,
            "level": self.level,
            "layout": self.layout.value,
        }

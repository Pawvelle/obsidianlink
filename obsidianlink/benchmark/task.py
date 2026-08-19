"""A Task describes a problem. It must not contain a solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Task:
    task_id: str
    goal: str
    max_steps: int
    initial_condition: str = ""
    allowed_actions: tuple[str, ...] = ()
    evaluation_condition: str = ""
    #: Evaluator-only. Must never enter Observation or the prompt.
    ground_truth: Any = None

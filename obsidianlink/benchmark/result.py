"""Episode result. Small metric set plus failure-attribution evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

AGENT_FAILURE = "agent_failure"
ENVIRONMENT_FAILURE = "environment_failure"
EVALUATOR_FAILURE = "evaluator_failure"


@dataclass(frozen=True)
class Result:
    task_id: str
    success: bool
    steps: int
    model_calls: int
    invalid_actions: int
    elapsed_time: float
    evidence: Mapping[str, Any] = field(default_factory=dict)

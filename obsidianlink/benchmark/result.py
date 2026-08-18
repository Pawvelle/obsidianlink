"""Episode result. Keep the metric set small until a research question needs more."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Result:
    task_id: str
    success: bool
    steps: int
    model_calls: int
    invalid_actions: int
    elapsed_time: float

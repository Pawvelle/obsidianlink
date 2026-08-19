"""Evaluator interface.

Must use environment-side truth, never model self-report as success.
Missing or untrusted truth is ``evaluation_error`` (evaluator_failure),
not an agent task-failure label.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from obsidianlink.benchmark.result import Result
from obsidianlink.benchmark.task import Task


class Evaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        task: Task,
        *,
        steps: int,
        model_calls: int,
        invalid_actions: int,
        elapsed_time: float,
        observation: Any = None,
        raw_response: Any = None,
        ground_truth: Any = None,
        hidden_state: Any = None,
        used_vision: bool | None = None,
        fallback_reason: str | None = None,
        vision_calls: int = 0,
    ) -> Result:
        raise NotImplementedError

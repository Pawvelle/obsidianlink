"""Evaluator interface.

Evaluator-only information must never enter agent observations, prompts, or memory.
This module does not implement a truth framework.
"""

from abc import ABC, abstractmethod

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
    ) -> Result:
        raise NotImplementedError

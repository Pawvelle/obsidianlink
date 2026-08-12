"""Runner contracts. No solver or deterministic plan is imported here."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from obsidianlink.benchmark.evaluator import EvaluatorVerdict
from obsidianlink.benchmark.task import TaskIdentity


RUNNER_STATUSES = frozenset({"completed", "blocked", "failed"})


@dataclass(frozen=True)
class RunnerResult:
    task: TaskIdentity
    status: str
    evaluator_verdict: EvaluatorVerdict | None

    def __post_init__(self) -> None:
        if self.status not in RUNNER_STATUSES:
            raise ValueError(f"unknown runner status: {self.status!r}")
        if self.evaluator_verdict is not None and not isinstance(
            self.evaluator_verdict, EvaluatorVerdict
        ):
            raise ValueError("evaluator_verdict must be EvaluatorVerdict or None")


@runtime_checkable
class BenchmarkRunner(Protocol):
    def run(self, task: TaskIdentity) -> RunnerResult:
        ...

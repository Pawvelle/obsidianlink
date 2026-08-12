"""Evaluator boundary for v2.

Concrete evaluators consume evaluator-only state. They must not consume an
Agent response, policy object, or scripted-driver completion flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

from obsidianlink.benchmark.evidence import EvidenceIdentity


StateT = TypeVar("StateT")


@dataclass(frozen=True)
class EvaluatorVerdict:
    identity: EvidenceIdentity
    success: bool
    outcome: str
    evidence_complete: bool

    def __post_init__(self) -> None:
        if not isinstance(self.success, bool):
            raise ValueError("success must be bool")
        if not isinstance(self.evidence_complete, bool):
            raise ValueError("evidence_complete must be bool")
        if not isinstance(self.outcome, str) or not self.outcome.strip():
            raise ValueError("outcome must be a non-empty string")
        if self.success and not self.evidence_complete:
            raise ValueError("success requires complete evidence")


@runtime_checkable
class Evaluator(Protocol, Generic[StateT]):
    def evaluate(self, state: StateT) -> EvaluatorVerdict:
        ...

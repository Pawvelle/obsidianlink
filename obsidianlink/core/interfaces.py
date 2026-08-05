from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

from obsidianlink.core.types import BackendStep, MacroAction, Observation, TaskInstance
from obsidianlink.env.capabilities import BackendCapabilities
from obsidianlink.evaluation.portal import EvaluationState


@runtime_checkable
class EnvironmentBackend(Protocol):
    @property
    def agent_ids(self) -> tuple[str, ...]:
        ...

    def capabilities(self) -> BackendCapabilities:
        ...

    def open(self) -> None:
        ...

    def reset(self, task: TaskInstance) -> Mapping[str, Observation]:
        ...

    def step(self, actions: Mapping[str, MacroAction]) -> BackendStep:
        ...

    def get_evaluation_state(self) -> EvaluationState:
        ...

    def close(self) -> None:
        ...

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class WorkflowStage:
    name: str
    depends_on: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("workflow stage name must be a non-empty string")
        if any(not isinstance(item, str) or not item.strip() for item in self.depends_on):
            raise ValueError("workflow dependencies must be non-empty strings")
        if self.name in self.depends_on:
            raise ValueError("workflow stage cannot depend on itself")


class WorkflowDefinition:
    def __init__(self, stages: Iterable[WorkflowStage]):
        stage_list = tuple(stages)
        if not stage_list:
            raise ValueError("workflow must contain at least one stage")
        names = tuple(stage.name for stage in stage_list)
        if len(set(names)) != len(names):
            raise ValueError("workflow stage names must be unique")
        known: set[str] = set()
        for stage in stage_list:
            unknown = set(stage.depends_on) - known
            if unknown:
                raise ValueError(
                    f"stage {stage.name} depends on unavailable stages: {sorted(unknown)}"
                )
            known.add(stage.name)
        self._stages = stage_list

    @property
    def stages(self) -> tuple[WorkflowStage, ...]:
        return self._stages

    def available(self, completed: Iterable[str]) -> tuple[WorkflowStage, ...]:
        completed_set = set(completed)
        known = {stage.name for stage in self._stages}
        unknown = completed_set - known
        if unknown:
            raise ValueError(f"unknown completed stages: {sorted(unknown)}")
        return tuple(
            stage
            for stage in self._stages
            if stage.name not in completed_set
            and set(stage.depends_on).issubset(completed_set)
        )

    def is_complete(self, completed: Iterable[str]) -> bool:
        return {stage.name for stage in self._stages} == set(completed)

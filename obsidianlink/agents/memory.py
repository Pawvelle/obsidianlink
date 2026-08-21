"""Episode-local working memory for planner decisions.

Memory is not an action log.  It stores task state, subgoal progress,
retrieved wiki knowledge, recent failures, and outcome reflections so the
planner can choose the next operation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from obsidianlink.env.environment import Observation

_TASK_IDLE = "idle"
_TASK_IN_PROGRESS = "in_progress"
_TASK_COMPLETED = "completed"
_TASK_FAILED = "failed"


@dataclass(frozen=True)
class StepRecord:
    skill: str
    arguments: dict[str, Any]
    success: bool
    message: str
    environment_steps: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FailureRecord:
    source: str
    message: str
    subgoal: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeRecord:
    query: str
    content: str
    subgoal: str = ""


@dataclass(frozen=True)
class ReflectionRecord:
    skill: str
    subgoal: str
    matched: bool
    reason: str
    expected: dict[str, Any] = field(default_factory=dict)
    observed: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMemory:
    """Decision-oriented episode state grounded in agent-visible observations."""

    goal: str = ""
    task_status: str = _TASK_IDLE
    current_subgoal: str | None = None
    completed_subgoals: list[str] = field(default_factory=list)
    pending_subgoals: list[str] = field(default_factory=list)
    completed_steps: list[StepRecord] = field(default_factory=list)
    failed_attempts: list[FailureRecord] = field(default_factory=list)
    known_knowledge: dict[str, str] = field(default_factory=dict)
    knowledge_uses: list[KnowledgeRecord] = field(default_factory=list)
    reflections: list[ReflectionRecord] = field(default_factory=list)
    last_reflection: ReflectionRecord | None = None
    inventory: dict[str, int] = field(default_factory=dict)
    inventory_delta: dict[str, int] = field(default_factory=dict)
    selected_item: str | None = None
    last_error: str | None = None
    last_observation: Observation | None = field(default=None, repr=False)
    _state_initialized: bool = field(default=False, repr=False)

    def reset(self, goal: str) -> None:
        self.goal = goal.strip()
        self.task_status = _TASK_IDLE
        self.current_subgoal = None
        self.completed_subgoals.clear()
        self.pending_subgoals.clear()
        self.completed_steps.clear()
        self.failed_attempts.clear()
        self.known_knowledge.clear()
        self.knowledge_uses.clear()
        self.reflections.clear()
        self.last_reflection = None
        self.inventory.clear()
        self.inventory_delta.clear()
        self.selected_item = None
        self.last_error = None
        self.last_observation = None
        self._state_initialized = False

    def update_state(
        self,
        observation: Observation,
        *,
        baseline: dict[str, int] | None = None,
    ) -> None:
        new_inventory = dict(observation.inventory or {})
        if baseline is not None:
            self.inventory_delta = _inventory_delta(dict(baseline), new_inventory)
            self._state_initialized = True
        elif self._state_initialized:
            self.inventory_delta = _inventory_delta(self.inventory, new_inventory)
        else:
            self.inventory_delta = {}
            self._state_initialized = True
        self.last_observation = observation
        self.inventory = new_inventory
        self.selected_item = observation.selected_item
        if self.task_status == _TASK_IDLE and self.goal:
            self.task_status = _TASK_IN_PROGRESS

    def begin_subgoal(self, description: str) -> None:
        description = description.strip()
        if not description:
            return
        if self.current_subgoal and self.current_subgoal != description:
            if self.last_error is None:
                self._remember_completed_subgoal(self.current_subgoal)
        self.current_subgoal = description
        self.pending_subgoals = [item for item in self.pending_subgoals if item != description]
        if self.task_status == _TASK_IDLE:
            self.task_status = _TASK_IN_PROGRESS

    def apply_plan(self, subgoal: str, pending: list[str] | tuple[str, ...] = ()) -> None:
        self.begin_subgoal(subgoal)
        cleaned = [str(item).strip() for item in pending if str(item).strip()]
        current = self.current_subgoal or ""
        self.pending_subgoals = [item for item in cleaned if item != current]

    def complete_current_subgoal(self) -> None:
        if self.current_subgoal:
            self._remember_completed_subgoal(self.current_subgoal)
            self.current_subgoal = None

    def mark_task_completed(self) -> None:
        self.complete_current_subgoal()
        self.pending_subgoals.clear()
        self.task_status = _TASK_COMPLETED
        self.last_error = None

    def mark_task_failed(self, reason: str | None = None) -> None:
        self.task_status = _TASK_FAILED
        if reason:
            self.last_error = reason

    def remember_knowledge(self, query: str, content: str) -> None:
        normalized_query = query.strip()
        normalized_content = content.strip()
        self.known_knowledge[normalized_query] = normalized_content
        self.knowledge_uses.append(
            KnowledgeRecord(
                query=normalized_query,
                content=normalized_content[:240],
                subgoal=self.current_subgoal or "",
            )
        )

    def record_failure(
        self,
        *,
        source: str,
        message: str,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        self.failed_attempts.append(
            FailureRecord(
                source=source.strip() or "unknown",
                message=message,
                subgoal=self.current_subgoal or "",
                arguments=dict(arguments or {}),
            )
        )
        self.last_error = message

    def record_reflection(self, record: ReflectionRecord) -> None:
        self.reflections.append(record)
        self.last_reflection = record

    def record_step(self, record: StepRecord) -> None:
        self.completed_steps.append(record)
        if record.success:
            self.last_error = None
        else:
            self.record_failure(
                source=record.skill,
                message=record.message,
                arguments=record.arguments,
            )

    def prompt_state(self) -> dict[str, Any]:
        """Compact planner-facing state. Skill metadata is omitted on purpose."""
        reflection = None
        if self.last_reflection is not None:
            reflection = {
                "skill": self.last_reflection.skill,
                "subgoal": self.last_reflection.subgoal,
                "matched": self.last_reflection.matched,
                "reason": self.last_reflection.reason,
            }
        return {
            "task": self.goal,
            "task_status": self.task_status,
            "task_state": {"task": self.goal, "status": self.task_status},
            "current_subgoal": self.current_subgoal,
            "completed_subgoals": list(self.completed_subgoals[-8:]),
            "pending_subgoals": list(self.pending_subgoals[-8:]),
            "subgoal_progress": {
                "current": self.current_subgoal,
                "completed": list(self.completed_subgoals[-8:]),
                "pending": list(self.pending_subgoals[-8:]),
            },
            "environment": {
                "inventory": dict(self.inventory),
                "selected_item": self.selected_item,
                "inventory_delta": dict(self.inventory_delta),
                "has_visual_frame": self.last_observation is not None
                and self.last_observation.frame is not None,
            },
            "wiki_knowledge": dict(list(self.known_knowledge.items())[-8:]),
            "knowledge_usage": {
                "retrieved": dict(list(self.known_knowledge.items())[-8:]),
                "recent": [asdict(item) for item in self.knowledge_uses[-4:]],
            },
            "recent_failures": [asdict(item) for item in self.failed_attempts[-8:]],
            "failure_history": [asdict(item) for item in self.failed_attempts[-8:]],
            "recent_skills": [_compact_step(step) for step in self.completed_steps[-6:]],
            "last_reflection": reflection,
            "last_error": self.last_error,
        }

    def _remember_completed_subgoal(self, description: str) -> None:
        if not description:
            return
        if self.completed_subgoals and self.completed_subgoals[-1] == description:
            return
        self.completed_subgoals.append(description)
        self.pending_subgoals = [item for item in self.pending_subgoals if item != description]


def _inventory_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    delta: dict[str, int] = {}
    for name in set(before) | set(after):
        change = int(after.get(name, 0) or 0) - int(before.get(name, 0) or 0)
        if change:
            delta[name] = change
    return delta


def _compact_step(step: StepRecord) -> dict[str, Any]:
    return {
        "skill": step.skill,
        "arguments": dict(step.arguments),
        "success": step.success,
        "message": step.message,
        "environment_steps": step.environment_steps,
    }


__all__ = [
    "AgentMemory",
    "FailureRecord",
    "KnowledgeRecord",
    "ReflectionRecord",
    "StepRecord",
]

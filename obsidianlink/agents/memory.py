"""Layered memory for long-horizon GeneralAgent decisions.

The public ``AgentMemory`` surface remains compatible with the Phase 1 agent,
while separating working, episodic, semantic, and spatial memory. Only working
memory is cleared between tasks unless ``clear_long_term`` is requested.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from obsidianlink.env.environment import Observation

_TASK_IDLE = "idle"
_TASK_IN_PROGRESS = "in_progress"
_TASK_COMPLETED = "completed"
_TASK_FAILED = "failed"
_SUBGOAL_STATUSES = frozenset(
    {"pending", "in_progress", "completed", "failed", "blocked", "skipped"}
)


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
    knowledge_type: str = "general"
    subject: str = ""
    cache_hit: bool = False


@dataclass(frozen=True)
class SemanticMemoryRecord:
    """Reusable Minecraft fact grounded in an external or observed source."""

    key: str
    query: str
    knowledge_type: str
    subject: str
    summary: str
    attributes: dict[str, Any] = field(default_factory=dict)
    source_url: str | None = None
    retrieval_count: int = 1


@dataclass(frozen=True)
class EpisodicMemoryRecord:
    """A task attempt retained after working memory is reset."""

    task: str
    subgoal: str
    action: str
    success: bool
    outcome: str
    arguments: dict[str, Any] = field(default_factory=dict)
    inventory_delta: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SpatialMemoryRecord:
    """An agent-observed place/resource; coordinates are optional and visible-only."""

    key: str
    label: str
    position: tuple[float, float, float] | None = None
    dimension: str = "unknown"
    resources: dict[str, int] = field(default_factory=dict)
    notes: str = ""
    confidence: float = 1.0


@dataclass
class SubgoalState:
    """Stable hierarchical plan node tracked across planner calls."""

    id: str
    description: str
    status: str = "pending"
    parent_id: str | None = None
    depends_on: tuple[str, ...] = ()
    attempts: int = 0
    last_outcome: str = ""


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
    """Planner-facing layered memory grounded in agent-visible information."""

    # Working memory (legacy fields remain public for compatibility).
    goal: str = ""
    task_status: str = _TASK_IDLE
    current_subgoal: str | None = None
    completed_subgoals: list[str] = field(default_factory=list)
    pending_subgoals: list[str] = field(default_factory=list)
    subgoal_states: dict[str, SubgoalState] = field(default_factory=dict)
    active_subgoal_id: str | None = None
    plan_revision: int = 0
    plan_revision_reason: str = ""
    completed_steps: list[StepRecord] = field(default_factory=list)
    failed_attempts: list[FailureRecord] = field(default_factory=list)
    reflections: list[ReflectionRecord] = field(default_factory=list)
    last_reflection: ReflectionRecord | None = None
    inventory: dict[str, int] = field(default_factory=dict)
    inventory_delta: dict[str, int] = field(default_factory=dict)
    selected_item: str | None = None
    last_error: str | None = None
    last_observation: Observation | None = field(default=None, repr=False)

    # Compatibility views plus durable memory stores.
    known_knowledge: dict[str, str] = field(default_factory=dict)
    knowledge_uses: list[KnowledgeRecord] = field(default_factory=list)
    semantic_memory: dict[str, SemanticMemoryRecord] = field(default_factory=dict)
    episodic_memory: list[EpisodicMemoryRecord] = field(default_factory=list)
    spatial_memory: dict[str, SpatialMemoryRecord] = field(default_factory=dict)
    _state_initialized: bool = field(default=False, repr=False)

    def reset(self, goal: str, *, clear_long_term: bool = False) -> None:
        """Start a task, preserving semantic/episodic/spatial memory by default."""
        self.goal = goal.strip()
        self.task_status = _TASK_IDLE
        self.current_subgoal = None
        self.completed_subgoals.clear()
        self.pending_subgoals.clear()
        self.subgoal_states.clear()
        self.active_subgoal_id = None
        self.plan_revision = 0
        self.plan_revision_reason = ""
        self.completed_steps.clear()
        self.failed_attempts.clear()
        self.knowledge_uses.clear()
        self.reflections.clear()
        self.last_reflection = None
        self.inventory.clear()
        self.inventory_delta.clear()
        self.selected_item = None
        self.last_error = None
        self.last_observation = None
        self._state_initialized = False
        if clear_long_term:
            self.known_knowledge.clear()
            self.semantic_memory.clear()
            self.episodic_memory.clear()
            self.spatial_memory.clear()

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

    def begin_subgoal(self, description: str, *, subgoal_id: str = "") -> None:
        description = description.strip()
        if not description:
            return
        if self.current_subgoal and self.current_subgoal != description:
            if self.last_error is None:
                self._remember_completed_subgoal(self.current_subgoal)
        node_id = subgoal_id.strip() or _subgoal_key(description)
        node = self.subgoal_states.get(node_id)
        if node is None:
            node = SubgoalState(node_id, description, status="in_progress")
            self.subgoal_states[node_id] = node
        elif node.status not in {"completed", "skipped"}:
            node.status = "in_progress"
        self.active_subgoal_id = node_id
        self.current_subgoal = description
        self.pending_subgoals = [item for item in self.pending_subgoals if item != description]
        if self.task_status == _TASK_IDLE:
            self.task_status = _TASK_IN_PROGRESS

    def apply_plan(
        self,
        subgoal: str,
        pending: list[str] | tuple[str, ...] = (),
        *,
        plan: Iterable[Any] = (),
        active_subgoal_id: str = "",
        revision_reason: str = "",
    ) -> None:
        """Merge a planner update; accepts both legacy strings and plan nodes."""
        plan_items = tuple(plan)
        if plan_items:
            self._merge_plan(plan_items)
            chosen_id = active_subgoal_id.strip()
            if not chosen_id:
                chosen_id = next(
                    (node.id for node in self.subgoal_states.values() if node.status == "in_progress"),
                    "",
                )
            chosen = self.subgoal_states.get(chosen_id)
            if chosen is not None and chosen.status not in {"completed", "skipped"}:
                self.active_subgoal_id = chosen.id
                self.current_subgoal = chosen.description
            elif any(node.status == "in_progress" for node in self.subgoal_states.values()):
                fallback = next(
                    node for node in self.subgoal_states.values() if node.status == "in_progress"
                )
                self.active_subgoal_id = fallback.id
                self.current_subgoal = fallback.description
            else:
                self.active_subgoal_id = None
                self.current_subgoal = None
            self._sync_legacy_plan_views()
            self.plan_revision += 1
            self.plan_revision_reason = revision_reason.strip()
            return

        self.begin_subgoal(subgoal)
        cleaned = [str(item).strip() for item in pending if str(item).strip()]
        current = self.current_subgoal or ""
        self.pending_subgoals = [item for item in cleaned if item != current]
        for item in self.pending_subgoals:
            key = _subgoal_key(item)
            self.subgoal_states.setdefault(key, SubgoalState(key, item))

    def complete_current_subgoal(self) -> None:
        if self.current_subgoal:
            self._remember_completed_subgoal(self.current_subgoal)
            if self.active_subgoal_id in self.subgoal_states:
                self.subgoal_states[self.active_subgoal_id].status = "completed"
            self.current_subgoal = None
            self.active_subgoal_id = None

    def mark_task_completed(self) -> None:
        self.complete_current_subgoal()
        self.pending_subgoals.clear()
        self.task_status = _TASK_COMPLETED
        self.last_error = None

    def mark_task_failed(self, reason: str | None = None) -> None:
        self.task_status = _TASK_FAILED
        if reason:
            self.last_error = reason

    def find_knowledge(self, query: str) -> SemanticMemoryRecord | None:
        return self.semantic_memory.get(_normalize_key(query))

    def remember_knowledge(
        self,
        query: str,
        content: str,
        *,
        knowledge_type: str = "general",
        subject: str = "",
        attributes: dict[str, Any] | None = None,
        source_url: str | None = None,
        cache_hit: bool = False,
    ) -> SemanticMemoryRecord:
        normalized_query = query.strip()
        normalized_content = content.strip()
        key = _normalize_key(normalized_query)
        previous = self.semantic_memory.get(key)
        record = SemanticMemoryRecord(
            key=key,
            query=normalized_query,
            knowledge_type=knowledge_type.strip() or "general",
            subject=subject.strip(),
            summary=normalized_content,
            attributes=dict(attributes or (previous.attributes if previous else {})),
            source_url=source_url or (previous.source_url if previous else None),
            retrieval_count=(previous.retrieval_count + 1) if previous else 1,
        )
        self.semantic_memory[key] = record
        self.known_knowledge[normalized_query] = normalized_content
        self.knowledge_uses.append(
            KnowledgeRecord(
                query=normalized_query,
                content=normalized_content[:240],
                subgoal=self.current_subgoal or "",
                knowledge_type=record.knowledge_type,
                subject=record.subject,
                cache_hit=cache_hit,
            )
        )
        return record

    def record_knowledge_use(self, record: SemanticMemoryRecord, *, cache_hit: bool) -> None:
        self.knowledge_uses.append(
            KnowledgeRecord(
                query=record.query,
                content=record.summary[:240],
                subgoal=self.current_subgoal or "",
                knowledge_type=record.knowledge_type,
                subject=record.subject,
                cache_hit=cache_hit,
            )
        )

    def remember_location(
        self,
        label: str,
        *,
        position: tuple[float, float, float] | None = None,
        dimension: str = "unknown",
        resources: dict[str, int] | None = None,
        notes: str = "",
        confidence: float = 1.0,
    ) -> SpatialMemoryRecord:
        """Store only location information observed or supplied by the agent."""
        key = _normalize_key(f"{dimension}:{label}")
        record = SpatialMemoryRecord(
            key=key,
            label=label.strip(),
            position=position,
            dimension=dimension.strip() or "unknown",
            resources={str(k): int(v) for k, v in (resources or {}).items()},
            notes=notes.strip(),
            confidence=max(0.0, min(1.0, float(confidence))),
        )
        self.spatial_memory[key] = record
        return record

    def record_failure(
        self,
        *,
        source: str,
        message: str,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        record = FailureRecord(
            source=source.strip() or "unknown",
            message=message,
            subgoal=self.current_subgoal or "",
            arguments=dict(arguments or {}),
        )
        self.failed_attempts.append(record)
        self.episodic_memory.append(
            EpisodicMemoryRecord(
                task=self.goal,
                subgoal=record.subgoal,
                action=record.source,
                success=False,
                outcome=record.message,
                arguments=record.arguments,
                inventory_delta=dict(self.inventory_delta),
            )
        )
        if self.active_subgoal_id in self.subgoal_states:
            self.subgoal_states[self.active_subgoal_id].last_outcome = message
        self.last_error = message

    def record_reflection(self, record: ReflectionRecord) -> None:
        self.reflections.append(record)
        self.last_reflection = record
        if self.active_subgoal_id in self.subgoal_states:
            self.subgoal_states[self.active_subgoal_id].last_outcome = record.reason

    def record_step(self, record: StepRecord) -> None:
        self.completed_steps.append(record)
        if self.active_subgoal_id in self.subgoal_states:
            node = self.subgoal_states[self.active_subgoal_id]
            node.attempts += 1
            node.last_outcome = record.message
        if record.success:
            self.last_error = None
            self.episodic_memory.append(
                EpisodicMemoryRecord(
                    task=self.goal,
                    subgoal=self.current_subgoal or "",
                    action=record.skill,
                    success=True,
                    outcome=record.message,
                    arguments=dict(record.arguments),
                    inventory_delta=dict(self.inventory_delta),
                )
            )
        else:
            self.record_failure(
                source=record.skill,
                message=record.message,
                arguments=record.arguments,
            )

    def prompt_state(self) -> dict[str, Any]:
        """Compact planner-facing views of all four memory layers."""
        reflection = None
        if self.last_reflection is not None:
            reflection = {
                "skill": self.last_reflection.skill,
                "subgoal": self.last_reflection.subgoal,
                "matched": self.last_reflection.matched,
                "reason": self.last_reflection.reason,
            }
        plan_nodes = [asdict(node) for node in self.subgoal_states.values()]
        working = {
            "task": self.goal,
            "status": self.task_status,
            "active_subgoal_id": self.active_subgoal_id,
            "plan_revision": self.plan_revision,
            "plan_revision_reason": self.plan_revision_reason,
            "plan": plan_nodes[-12:],
            "inventory": dict(self.inventory),
            "selected_item": self.selected_item,
            "inventory_delta": dict(self.inventory_delta),
        }
        semantic = [asdict(item) for item in self.semantic_memory.values()]
        spatial = [asdict(item) for item in self.spatial_memory.values()]
        episodic_failures = [item for item in self.episodic_memory if not item.success]
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
                "active_id": self.active_subgoal_id,
                "nodes": plan_nodes[-12:],
            },
            "working_memory": working,
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
            "semantic_memory": semantic[-8:],
            "episodic_memory": [asdict(item) for item in self.episodic_memory[-10:]],
            "relevant_failure_experience": [
                asdict(item) for item in episodic_failures[-6:]
            ],
            "spatial_memory": spatial[-8:],
            "recent_failures": [asdict(item) for item in self.failed_attempts[-8:]],
            "failure_history": [asdict(item) for item in self.failed_attempts[-8:]],
            "recent_skills": [_compact_step(step) for step in self.completed_steps[-6:]],
            "last_reflection": reflection,
            "last_error": self.last_error,
        }

    def _merge_plan(self, plan: tuple[Any, ...]) -> None:
        for raw in plan:
            node_id = str(getattr(raw, "id", "") or "").strip()
            description = str(getattr(raw, "description", "") or "").strip()
            if not node_id or not description:
                continue
            status = str(getattr(raw, "status", "pending") or "pending").strip()
            if status not in _SUBGOAL_STATUSES:
                status = "pending"
            parent = str(getattr(raw, "parent_id", "") or "").strip() or None
            dependencies = tuple(
                str(item).strip()
                for item in (getattr(raw, "depends_on", ()) or ())
                if str(item).strip()
            )
            previous = self.subgoal_states.get(node_id)
            # Completed work is monotonic. A later model call cannot silently
            # reopen it and cause a long-horizon plan to loop.
            if previous is not None and previous.status in {"completed", "skipped"}:
                status = previous.status
            self.subgoal_states[node_id] = SubgoalState(
                id=node_id,
                description=description,
                status=status,
                parent_id=parent,
                depends_on=dependencies,
                attempts=previous.attempts if previous else 0,
                last_outcome=previous.last_outcome if previous else "",
            )

    def _sync_legacy_plan_views(self) -> None:
        self.completed_subgoals = [
            node.description for node in self.subgoal_states.values() if node.status == "completed"
        ]
        self.pending_subgoals = [
            node.description
            for node in self.subgoal_states.values()
            if node.status in {"pending", "blocked", "failed"}
        ]

    def _remember_completed_subgoal(self, description: str) -> None:
        if not description:
            return
        if not self.completed_subgoals or self.completed_subgoals[-1] != description:
            self.completed_subgoals.append(description)
        self.pending_subgoals = [item for item in self.pending_subgoals if item != description]
        for node in self.subgoal_states.values():
            if node.description == description:
                node.status = "completed"


def _normalize_key(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def _subgoal_key(description: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "-", description.casefold()).strip("-")
    return key or f"subgoal-{abs(hash(description))}"


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
    "EpisodicMemoryRecord",
    "FailureRecord",
    "KnowledgeRecord",
    "ReflectionRecord",
    "SemanticMemoryRecord",
    "SpatialMemoryRecord",
    "StepRecord",
    "SubgoalState",
]

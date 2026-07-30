from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


def _require_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True)
class EvaluationState:
    """Evaluator-only truth. This object must never enter an agent observation."""

    episode_id: str
    step_id: int
    portal_built_by_episode: bool = False
    valid_portal_frame: bool = False
    portal_activated: bool = False
    agents_in_nether: frozenset[str] = frozenset()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.episode_id, "episode_id")
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative integer")
        for name in (
            "portal_built_by_episode",
            "valid_portal_frame",
            "portal_activated",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for agent_id in self.agents_in_nether:
            _require_identifier(agent_id, "agent_id")
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))


@dataclass(frozen=True)
class EvaluationResult:
    episode_id: str
    step_id: int
    success: bool
    milestones: tuple[str, ...]
    blocking_conditions: tuple[str, ...]
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))


class PortalEvaluator:
    """Evaluate portal completion from environment truth, never model claims."""

    def evaluate(self, state: EvaluationState) -> EvaluationResult:
        built_frame = (
            state.portal_built_by_episode and state.valid_portal_frame
        )
        activated = built_frame and state.portal_activated
        entered = activated and bool(state.agents_in_nether)
        milestones: list[str] = []
        if built_frame:
            milestones.append("valid_portal_frame")
        if activated:
            milestones.append("portal_activated")
        if entered:
            milestones.append("agent_entered_nether")

        blocking: list[str] = []
        if not state.portal_built_by_episode:
            blocking.append("portal_not_built_by_episode")
        if not state.valid_portal_frame:
            blocking.append("invalid_portal_frame")
        if not state.portal_activated:
            blocking.append("portal_not_activated")
        if not state.agents_in_nether:
            blocking.append("no_agent_entered_nether")

        return EvaluationResult(
            episode_id=state.episode_id,
            step_id=state.step_id,
            success=entered,
            milestones=tuple(milestones),
            blocking_conditions=tuple(blocking),
            evidence=state.evidence,
        )

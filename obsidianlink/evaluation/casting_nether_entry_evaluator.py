"""Offline Casting-S-C5 Nether-entry evaluator.

The evaluator extends the frozen C4 ignition verdict with typed, evaluator-only
dimension-transition evidence.  It never reads observations, drivers, task
scenario parameters, or backend state directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from obsidianlink.evaluation.casting import NORMAL_TERMINATION_REASONS
from obsidianlink.evaluation.casting_ignition_evaluator import (
    CASTING_S_C4_AGENT_ID,
    FrozenFrameIdentity,
    FrozenIgnitionEvaluationState,
    FrozenIgnitionEvaluator,
    OUTCOME_IN_PROGRESS as IGNITION_OUTCOME_IN_PROGRESS,
    OUTCOME_SUCCESS as IGNITION_OUTCOME_SUCCESS,
)


CASTING_S_C5_AGENT_ID = CASTING_S_C4_AGENT_ID
CASTING_S_C5_SOURCE_DIMENSION = "minecraft:overworld"
CASTING_S_C5_TARGET_DIMENSION = "minecraft:the_nether"

OUTCOME_SUCCESS = "success"
OUTCOME_IN_PROGRESS = "in_progress"
OUTCOME_IGNITION_NOT_COMPLETED = "ignition_not_completed"
OUTCOME_NO_AGENT_ENTERED_NETHER = "no_agent_entered_nether"
OUTCOME_WRONG_ENTRY_AGENT = "wrong_entry_agent"
OUTCOME_WRONG_SOURCE_DIMENSION = "wrong_source_dimension"
OUTCOME_WRONG_TARGET_DIMENSION = "wrong_target_dimension"
OUTCOME_TRANSITION_STEP_MISSING = "transition_step_missing"
OUTCOME_TRANSITION_BEFORE_ACTIVATION = "transition_before_activation"
OUTCOME_PRE_TRANSITION_POSITION_MISSING = "pre_transition_position_missing"
OUTCOME_NETHER_ENTRY_PORTAL_UNKNOWN = "nether_entry_portal_unknown"
OUTCOME_NETHER_ENTRY_NOT_VIA_EPISODE_PORTAL = (
    "nether_entry_not_via_episode_portal"
)
OUTCOME_FRAME_IDENTITY_MISSING = "frame_identity_missing"
OUTCOME_FRAME_IDENTITY_MISMATCH = "frame_identity_mismatch"
OUTCOME_STEP_BUDGET_EXCEEDED = "step_budget_exceeded"
OUTCOME_TIME_BUDGET_EXCEEDED = "time_budget_exceeded"
OUTCOME_ABNORMAL_TERMINATION = "abnormal_termination"

C5_NETHER_ENTRY_OUTCOMES = frozenset(
    {
        OUTCOME_SUCCESS,
        OUTCOME_IN_PROGRESS,
        OUTCOME_IGNITION_NOT_COMPLETED,
        OUTCOME_NO_AGENT_ENTERED_NETHER,
        OUTCOME_WRONG_ENTRY_AGENT,
        OUTCOME_WRONG_SOURCE_DIMENSION,
        OUTCOME_WRONG_TARGET_DIMENSION,
        OUTCOME_TRANSITION_STEP_MISSING,
        OUTCOME_TRANSITION_BEFORE_ACTIVATION,
        OUTCOME_PRE_TRANSITION_POSITION_MISSING,
        OUTCOME_NETHER_ENTRY_PORTAL_UNKNOWN,
        OUTCOME_NETHER_ENTRY_NOT_VIA_EPISODE_PORTAL,
        OUTCOME_FRAME_IDENTITY_MISSING,
        OUTCOME_FRAME_IDENTITY_MISMATCH,
        OUTCOME_STEP_BUDGET_EXCEEDED,
        OUTCOME_TIME_BUDGET_EXCEEDED,
        OUTCOME_ABNORMAL_TERMINATION,
    }
)

_TERMINAL_FAILURES = C5_NETHER_ENTRY_OUTCOMES - {
    OUTCOME_SUCCESS,
    OUTCOME_IN_PROGRESS,
}


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative strict integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive strict integer")
    return value


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _non_negative_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-negative finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a non-negative finite number")
    return result


def _position(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be an xyz sequence")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name} components must be finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name} components must be finite numbers")
        result.append(number)
    return (result[0], result[1], result[2])


def _freeze_json(value: Any, name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain finite numbers")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} keys must be strings")
            result[key] = _freeze_json(item, f"{name}.{key}")
        return MappingProxyType(result)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{name}[]") for item in value)
    raise ValueError(f"{name} must be JSON-compatible")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class NetherEntryEvidence:
    """Typed transition evidence; malformed structure fails at construction.

    Optional attribution fields deliberately preserve ``unknown`` separately
    from a known negative value so the evaluator can implement the frozen C5
    ``unknown_attribution_outcome`` contract.
    """

    episode_id: str
    agent_id: str
    source_dimension: str
    target_dimension: str
    transition_step: int | None
    pre_transition_position: tuple[float, float, float] | None
    entered_via_episode_portal: bool | None
    matched_frame_identity: FrozenFrameIdentity | None

    def __post_init__(self) -> None:
        _identifier(self.episode_id, "episode_id")
        _identifier(self.agent_id, "agent_id")
        _identifier(self.source_dimension, "source_dimension")
        _identifier(self.target_dimension, "target_dimension")
        if self.transition_step is not None:
            _non_negative_int(self.transition_step, "transition_step")
        if self.pre_transition_position is not None:
            object.__setattr__(
                self,
                "pre_transition_position",
                _position(self.pre_transition_position, "pre_transition_position"),
            )
        if self.entered_via_episode_portal is not None and type(
            self.entered_via_episode_portal
        ) is not bool:
            raise ValueError("entered_via_episode_portal must be bool or None")
        if self.matched_frame_identity is not None and not isinstance(
            self.matched_frame_identity, FrozenFrameIdentity
        ):
            raise ValueError(
                "matched_frame_identity must be FrozenFrameIdentity or None"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "agent_id": self.agent_id,
            "source_dimension": self.source_dimension,
            "target_dimension": self.target_dimension,
            "transition_step": self.transition_step,
            "pre_transition_position": (
                list(self.pre_transition_position)
                if self.pre_transition_position is not None
                else None
            ),
            "entered_via_episode_portal": self.entered_via_episode_portal,
            "matched_frame_identity": (
                self.matched_frame_identity.as_dict()
                if self.matched_frame_identity is not None
                else None
            ),
        }


@dataclass(frozen=True)
class FrozenNetherEntryEvaluationState:
    """The complete evaluator-only truth input for Casting-S-C5."""

    episode_id: str
    step_id: int
    ignition_state: FrozenIgnitionEvaluationState
    agents_in_nether: frozenset[str] = field(default_factory=frozenset)
    entry_evidence: NetherEntryEvidence | None = None
    agent_id: str = CASTING_S_C5_AGENT_ID
    episode_terminated: bool = False
    terminated_step: int | None = None
    terminated_reason: str | None = None
    current_time_seconds: float = 0.0
    max_environment_steps: int = 1
    max_game_time_seconds: float = 1.0
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identifier(self.episode_id, "episode_id")
        _non_negative_int(self.step_id, "step_id")
        _identifier(self.agent_id, "agent_id")
        if self.agent_id != CASTING_S_C5_AGENT_ID:
            raise ValueError(f"agent_id must be {CASTING_S_C5_AGENT_ID!r}")
        if not isinstance(self.ignition_state, FrozenIgnitionEvaluationState):
            raise ValueError("ignition_state must be FrozenIgnitionEvaluationState")
        for name, actual, expected in (
            ("episode_id", self.ignition_state.episode_id, self.episode_id),
            ("step_id", self.ignition_state.step_id, self.step_id),
            ("agent_id", self.ignition_state.agent_id, self.agent_id),
            (
                "max_environment_steps",
                self.ignition_state.max_environment_steps,
                self.max_environment_steps,
            ),
            (
                "max_game_time_seconds",
                self.ignition_state.max_game_time_seconds,
                self.max_game_time_seconds,
            ),
            (
                "current_time_seconds",
                self.ignition_state.current_time_seconds,
                self.current_time_seconds,
            ),
        ):
            if actual != expected:
                raise ValueError(f"ignition_state.{name} must match C5 state")
        if not isinstance(self.agents_in_nether, (set, frozenset)):
            raise ValueError("agents_in_nether must be a set or frozenset")
        frozen_agents = frozenset(self.agents_in_nether)
        for agent in frozen_agents:
            _identifier(agent, "agents_in_nether item")
        object.__setattr__(self, "agents_in_nether", frozen_agents)
        if self.entry_evidence is not None:
            if not isinstance(self.entry_evidence, NetherEntryEvidence):
                raise ValueError("entry_evidence must be NetherEntryEvidence or None")
            if self.entry_evidence.episode_id != self.episode_id:
                raise ValueError("entry_evidence.episode_id must match C5 state")
            if (
                self.entry_evidence.transition_step is not None
                and self.entry_evidence.transition_step > self.step_id
            ):
                raise ValueError("entry transition_step cannot be in the future")
        if type(self.episode_terminated) is not bool:
            raise ValueError("episode_terminated must be a boolean")
        if self.episode_terminated:
            if self.terminated_step is None:
                raise ValueError("terminated episode requires terminated_step")
            _non_negative_int(self.terminated_step, "terminated_step")
            if self.terminated_step > self.step_id:
                raise ValueError("terminated_step cannot be in the future")
            if self.terminated_reason is not None:
                _identifier(self.terminated_reason, "terminated_reason")
        elif self.terminated_step is not None or self.terminated_reason is not None:
            raise ValueError("termination metadata requires episode_terminated=True")
        if self.ignition_state.episode_terminated != self.episode_terminated:
            raise ValueError("ignition_state termination must match C5 state")
        if self.ignition_state.terminated_step != self.terminated_step:
            raise ValueError("ignition_state.terminated_step must match C5 state")
        if self.ignition_state.terminated_reason != self.terminated_reason:
            raise ValueError("ignition_state.terminated_reason must match C5 state")
        _non_negative_number(self.current_time_seconds, "current_time_seconds")
        _positive_int(self.max_environment_steps, "max_environment_steps")
        _positive_number(self.max_game_time_seconds, "max_game_time_seconds")
        if not isinstance(self.evidence, Mapping):
            raise ValueError("evidence must be a mapping")
        object.__setattr__(self, "evidence", _freeze_json(self.evidence, "evidence"))


@dataclass(frozen=True)
class FrozenNetherEntryEvaluationResult:
    episode_id: str
    step_id: int
    success: bool
    outcome: str
    ignition_outcome: str
    entry_agent_id: str | None
    transition_step: int | None
    source_dimension: str | None
    target_dimension: str | None
    pre_transition_position: tuple[float, float, float] | None
    entered_via_episode_portal: bool | None
    frame_identity_matched: bool | None
    blocking_conditions: tuple[str, ...]
    evidence: Mapping[str, Any]
    failure_type: str | None = None
    failure_step: int | None = None
    episode_terminated: bool = False
    terminated_step: int | None = None
    terminated_reason: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.episode_id, "episode_id")
        _non_negative_int(self.step_id, "step_id")
        if type(self.success) is not bool or self.success != (
            self.outcome == OUTCOME_SUCCESS
        ):
            raise ValueError("success must equal (outcome == 'success')")
        if self.outcome not in C5_NETHER_ENTRY_OUTCOMES:
            raise ValueError(f"unknown C5 outcome: {self.outcome!r}")
        _identifier(self.ignition_outcome, "ignition_outcome")
        if self.entry_agent_id is not None:
            _identifier(self.entry_agent_id, "entry_agent_id")
        if self.source_dimension is not None:
            _identifier(self.source_dimension, "source_dimension")
        if self.target_dimension is not None:
            _identifier(self.target_dimension, "target_dimension")
        if self.transition_step is not None:
            _non_negative_int(self.transition_step, "transition_step")
        if self.pre_transition_position is not None:
            object.__setattr__(
                self,
                "pre_transition_position",
                _position(self.pre_transition_position, "pre_transition_position"),
            )
        for name in ("entered_via_episode_portal", "frame_identity_matched"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{name} must be bool or None")
        expected_failure = self.outcome if self.outcome in _TERMINAL_FAILURES else None
        if self.failure_type != expected_failure:
            raise ValueError("failure_type must match terminal outcome")
        if self.failure_step is not None:
            _non_negative_int(self.failure_step, "failure_step")
        if not isinstance(self.blocking_conditions, tuple):
            raise ValueError("blocking_conditions must be a tuple")
        for condition in self.blocking_conditions:
            _identifier(condition, "blocking_conditions item")
        if type(self.episode_terminated) is not bool:
            raise ValueError("episode_terminated must be a boolean")
        if self.episode_terminated:
            if self.terminated_step is None:
                raise ValueError("terminated result requires terminated_step")
            _non_negative_int(self.terminated_step, "terminated_step")
            if self.terminated_step > self.step_id:
                raise ValueError("terminated_step cannot be in the future")
            if self.terminated_reason is not None:
                _identifier(self.terminated_reason, "terminated_reason")
        elif self.terminated_step is not None or self.terminated_reason is not None:
            raise ValueError("termination metadata requires episode_terminated=True")
        object.__setattr__(self, "evidence", _freeze_json(self.evidence, "evidence"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "success": self.success,
            "outcome": self.outcome,
            "ignition_outcome": self.ignition_outcome,
            "entry_agent_id": self.entry_agent_id,
            "transition_step": self.transition_step,
            "source_dimension": self.source_dimension,
            "target_dimension": self.target_dimension,
            "pre_transition_position": (
                list(self.pre_transition_position)
                if self.pre_transition_position is not None
                else None
            ),
            "entered_via_episode_portal": self.entered_via_episode_portal,
            "frame_identity_matched": self.frame_identity_matched,
            "blocking_conditions": list(self.blocking_conditions),
            "evidence": _thaw(self.evidence),
            "failure_type": self.failure_type,
            "failure_step": self.failure_step,
            "episode_terminated": self.episode_terminated,
            "terminated_step": self.terminated_step,
            "terminated_reason": self.terminated_reason,
        }


def _outcome(
    state: FrozenNetherEntryEvaluationState,
    ignition_outcome: str,
) -> tuple[str, int | None, bool | None]:
    latest = max(
        state.step_id,
        state.terminated_step if state.terminated_step is not None else 0,
    )
    if latest > state.max_environment_steps:
        return OUTCOME_STEP_BUDGET_EXCEEDED, latest, None
    if state.current_time_seconds > state.max_game_time_seconds:
        return OUTCOME_TIME_BUDGET_EXCEEDED, state.step_id, None
    if (
        state.episode_terminated
        and state.terminated_reason is not None
        and state.terminated_reason not in NORMAL_TERMINATION_REASONS
    ):
        return OUTCOME_ABNORMAL_TERMINATION, state.terminated_step, None
    if ignition_outcome == IGNITION_OUTCOME_IN_PROGRESS:
        return OUTCOME_IN_PROGRESS, None, None
    if ignition_outcome != IGNITION_OUTCOME_SUCCESS:
        return OUTCOME_IGNITION_NOT_COMPLETED, state.step_id, None
    if state.agent_id not in state.agents_in_nether:
        if not state.episode_terminated:
            return OUTCOME_IN_PROGRESS, None, None
        return OUTCOME_NO_AGENT_ENTERED_NETHER, state.terminated_step, None
    entry = state.entry_evidence
    if entry is None:
        return OUTCOME_NETHER_ENTRY_PORTAL_UNKNOWN, state.step_id, None
    if entry.agent_id != state.agent_id:
        return OUTCOME_WRONG_ENTRY_AGENT, entry.transition_step, None
    if entry.source_dimension != CASTING_S_C5_SOURCE_DIMENSION:
        return OUTCOME_WRONG_SOURCE_DIMENSION, entry.transition_step, None
    if entry.target_dimension != CASTING_S_C5_TARGET_DIMENSION:
        return OUTCOME_WRONG_TARGET_DIMENSION, entry.transition_step, None
    if entry.transition_step is None:
        return OUTCOME_TRANSITION_STEP_MISSING, state.step_id, None
    if entry.pre_transition_position is None:
        return OUTCOME_PRE_TRANSITION_POSITION_MISSING, entry.transition_step, None
    activation = state.ignition_state.activation_evidence
    if activation is None:
        return OUTCOME_IGNITION_NOT_COMPLETED, entry.transition_step, None
    if entry.transition_step < activation.update_step:
        return OUTCOME_TRANSITION_BEFORE_ACTIVATION, entry.transition_step, None
    if entry.entered_via_episode_portal is None:
        return OUTCOME_NETHER_ENTRY_PORTAL_UNKNOWN, entry.transition_step, None
    if entry.entered_via_episode_portal is False:
        return OUTCOME_NETHER_ENTRY_NOT_VIA_EPISODE_PORTAL, entry.transition_step, None
    if entry.matched_frame_identity is None:
        return OUTCOME_FRAME_IDENTITY_MISSING, entry.transition_step, None
    identity_match = (
        entry.matched_frame_identity.as_dict()
        == state.ignition_state.latched_frame_identity.as_dict()
    )
    if not identity_match:
        return OUTCOME_FRAME_IDENTITY_MISMATCH, entry.transition_step, False
    if not state.episode_terminated:
        return OUTCOME_IN_PROGRESS, None, True
    return OUTCOME_SUCCESS, None, True


class FrozenNetherEntryEvaluator:
    """Pure deterministic evaluator for ``casting_s_c5_fixed``."""

    def evaluate(
        self, state: FrozenNetherEntryEvaluationState
    ) -> FrozenNetherEntryEvaluationResult:
        ignition_result = FrozenIgnitionEvaluator().evaluate(state.ignition_state)
        outcome, failure_step, identity_match = _outcome(
            state, ignition_result.outcome
        )
        entry = state.entry_evidence
        blocking = () if outcome == OUTCOME_SUCCESS else (
            "episode_not_terminated" if outcome == OUTCOME_IN_PROGRESS else outcome,
        )
        evidence = {
            "episode_id": state.episode_id,
            "step_id": state.step_id,
            "agent_id": state.agent_id,
            "agents_in_nether": sorted(state.agents_in_nether),
            "ignition_outcome": ignition_result.outcome,
            "entry_evidence_present": entry is not None,
            "entry": entry.as_dict() if entry is not None else None,
            "state_evidence": _thaw(state.evidence),
        }
        return FrozenNetherEntryEvaluationResult(
            episode_id=state.episode_id,
            step_id=state.step_id,
            success=outcome == OUTCOME_SUCCESS,
            outcome=outcome,
            ignition_outcome=ignition_result.outcome,
            entry_agent_id=entry.agent_id if entry is not None else None,
            transition_step=entry.transition_step if entry is not None else None,
            source_dimension=entry.source_dimension if entry is not None else None,
            target_dimension=entry.target_dimension if entry is not None else None,
            pre_transition_position=(
                entry.pre_transition_position if entry is not None else None
            ),
            entered_via_episode_portal=(
                entry.entered_via_episode_portal if entry is not None else None
            ),
            frame_identity_matched=identity_match,
            blocking_conditions=blocking,
            evidence=evidence,
            failure_type=outcome if outcome in _TERMINAL_FAILURES else None,
            failure_step=failure_step,
            episode_terminated=state.episode_terminated,
            terminated_step=state.terminated_step,
            terminated_reason=state.terminated_reason,
        )


__all__ = [
    "C5_NETHER_ENTRY_OUTCOMES",
    "CASTING_S_C5_AGENT_ID",
    "CASTING_S_C5_SOURCE_DIMENSION",
    "CASTING_S_C5_TARGET_DIMENSION",
    "FrozenNetherEntryEvaluationResult",
    "FrozenNetherEntryEvaluationState",
    "FrozenNetherEntryEvaluator",
    "NetherEntryEvidence",
    "OUTCOME_ABNORMAL_TERMINATION",
    "OUTCOME_FRAME_IDENTITY_MISMATCH",
    "OUTCOME_FRAME_IDENTITY_MISSING",
    "OUTCOME_IGNITION_NOT_COMPLETED",
    "OUTCOME_IN_PROGRESS",
    "OUTCOME_NETHER_ENTRY_NOT_VIA_EPISODE_PORTAL",
    "OUTCOME_NETHER_ENTRY_PORTAL_UNKNOWN",
    "OUTCOME_NO_AGENT_ENTERED_NETHER",
    "OUTCOME_PRE_TRANSITION_POSITION_MISSING",
    "OUTCOME_STEP_BUDGET_EXCEEDED",
    "OUTCOME_SUCCESS",
    "OUTCOME_TIME_BUDGET_EXCEEDED",
    "OUTCOME_TRANSITION_BEFORE_ACTIVATION",
    "OUTCOME_TRANSITION_STEP_MISSING",
    "OUTCOME_WRONG_ENTRY_AGENT",
    "OUTCOME_WRONG_SOURCE_DIMENSION",
    "OUTCOME_WRONG_TARGET_DIMENSION",
]

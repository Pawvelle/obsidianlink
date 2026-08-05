"""Offline casting evaluator for ``casting_c1_fixed``.

This module defines a small, frozen, type-strict evaluator for the
minimum ``casting_c1_fixed`` task: turning a single non-obsidian
target cell into obsidian via a vanilla water + lava interaction. It
is deliberately independent of :class:`obsidianlink.evaluation.portal.PortalEvaluator`
(the frame / activation / Nether transition evaluator) and is meant
to be exercised on the :class:`obsidianlink.env.fake.FakeEnvironmentBackend`
without starting Minecraft, MineRL, or the model.

The module never reads Agent text, prompts, or images. Its inputs
are *only* typed evaluator-only truth (block ids, fluid evidence,
relevant action steps, budget numbers, termination signals). The
output is a :class:`CastingEvaluationResult` with a stable
``outcome`` id, blocking conditions, and JSON-serializable evidence.

Stability contract
------------------

* The :class:`CastingEvaluationState` dataclass is frozen. All fields
  are validated in ``__post_init__``; an invalid state raises
  :class:`ValueError` or :class:`TypeError` *before* the evaluator
  ever sees it. The evaluator itself never mutates the state.
* The :class:`CastingEvaluationResult` dataclass is frozen. Its
  ``evidence`` tree is recursively frozen and :meth:`as_dict`
  returns a detached JSON-serializable snapshot; ``outcome`` and
  ``failure_type`` use a closed set of stable string ids.
* :class:`CastingEvaluator` is a *pure* deterministic object.
  ``evaluate()`` has no side effects, reads no global state, and
  returns the same ``CastingEvaluationResult`` for the same state
  on repeated calls. The evaluator never instantiates an
  :class:`Observation`, never reads an Agent prompt, and never
  imports from any other module of the project that exposes the
  Planner, driver, or model-adapter surface.

* :data:`OUTCOME_IN_PROGRESS`, :data:`OUTCOME_SUCCESS`,
  :data:`OUTCOME_WRONG_BLOCK`, :data:`OUTCOME_TRUTH_MISSING`,
  :data:`OUTCOME_STEP_BUDGET_EXCEEDED`,
  :data:`OUTCOME_TIME_BUDGET_EXCEEDED`,
  :data:`OUTCOME_INVALID_INITIAL_STATE`,
  :data:`OUTCOME_CAUSALITY_MISSING`,
  :data:`OUTCOME_ABNORMAL_TERMINATION` form the closed set of
  ``outcome`` ids the evaluator may emit. The
  :data:`OUTCOMES` frozenset is the canonical list.

* The causality window is finite, type-explicit, and bounded by
  :data:`DEFAULT_CAUSALITY_WINDOW_STEPS`. The evaluator never
  accepts a state with an unbounded causality window.

The two failure classifications (`outcome` and `failure_type`) are
deliberately aligned: ``outcome == failure_type`` for every outcome
in :data:`OUTCOMES` except :data:`OUTCOME_SUCCESS` and
:data:`OUTCOME_IN_PROGRESS`, which use ``None`` for
``failure_type``. This keeps the existing
:class:`obsidianlink.evaluation.portal.EvaluationResult` convention
while letting callers switch on a single stable id.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping


# Closed set of stable outcome ids. The order here is also the
# *priority* order used by the evaluator: the first applicable rule
# wins. Do not reorder existing entries; new ones must be appended.
OUTCOME_IN_PROGRESS = "in_progress"
OUTCOME_SUCCESS = "success"
OUTCOME_WRONG_BLOCK = "wrong_block"
OUTCOME_TRUTH_MISSING = "truth_missing"
OUTCOME_STEP_BUDGET_EXCEEDED = "step_budget_exceeded"
OUTCOME_TIME_BUDGET_EXCEEDED = "time_budget_exceeded"
OUTCOME_INVALID_INITIAL_STATE = "invalid_initial_state"
OUTCOME_CAUSALITY_MISSING = "causality_missing"
OUTCOME_ABNORMAL_TERMINATION = "abnormal_termination"

OUTCOMES: frozenset[str] = frozenset(
    {
        OUTCOME_IN_PROGRESS,
        OUTCOME_SUCCESS,
        OUTCOME_WRONG_BLOCK,
        OUTCOME_TRUTH_MISSING,
        OUTCOME_STEP_BUDGET_EXCEEDED,
        OUTCOME_TIME_BUDGET_EXCEEDED,
        OUTCOME_INVALID_INITIAL_STATE,
        OUTCOME_CAUSALITY_MISSING,
        OUTCOME_ABNORMAL_TERMINATION,
    }
)


# Outcomes that count as terminal failures (i.e. the episode ended
# without success). Used by ``CastingEvaluationResult.failure_type``.
_TERMINAL_FAILURE_OUTCOMES: frozenset[str] = frozenset(
    {
        OUTCOME_WRONG_BLOCK,
        OUTCOME_STEP_BUDGET_EXCEEDED,
        OUTCOME_TIME_BUDGET_EXCEEDED,
        OUTCOME_INVALID_INITIAL_STATE,
        OUTCOME_CAUSALITY_MISSING,
        OUTCOME_ABNORMAL_TERMINATION,
    }
)


# Outcomes that are not terminal failures: ``in_progress`` is not a
# failure and ``truth_missing`` is not a failure — it is an explicit
# fail-closed verdict for missing evaluator truth.
_NON_FAILURE_OUTCOMES: frozenset[str] = frozenset(
    {OUTCOME_IN_PROGRESS, OUTCOME_SUCCESS, OUTCOME_TRUTH_MISSING}
)


# Block ids the casting evaluator accepts as target-cell truth.
# The list is a closed whitelist; an unknown block id is rejected
# in ``CastingEvaluationState.__post_init__``. ``obsidian`` is the
# success target; everything else is the ``wrong_block`` class.
TARGET_BLOCK_IDS: frozenset[str] = frozenset(
    {"air", "obsidian", "cobblestone", "stone", "water", "lava", "missing"}
)


# Termination reasons that count as a "normal" episode end (i.e.
# the driver reached a clean end and the result is allowed to be
# success / wrong_block / causality_missing). Anything else is
# treated as :data:`OUTCOME_ABNORMAL_TERMINATION`.
NORMAL_TERMINATION_REASONS: frozenset[str] = frozenset(
    {
        "driver_done",
        "episode_complete",
        "task_complete",
        "max_steps",
        "max_time",
        "budget_exhausted",
        "goal_reached",
    }
)


# Default finite window (in environment steps) between the last
# relevant Agent action and the observed target-cell block update.
# The window is a hard upper bound, never a soft heuristic. A
# state may set a smaller value; it may not set a larger one
# (the validator caps it at this constant for safety, but tests
# may construct a smaller window freely).
DEFAULT_CAUSALITY_WINDOW_STEPS: int = 4
MAX_CAUSALITY_WINDOW_STEPS: int = 32


def _require_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_non_negative_int(value: int, field_name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_positive_int(value: int, field_name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_finite_number(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def _require_non_negative_number(value: float, field_name: str) -> float:
    value = _require_finite_number(value, field_name)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _require_positive_number(value: float, field_name: str) -> float:
    value = _require_finite_number(value, field_name)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _freeze_json_value(value: Any, field_name: str) -> Any:
    """Validate and recursively freeze a JSON-compatible value tree."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field_name} keys must be strings")
            frozen[key] = _freeze_json_value(item, f"{field_name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json_value(item, f"{field_name}[]") for item in value
        )
    raise ValueError(
        f"{field_name} must contain only JSON-compatible values, "
        f"got {type(value).__name__}"
    )


def _thaw_json_value(value: Any) -> Any:
    """Return a detached tree accepted by the standard JSON encoder."""
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _require_target_cell(value: Any) -> tuple[int, int, int]:
    """Validate a target cell as a 3-tuple of strict ints (no bools)."""
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(
            type(coordinate) is not int or isinstance(coordinate, bool)
            for coordinate in value
        )
    ):
        raise ValueError(
            "target_cell must be a (x, y, z) tuple of strict integers"
        )
    return value


def _require_block_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if value not in TARGET_BLOCK_IDS:
        raise ValueError(
            f"{field_name} must be one of {sorted(TARGET_BLOCK_IDS)}, got {value!r}"
        )
    return value


def _require_tri_bool(value: Any, field_name: str) -> bool | None:
    """Tri-state boolean validator.

    ``True`` / ``False`` mean the bridge positively observed the
    condition; ``None`` means the truth is unavailable.
    """
    if value is None or isinstance(value, bool):
        return value
    raise ValueError(
        f"{field_name} must be a tri-state boolean (True / False / None)"
    )


@dataclass(frozen=True)
class CastingFluidTruth:
    """Typed, optional water / lava evidence.

    Each of ``present`` and ``evidence_step`` is independently
    optional. ``present is None`` means "the bridge did not supply a
    water / lava verdict"; ``present is False`` means the bridge
    positively observed no water / lava; ``present is True`` means
    water / lava was positively observed at ``evidence_step`` (or at
    an earlier step, recorded in :attr:`CastingEvaluationState`).
    """

    present: bool | None = None
    evidence_step: int | None = None

    def __post_init__(self) -> None:
        _require_tri_bool(self.present, "present")
        if self.evidence_step is not None:
            _require_non_negative_int(self.evidence_step, "evidence_step")


@dataclass(frozen=True)
class CastingTransitionEvidence:
    """Typed target-cell block update evidence.

    ``before_block`` and ``after_block`` use the same closed block
    whitelist as :data:`TARGET_BLOCK_IDS`. ``update_step`` is the
    environment step at which the backend first observed the
    change. ``None`` for any of these fields means "the bridge did
    not supply this evidence"; it is *not* a "False" verdict.
    """

    before_block: str | None = None
    after_block: str | None = None
    update_step: int | None = None

    def __post_init__(self) -> None:
        if self.before_block is not None:
            _require_block_id(self.before_block, "before_block")
        if self.after_block is not None:
            _require_block_id(self.after_block, "after_block")
        if self.update_step is not None:
            _require_non_negative_int(self.update_step, "update_step")


@dataclass(frozen=True)
class CastingEvaluationState:
    """Evaluator-only truth for the ``casting_c1_fixed`` task.

    The state must never enter an Agent-visible :class:`Observation`.
    Construction validates every field; an invalid state raises
    before the evaluator is invoked. Container fields are converted
    to immutable tuples / MappingProxyType so external callers
    cannot mutate the state after construction.

    Field semantics
    ---------------

    ``episode_id`` / ``step_id``
        Identity fields required by the project's
        ``observation/action/evaluation/log`` contract. ``step_id``
        is the *current* environment step at which this state was
        captured; ``terminated_step`` (if any) is the step at which
        the episode was marked terminated.

    ``agent_id``
        Optional. The single Agent this state is about. ``None`` is
        allowed because the contract supports multi-agent tasks in
        the future; today the casting task has a single agent.

    ``target_cell``
        ``(x, y, z)`` tuple of strict ``int`` (no bools). Evaluator-
        only: must never enter an Agent observation.

    ``initial_target_block`` / ``current_target_block``
        Stable block ids from :data:`TARGET_BLOCK_IDS`. ``None``
        means the bridge did not supply this truth. ``"obsidian"``
        in ``initial_target_block`` is an explicit fail-closed
        verdict (:data:`OUTCOME_INVALID_INITIAL_STATE`).

    ``target_update_evidence``
        Optional :class:`CastingTransitionEvidence` describing the
        first observed target-cell block update. ``None`` for the
        whole object means "no update ever observed"; the absence of
        a single field inside the object is a partial-truth marker.

    ``water_truth`` / ``lava_truth``
        Optional :class:`CastingFluidTruth` for each fluid. ``None``
        means the bridge did not supply this verdict; ``True`` /
        ``False`` is a positive / negative observation.

    ``relevant_action_steps``
        Tuple of environment step ids at which the Agent performed a
        legal water / lava / support-block action that *could* have
        caused the target-cell block update. Empty tuple means "no
        relevant action observed". Step ids are non-negative ints.

    ``causality_window_steps``
        Finite upper bound (in environment steps) between the last
        relevant Agent action and the observed target-cell block
        update. Must be a positive int no greater than
        :data:`MAX_CAUSALITY_WINDOW_STEPS`. A block update is
        considered causally linked if it happens within this window
        after the latest relevant action.

    ``episode_terminated`` / ``terminated_step`` / ``terminated_reason``
        Termination signal. While ``episode_terminated`` is ``False``
        the result is :data:`OUTCOME_IN_PROGRESS` (regardless of how
        the target cell currently looks). ``terminated_step`` is
        required when ``episode_terminated`` is ``True``.

    ``current_time_seconds``
        Wall-clock time at which this state was captured. Used for
        the time budget check; must be finite.

    ``max_environment_steps`` / ``max_game_time_seconds``
        Hard upper bounds. Both must be positive; the evaluator
        uses them directly to decide :data:`OUTCOME_STEP_BUDGET_EXCEEDED`
        and :data:`OUTCOME_TIME_BUDGET_EXCEEDED`.

    ``evidence``
        Optional auxiliary structured evidence. Kept opaque by the
        evaluator; only the keys written by the evaluator itself
        appear in the result. The state-side evidence must itself
        be JSON-serializable.
    """

    episode_id: str
    step_id: int

    agent_id: str | None = None

    target_cell: tuple[int, int, int] = (0, 0, 0)
    initial_target_block: str | None = None
    current_target_block: str | None = None

    target_update_evidence: CastingTransitionEvidence | None = None
    water_truth: CastingFluidTruth | None = None
    lava_truth: CastingFluidTruth | None = None

    relevant_action_steps: tuple[int, ...] = ()
    causality_window_steps: int = DEFAULT_CAUSALITY_WINDOW_STEPS

    episode_terminated: bool = False
    terminated_step: int | None = None
    terminated_reason: str | None = None

    current_time_seconds: float = 0.0
    max_environment_steps: int = 1
    max_game_time_seconds: float = 1.0

    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.episode_id, "episode_id")
        _require_non_negative_int(self.step_id, "step_id")
        if self.agent_id is not None:
            _require_identifier(self.agent_id, "agent_id")
        object.__setattr__(self, "target_cell", _require_target_cell(self.target_cell))
        if self.initial_target_block is not None:
            _require_block_id(
                self.initial_target_block, "initial_target_block"
            )
        if self.current_target_block is not None:
            _require_block_id(
                self.current_target_block, "current_target_block"
            )
        if self.target_update_evidence is not None and not isinstance(
            self.target_update_evidence, CastingTransitionEvidence
        ):
            raise ValueError(
                "target_update_evidence must be a CastingTransitionEvidence or None"
            )
        if self.water_truth is not None and not isinstance(
            self.water_truth, CastingFluidTruth
        ):
            raise ValueError(
                "water_truth must be a CastingFluidTruth or None"
            )
        if self.lava_truth is not None and not isinstance(
            self.lava_truth, CastingFluidTruth
        ):
            raise ValueError(
                "lava_truth must be a CastingFluidTruth or None"
            )
        try:
            relevant_action_steps = tuple(self.relevant_action_steps)
        except TypeError as exc:
            raise ValueError("relevant_action_steps must be iterable") from exc
        for step in relevant_action_steps:
            _require_non_negative_int(step, "relevant_action_steps")
            if step > self.step_id:
                raise ValueError(
                    "relevant_action_steps cannot contain a future step"
                )
        for name, truth in (
            ("water_truth", self.water_truth),
            ("lava_truth", self.lava_truth),
        ):
            if truth is not None and truth.evidence_step is not None:
                if truth.evidence_step > self.step_id:
                    raise ValueError(f"{name}.evidence_step cannot be in the future")
        if (
            self.target_update_evidence is not None
            and self.target_update_evidence.update_step is not None
            and self.target_update_evidence.update_step > self.step_id
        ):
            raise ValueError("target_update_evidence.update_step cannot be in the future")
        if (
            type(self.causality_window_steps) is not int
            or isinstance(self.causality_window_steps, bool)
            or self.causality_window_steps < 1
            or self.causality_window_steps > MAX_CAUSALITY_WINDOW_STEPS
        ):
            raise ValueError(
                "causality_window_steps must be a positive int "
                f"<= {MAX_CAUSALITY_WINDOW_STEPS}"
            )
        if type(self.episode_terminated) is not bool:
            raise ValueError("episode_terminated must be a boolean")
        if self.episode_terminated:
            if self.terminated_step is None:
                raise ValueError(
                    "episode_terminated=True requires terminated_step to be set"
                )
            _require_non_negative_int(self.terminated_step, "terminated_step")
            if self.terminated_step > self.step_id:
                raise ValueError("terminated_step cannot be in the future")
            if self.terminated_reason is not None:
                _require_identifier(
                    self.terminated_reason, "terminated_reason"
                )
        elif (
            self.terminated_step is not None or self.terminated_reason is not None
        ):
            raise ValueError(
                "terminated_step / terminated_reason require "
                "episode_terminated=True"
            )
        _require_non_negative_number(
            self.current_time_seconds, "current_time_seconds"
        )
        _require_positive_int(self.max_environment_steps, "max_environment_steps")
        _require_positive_number(
            self.max_game_time_seconds, "max_game_time_seconds"
        )
        if not isinstance(self.evidence, Mapping):
            raise ValueError("evidence must be a mapping")
        # Freeze the complete evidence tree, not only its outer mapping.
        object.__setattr__(self, "evidence", _freeze_json_value(self.evidence, "evidence"))
        # Freeze relevant_action_steps in place as a strict tuple of
        # non-negative ints.
        object.__setattr__(
            self,
            "relevant_action_steps",
            tuple(
                _require_non_negative_int(step, "relevant_action_steps")
                for step in relevant_action_steps
            ),
        )


@dataclass(frozen=True)
class CastingEvaluationResult:
    """Typed, frozen, JSON-serializable result of
    :meth:`CastingEvaluator.evaluate`.

    ``outcome`` is one of :data:`OUTCOMES`. ``success`` is derived:
    ``success == (outcome == OUTCOME_SUCCESS)``.

    ``blocking_conditions`` is a tuple of stable string ids; every
    condition that *currently* blocks success is listed. The tuple
    is sorted in priority order so it is stable across runs.

    ``evidence`` is a recursively immutable mapping populated with
    the inputs that drove the decision. :meth:`as_dict` returns a
    detached JSON-serializable snapshot. The caller's state-side
    evidence is *not* re-exported.

    ``failure_type`` mirrors ``outcome`` for terminal failures and is
    ``None`` for :data:`OUTCOME_IN_PROGRESS`, :data:`OUTCOME_SUCCESS`,
    and :data:`OUTCOME_TRUTH_MISSING`. This mirrors the
    :class:`obsidianlink.evaluation.portal.EvaluationResult` shape.
    """

    episode_id: str
    step_id: int
    success: bool
    outcome: str
    blocking_conditions: tuple[str, ...]
    evidence: Mapping[str, Any]
    failure_type: str | None = None
    failure_step: int | None = None
    last_successful_milestone: str | None = None
    episode_terminated: bool = False
    terminated_step: int | None = None
    terminated_reason: str | None = None

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome: {self.outcome!r}")
        if (
            self.failure_type is not None
            and self.failure_type not in _TERMINAL_FAILURE_OUTCOMES
        ):
            raise ValueError(f"unknown failure_type: {self.failure_type!r}")
        object.__setattr__(
            self, "evidence", _freeze_json_value(self.evidence, "evidence")
        )
        for condition in self.blocking_conditions:
            if not isinstance(condition, str) or not condition.strip():
                raise ValueError("blocking_conditions must be stable strings")

    def as_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-serializable result snapshot."""
        return {
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "success": self.success,
            "outcome": self.outcome,
            "blocking_conditions": list(self.blocking_conditions),
            "evidence": _thaw_json_value(self.evidence),
            "failure_type": self.failure_type,
            "failure_step": self.failure_step,
            "last_successful_milestone": self.last_successful_milestone,
            "episode_terminated": self.episode_terminated,
            "terminated_step": self.terminated_step,
            "terminated_reason": self.terminated_reason,
        }


def _last_relevant_action_before(
    steps: Iterable[int], before: int
) -> int | None:
    """Return the largest step in ``steps`` strictly ``< before``."""
    best: int | None = None
    for step in steps:
        if step < before:
            if best is None or step > best:
                best = step
    return best


class CastingEvaluator:
    """Deterministic, offline evaluator for ``casting_c1_fixed``.

    The evaluator is a *pure* object: ``evaluate()`` has no side
    effects, reads no global state, and never inspects Agent
    prompts / images / memory. Its single input is a
    :class:`CastingEvaluationState`; its single output is a
    :class:`CastingEvaluationResult`.

    Failure classification priority (most specific first)
    ---------------------------------------------------

    1. :data:`OUTCOME_STEP_BUDGET_EXCEEDED` — the episode's step
       budget was exceeded *before* the casting condition could be
       established. Always outranks success / wrong_block / etc.
    2. :data:`OUTCOME_TIME_BUDGET_EXCEEDED` — the episode's time
       budget was exceeded. Always outranks success / wrong_block.
    3. :data:`OUTCOME_INVALID_INITIAL_STATE` — the target cell was
       already obsidian at reset. The casting task requires a
       non-obsidian initial state; this is a fail-closed verdict,
       not a success.
    4. :data:`OUTCOME_TRUTH_MISSING` — at least one required
       evaluator truth (initial block, current block, water, lava,
       target update, relevant action) is unavailable. The
       evaluator must not return ``success`` in that case.
    5. :data:`OUTCOME_IN_PROGRESS` — the episode has not been
       terminated yet. The evaluator must not return
       :data:`OUTCOME_SUCCESS` until the termination signal is set.
    6. :data:`OUTCOME_ABNORMAL_TERMINATION` — the episode was
       terminated for a reason that is not in
       :data:`NORMAL_TERMINATION_REASONS`. The evaluator treats
       that as a terminal failure.
    7. :data:`OUTCOME_CAUSALITY_MISSING` — the target cell is
       obsidian at termination, but the block update is *not*
       within the finite causality window after a relevant Agent
       action (or no relevant Agent action is recorded, or the
       update step precedes all relevant actions).
    8. :data:`OUTCOME_WRONG_BLOCK` — the episode terminated
       normally, all truth is present, but the target cell is not
       obsidian.
    9. :data:`OUTCOME_SUCCESS` — every condition above is met.

    The priority is encoded in :func:`_classify_outcome` and is
    locked by ``test_priority_is_stable_for_same_input`` in the
    test suite. The same input state always produces the same
    outcome regardless of evaluation order.
    """

    def evaluate(
        self, state: CastingEvaluationState
    ) -> CastingEvaluationResult:
        outcome, failure_step, last_milestone, evidence = _classify_outcome(
            state
        )
        success = outcome == OUTCOME_SUCCESS
        if outcome in _TERMINAL_FAILURE_OUTCOMES:
            failure_type = outcome
        else:
            failure_type = None
        return CastingEvaluationResult(
            episode_id=state.episode_id,
            step_id=state.step_id,
            success=success,
            outcome=outcome,
            blocking_conditions=_blocking_conditions(outcome, state, evidence),
            evidence=evidence,
            failure_type=failure_type,
            failure_step=failure_step,
            last_successful_milestone=last_milestone,
            episode_terminated=state.episode_terminated,
            terminated_step=state.terminated_step,
            terminated_reason=state.terminated_reason,
        )


def _classify_outcome(
    state: CastingEvaluationState,
) -> tuple[str, int | None, str | None, Mapping[str, Any]]:
    """Apply the priority rules and return
    ``(outcome, failure_step, last_successful_milestone, evidence)``.
    """
    evidence: dict[str, Any] = {
        "target_cell": list(state.target_cell),
        "initial_target_block": state.initial_target_block,
        "current_target_block": state.current_target_block,
        "episode_id": state.episode_id,
        "step_id": state.step_id,
        "agent_id": state.agent_id,
        "max_environment_steps": state.max_environment_steps,
        "max_game_time_seconds": state.max_game_time_seconds,
        "current_time_seconds": state.current_time_seconds,
        "relevant_action_steps": list(state.relevant_action_steps),
        "causality_window_steps": state.causality_window_steps,
    }
    if state.target_update_evidence is not None:
        evidence["target_update_evidence"] = {
            "before_block": state.target_update_evidence.before_block,
            "after_block": state.target_update_evidence.after_block,
            "update_step": state.target_update_evidence.update_step,
        }
    if state.water_truth is not None:
        evidence["water_truth"] = {
            "present": state.water_truth.present,
            "evidence_step": state.water_truth.evidence_step,
        }
    if state.lava_truth is not None:
        evidence["lava_truth"] = {
            "present": state.lava_truth.present,
            "evidence_step": state.lava_truth.evidence_step,
        }

    # ------------------------------------------------------------------
    # 1 / 2. Budget violations always outrank everything else.
    # ------------------------------------------------------------------
    step_budget_step: int | None = None
    observed_steps = [state.step_id]
    if state.terminated_step is not None:
        observed_steps.append(state.terminated_step)
    latest_observed_step = max(observed_steps)
    if latest_observed_step > state.max_environment_steps:
        step_budget_step = latest_observed_step
    if step_budget_step is not None:
        evidence["budget_exceeded_kind"] = "step"
        evidence["budget_exceeded_value"] = step_budget_step
        evidence["budget_limit"] = state.max_environment_steps
        return (
            OUTCOME_STEP_BUDGET_EXCEEDED,
            step_budget_step,
            None,
            MappingProxyType(evidence),
        )

    if state.current_time_seconds > state.max_game_time_seconds:
        evidence["budget_exceeded_kind"] = "time"
        evidence["budget_exceeded_value"] = state.current_time_seconds
        evidence["budget_limit"] = state.max_game_time_seconds
        return (
            OUTCOME_TIME_BUDGET_EXCEEDED,
            state.terminated_step if state.episode_terminated else state.step_id,
            None,
            MappingProxyType(evidence),
        )

    # ------------------------------------------------------------------
    # 3. Invalid initial state — obsidian at reset fails closed.
    # ------------------------------------------------------------------
    if state.initial_target_block == "obsidian":
        return (
            OUTCOME_INVALID_INITIAL_STATE,
            state.step_id,
            None,
            MappingProxyType(evidence),
        )

    # ------------------------------------------------------------------
    # 4. Truth missing.
    # ------------------------------------------------------------------
    missing = _missing_truth(state)
    if missing:
        evidence["missing_truth"] = list(missing)
        return (
            OUTCOME_TRUTH_MISSING,
            None,
            None,
            MappingProxyType(evidence),
        )

    # ------------------------------------------------------------------
    # 5. Episode not terminated yet.
    # ------------------------------------------------------------------
    if not state.episode_terminated:
        return (
            OUTCOME_IN_PROGRESS,
            None,
            None,
            MappingProxyType(evidence),
        )

    # ------------------------------------------------------------------
    # 6. Abnormal termination.
    # ------------------------------------------------------------------
    if (
        state.terminated_reason is not None
        and state.terminated_reason not in NORMAL_TERMINATION_REASONS
    ):
        return (
            OUTCOME_ABNORMAL_TERMINATION,
            state.terminated_step,
            None,
            MappingProxyType(evidence),
        )

    # ------------------------------------------------------------------
    # 7 / 8 / 9. Target cell block, causality, and success.
    # ------------------------------------------------------------------
    if state.current_target_block != "obsidian":
        evidence["expected_block"] = "obsidian"
        evidence["actual_block"] = state.current_target_block
        return (
            OUTCOME_WRONG_BLOCK,
            state.terminated_step,
            None,
            MappingProxyType(evidence),
        )

    # current_target_block == "obsidian" — we need causality.
    update = state.target_update_evidence
    if (
        update is None
        or update.update_step is None
        or update.after_block is None
    ):
        # We already filtered truth_missing above, so this branch
        # should not run. Kept as a defensive guard; the priority
        # order still says causality_missing > wrong_block.
        return (
            OUTCOME_CAUSALITY_MISSING,
            state.terminated_step,
            None,
            MappingProxyType(evidence),
        )
    if state.water_truth is None or state.water_truth.present is not True:
        evidence["causality_reason"] = "water_not_present"
        return (
            OUTCOME_CAUSALITY_MISSING,
            state.terminated_step,
            None,
            MappingProxyType(evidence),
        )
    if state.lava_truth is None or state.lava_truth.present is not True:
        evidence["causality_reason"] = "lava_not_present"
        return (
            OUTCOME_CAUSALITY_MISSING,
            state.terminated_step,
            None,
            MappingProxyType(evidence),
        )
    if update.after_block != "obsidian":
        evidence["causality_reason"] = "transition_did_not_produce_obsidian"
        return (
            OUTCOME_CAUSALITY_MISSING,
            state.terminated_step,
            None,
            MappingProxyType(evidence),
        )
    if update.after_block != state.current_target_block:
        evidence["causality_reason"] = "transition_current_block_mismatch"
        return (
            OUTCOME_CAUSALITY_MISSING,
            state.terminated_step,
            None,
            MappingProxyType(evidence),
        )
    update_step = update.update_step
    relevant = tuple(state.relevant_action_steps)
    if not relevant:
        evidence["causality_reason"] = "no_relevant_actions"
        return (
            OUTCOME_CAUSALITY_MISSING,
            state.terminated_step,
            None,
            MappingProxyType(evidence),
        )
    last_action = _last_relevant_action_before(relevant, update_step + 1)
    if last_action is None:
        evidence["causality_reason"] = "update_before_any_action"
        evidence["update_step"] = update_step
        return (
            OUTCOME_CAUSALITY_MISSING,
            state.terminated_step,
            None,
            MappingProxyType(evidence),
        )
    delta = update_step - last_action
    evidence["causality_delta_steps"] = delta
    evidence["causality_action_step"] = last_action
    if delta < 0 or delta > state.causality_window_steps:
        evidence["causality_reason"] = "outside_window"
        return (
            OUTCOME_CAUSALITY_MISSING,
            state.terminated_step,
            None,
            MappingProxyType(evidence),
        )

    return (
        OUTCOME_SUCCESS,
        None,
        None,
        MappingProxyType(evidence),
    )


def _missing_truth(
    state: CastingEvaluationState,
) -> tuple[str, ...]:
    """Return the list of evaluator truth fields that are missing.

    The list is the *closed* set of truths the casting evaluator
    needs to make a success / wrong_block decision; a missing
    ``target_cell`` is *not* part of the list because it is
    validated at construction time.
    """
    missing: list[str] = []
    if state.initial_target_block is None:
        missing.append("initial_target_block")
    if state.current_target_block is None:
        missing.append("current_target_block")
    if state.water_truth is None or state.water_truth.present is None:
        missing.append("water_truth")
    if state.lava_truth is None or state.lava_truth.present is None:
        missing.append("lava_truth")
    update = state.target_update_evidence
    if (
        update is None
        or update.update_step is None
        or update.before_block is None
        or update.after_block is None
    ):
        missing.append("target_update_evidence")
    if not state.relevant_action_steps:
        missing.append("relevant_action_steps")
    if state.episode_terminated and state.terminated_reason is None:
        missing.append("terminated_reason")
    return tuple(missing)


def _blocking_conditions(
    outcome: str,
    state: CastingEvaluationState,
    evidence: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return a stable tuple of blocking conditions for the result.

    The conditions mirror the priority rules above; the tuple is
    the *single source of truth* for "what is currently preventing
    success". ``success == True`` always implies an empty tuple.
    """
    if outcome == OUTCOME_SUCCESS:
        return ()
    if outcome == OUTCOME_IN_PROGRESS:
        return ("episode_not_terminated",)
    if outcome == OUTCOME_TRUTH_MISSING:
        return tuple(
            f"missing_truth:{name}" for name in evidence.get("missing_truth", ())
        )
    if outcome == OUTCOME_STEP_BUDGET_EXCEEDED:
        return ("step_budget_exceeded",)
    if outcome == OUTCOME_TIME_BUDGET_EXCEEDED:
        return ("time_budget_exceeded",)
    if outcome == OUTCOME_INVALID_INITIAL_STATE:
        return ("target_already_obsidian_at_reset",)
    if outcome == OUTCOME_CAUSALITY_MISSING:
        reason = evidence.get("causality_reason")
        if isinstance(reason, str) and reason:
            return (f"causality_missing:{reason}",)
        return ("causality_missing",)
    if outcome == OUTCOME_ABNORMAL_TERMINATION:
        return ("abnormal_termination",)
    if outcome == OUTCOME_WRONG_BLOCK:
        return (
            f"wrong_block:expected_obsidian_got_"
            f"{state.current_target_block or 'unknown'}"
        )
    return ()


__all__ = [
    "CastingEvaluationState",
    "CastingEvaluationResult",
    "CastingEvaluator",
    "CastingFluidTruth",
    "CastingTransitionEvidence",
    "DEFAULT_CAUSALITY_WINDOW_STEPS",
    "MAX_CAUSALITY_WINDOW_STEPS",
    "NORMAL_TERMINATION_REASONS",
    "OUTCOME_ABNORMAL_TERMINATION",
    "OUTCOME_CAUSALITY_MISSING",
    "OUTCOME_IN_PROGRESS",
    "OUTCOME_INVALID_INITIAL_STATE",
    "OUTCOME_STEP_BUDGET_EXCEEDED",
    "OUTCOME_SUCCESS",
    "OUTCOME_TIME_BUDGET_EXCEEDED",
    "OUTCOME_TRUTH_MISSING",
    "OUTCOME_WRONG_BLOCK",
    "OUTCOMES",
    "TARGET_BLOCK_IDS",
]

"""Offline continuous casting evaluator for ``casting_c3_fixed``.

R5 extends the R3 single-cell :class:`CastingEvaluator` to a fixed,
short multi-cell straight line segment (the minimum viable "almost a
portal side" slice). The new evaluator lives next to the R3 surface
instead of replacing it; R3 and R5 both keep their own state and
result types so the two benchmarks can be replayed independently.

Design contract
---------------

The :class:`ContinuousCastingEvaluator` is a *pure* deterministic
function over :class:`ContinuousCastingEvaluationState`. It:

* never reads Agent text, prompts, or images;
* never imports the planner, the driver, or the model-adapter surface;
* never calls into a backend or reads wall-clock time;
* gives the same output for the same state on repeated calls;
* never reads :class:`Observation` or public action values;
* treats truth for *one* cell as evidence only for that cell. The
  evaluator never lifts another cell's relevant action, fluid
  evidence, or transition into a different cell's causality check.

Stability contract
------------------

* The :class:`ContinuousCastingCellTruth`,
  :class:`ContinuousCastingEvaluationState`, and
  :class:`ContinuousCastingEvaluationResult` dataclasses are frozen.
  All fields are validated in ``__post_init__``; an invalid state
  raises :class:`ValueError` or :class:`TypeError` *before* the
  evaluator ever sees it.
* :data:`CONTINUOUS_OUTCOMES` is the closed set of ``outcome`` ids.
* The priority is encoded in :func:`_classify_outcome` and is locked
  by ``test_priority_is_stable_for_same_input`` in the test suite.
  The same input state always produces the same outcome regardless
  of evaluation order.
* :meth:`ContinuousCastingEvaluationResult.as_dict` returns a
  detached, JSON-serializable snapshot. The state-side evidence is
  *not* re-exported.

Outcome set
-----------

The closed set of outcome ids is:

* :data:`OUTCOME_SUCCESS` — every cell became obsidian with full
  water / lava / transition / relevant-action evidence, the episode
  terminated normally, and no budget was exceeded.
* :data:`OUTCOME_IN_PROGRESS` — the episode has not been terminated
  yet.
* :data:`OUTCOME_PARTIAL_COMPLEMENT` — the episode terminated
  normally after completing a non-empty strict prefix of the three
  ordered cells; the remaining cells have complete truth but are not
  obsidian. Missing truth, broken causality, or an out-of-order hole
  is reported as a more specific failure.
* :data:`OUTCOME_WRONG_BLOCK` — the episode terminated normally,
  every cell's truth is present, but at least one cell ended on a
  block that is not obsidian.
* :data:`OUTCOME_TRUTH_MISSING` — at least one cell is missing a
  required evaluator truth (initial block, current block, water,
  lava, transition, or relevant action) *or* the cells tuple has the
  wrong length.
* :data:`OUTCOME_STEP_BUDGET_EXCEEDED` — ``max(step_id,
  terminated_step) > max_environment_steps``.
* :data:`OUTCOME_TIME_BUDGET_EXCEEDED` — ``current_time_seconds >
  max_game_time_seconds``.
* :data:`OUTCOME_INVALID_INITIAL_STATE` — at least one cell already
  was obsidian at reset.
* :data:`OUTCOME_CAUSALITY_MISSING` — at least one cell became
  obsidian but its block update is not within the finite causality
  window after its own relevant action, or its water / lava truth
  is not ``True``, or its transition is not obsidian.
* :data:`OUTCOME_ABNORMAL_TERMINATION` — the episode was terminated
  for a reason that is not in
  :data:`NORMAL_TERMINATION_REASONS` (the closed list is shared with
  the R3 :class:`CastingEvaluator`).

The failure classification is aligned with R3:
``failure_type == outcome`` for every terminal failure; ``success``
and ``in_progress`` and ``truth_missing`` map to ``failure_type =
None``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from obsidianlink.evaluation.casting import (
    CastingFluidTruth,
    CastingTransitionEvidence,
    DEFAULT_CAUSALITY_WINDOW_STEPS,
    MAX_CAUSALITY_WINDOW_STEPS,
    NORMAL_TERMINATION_REASONS,
    OUTCOME_ABNORMAL_TERMINATION as _R3_OUTCOME_ABNORMAL_TERMINATION,
    OUTCOME_CAUSALITY_MISSING as _R3_OUTCOME_CAUSALITY_MISSING,
    OUTCOME_INVALID_INITIAL_STATE as _R3_OUTCOME_INVALID_INITIAL_STATE,
    OUTCOME_STEP_BUDGET_EXCEEDED as _R3_OUTCOME_STEP_BUDGET_EXCEEDED,
    OUTCOME_TIME_BUDGET_EXCEEDED as _R3_OUTCOME_TIME_BUDGET_EXCEEDED,
    OUTCOME_TRUTH_MISSING as _R3_OUTCOME_TRUTH_MISSING,
    OUTCOME_WRONG_BLOCK as _R3_OUTCOME_WRONG_BLOCK,
    OUTCOME_SUCCESS as _R3_OUTCOME_SUCCESS,
    OUTCOME_IN_PROGRESS as _R3_OUTCOME_IN_PROGRESS,
)


# ----------------------------------------------------------------------
# Outcome constants. The R3 outcome ids are *reused* (the
# :class:`ContinuousCastingEvaluator` returns the same id set as the
# R3 evaluator for the overlapping cases). R5 only adds one new
# terminal-failure id: :data:`OUTCOME_PARTIAL_COMPLEMENT`.
# ----------------------------------------------------------------------

OUTCOME_SUCCESS: str = _R3_OUTCOME_SUCCESS
OUTCOME_IN_PROGRESS: str = _R3_OUTCOME_IN_PROGRESS
OUTCOME_WRONG_BLOCK: str = _R3_OUTCOME_WRONG_BLOCK
OUTCOME_TRUTH_MISSING: str = _R3_OUTCOME_TRUTH_MISSING
OUTCOME_STEP_BUDGET_EXCEEDED: str = _R3_OUTCOME_STEP_BUDGET_EXCEEDED
OUTCOME_TIME_BUDGET_EXCEEDED: str = _R3_OUTCOME_TIME_BUDGET_EXCEEDED
OUTCOME_INVALID_INITIAL_STATE: str = _R3_OUTCOME_INVALID_INITIAL_STATE
OUTCOME_CAUSALITY_MISSING: str = _R3_OUTCOME_CAUSALITY_MISSING
OUTCOME_ABNORMAL_TERMINATION: str = _R3_OUTCOME_ABNORMAL_TERMINATION
OUTCOME_PARTIAL_COMPLEMENT: str = "partial_completion"

CONTINUOUS_OUTCOMES: frozenset[str] = frozenset(
    {
        OUTCOME_SUCCESS,
        OUTCOME_IN_PROGRESS,
        OUTCOME_PARTIAL_COMPLEMENT,
        OUTCOME_WRONG_BLOCK,
        OUTCOME_TRUTH_MISSING,
        OUTCOME_STEP_BUDGET_EXCEEDED,
        OUTCOME_TIME_BUDGET_EXCEEDED,
        OUTCOME_INVALID_INITIAL_STATE,
        OUTCOME_CAUSALITY_MISSING,
        OUTCOME_ABNORMAL_TERMINATION,
    }
)

_TERMINAL_FAILURE_OUTCOMES: frozenset[str] = frozenset(
    {
        OUTCOME_PARTIAL_COMPLEMENT,
        OUTCOME_WRONG_BLOCK,
        OUTCOME_STEP_BUDGET_EXCEEDED,
        OUTCOME_TIME_BUDGET_EXCEEDED,
        OUTCOME_INVALID_INITIAL_STATE,
        OUTCOME_CAUSALITY_MISSING,
        OUTCOME_ABNORMAL_TERMINATION,
    }
)

#: Default number of ordered target cells in the minimum R5 task.
DEFAULT_CELL_COUNT: int = 3
#: Hard minimum number of cells an R5 state must carry.
MIN_CELL_COUNT: int = DEFAULT_CELL_COUNT
#: Hard maximum number of cells an R5 state may carry.
MAX_CELL_COUNT: int = DEFAULT_CELL_COUNT
#: Frozen ordered target cells for the ``casting_c3_fixed`` contract.
CASTING_C3_TARGET_CELLS: tuple[tuple[int, int, int], ...] = (
    (2, 4, 3),
    (3, 4, 3),
    (4, 4, 3),
)
#: Per-cell causality window upper bound. The R5 window is the same
#: per-cell window as the R3 single-cell evaluator, so we reuse the
#: R3 constants.
DEFAULT_C3_CELL_BUDGET: int = DEFAULT_CELL_COUNT


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------


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
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _require_target_cell(value: Any) -> tuple[int, int, int]:
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


def _last_relevant_action_before(
    steps: Iterable[int], before: int
) -> int | None:
    best: int | None = None
    for step in steps:
        if step < before:
            if best is None or step > best:
                best = step
    return best


# ----------------------------------------------------------------------
# Data types
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ContinuousCastingCellTruth:
    """Typed, per-cell evaluator truth.

    Each cell's evidence is independent: ``relevant_action_steps`` is
    the *per-cell* list of step ids where the Agent performed a
    relevant fluid or support action for **this cell**, never a list
    aggregated across cells. The evaluator never borrows another
    cell's evidence to satisfy a different cell's causality check.
    """

    target_cell: tuple[int, int, int]
    initial_block: str | None
    current_block: str | None
    water_truth: CastingFluidTruth | None
    lava_truth: CastingFluidTruth | None
    transition_evidence: CastingTransitionEvidence | None
    relevant_action_steps: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "target_cell", _require_target_cell(self.target_cell)
        )
        from obsidianlink.evaluation.casting import (
            _require_block_id,
        )

        if self.initial_block is not None:
            _require_block_id(self.initial_block, "initial_block")
        if self.current_block is not None:
            _require_block_id(self.current_block, "current_block")
        if self.water_truth is not None and not isinstance(
            self.water_truth, CastingFluidTruth
        ):
            raise ValueError("water_truth must be a CastingFluidTruth or None")
        if self.lava_truth is not None and not isinstance(
            self.lava_truth, CastingFluidTruth
        ):
            raise ValueError("lava_truth must be a CastingFluidTruth or None")
        if (
            self.transition_evidence is not None
            and not isinstance(self.transition_evidence, CastingTransitionEvidence)
        ):
            raise ValueError(
                "transition_evidence must be a CastingTransitionEvidence or None"
            )
        try:
            steps = tuple(self.relevant_action_steps)
        except TypeError as exc:
            raise ValueError(
                "relevant_action_steps must be iterable"
            ) from exc
        for step in steps:
            _require_non_negative_int(step, "relevant_action_steps")
        object.__setattr__(self, "relevant_action_steps", steps)


@dataclass(frozen=True)
class ContinuousCastingEvaluationState:
    """Evaluator-only truth for the ``casting_c3_fixed`` task.

    The state is the *only* input the evaluator accepts. All fields
    are validated at construction; an invalid state raises
    :class:`ValueError` before the evaluator is invoked.
    """

    episode_id: str
    step_id: int
    cells: tuple[ContinuousCastingCellTruth, ...]
    agent_id: str | None = None
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
        try:
            cells = tuple(self.cells)
        except TypeError as exc:
            raise ValueError("cells must be iterable") from exc
        if not cells:
            raise ValueError("cells must be a non-empty tuple")
        for index, cell in enumerate(cells):
            if not isinstance(cell, ContinuousCastingCellTruth):
                raise ValueError(
                    f"cells[{index}] must be a ContinuousCastingCellTruth"
                )
        seen: set[tuple[int, int, int]] = set()
        for cell in cells:
            if cell.target_cell in seen:
                raise ValueError(
                    f"cells contain duplicate target_cell {cell.target_cell!r}"
                )
            seen.add(cell.target_cell)
        for cell in cells:
            for step in cell.relevant_action_steps:
                if step > self.step_id:
                    raise ValueError(
                        "cell relevant_action_steps cannot contain a future step"
                    )
            if cell.water_truth is not None and cell.water_truth.evidence_step is not None:
                if cell.water_truth.evidence_step > self.step_id:
                    raise ValueError(
                        "cell water_truth.evidence_step cannot be in the future"
                    )
            if cell.lava_truth is not None and cell.lava_truth.evidence_step is not None:
                if cell.lava_truth.evidence_step > self.step_id:
                    raise ValueError(
                        "cell lava_truth.evidence_step cannot be in the future"
                    )
            if (
                cell.transition_evidence is not None
                and cell.transition_evidence.update_step is not None
                and cell.transition_evidence.update_step > self.step_id
            ):
                raise ValueError(
                    "cell transition.update_step cannot be in the future"
                )
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
                    "episode_terminated=True requires terminated_step"
                )
            _require_non_negative_int(self.terminated_step, "terminated_step")
            if self.terminated_step > self.step_id:
                raise ValueError("terminated_step cannot be in the future")
            if self.terminated_reason is not None:
                _require_identifier(
                    self.terminated_reason, "terminated_reason"
                )
        elif (
            self.terminated_step is not None
            or self.terminated_reason is not None
        ):
            raise ValueError(
                "terminated_step/terminated_reason require episode_terminated=True"
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
        if len(cells) != DEFAULT_CELL_COUNT:
            raise ValueError(
                "casting_c3_fixed requires exactly "
                f"{DEFAULT_CELL_COUNT} cells, got {len(cells)}"
            )
        actual_target_cells = tuple(cell.target_cell for cell in cells)
        if actual_target_cells != CASTING_C3_TARGET_CELLS:
            raise ValueError(
                "cells must match the frozen casting_c3_fixed target order: "
                f"expected {CASTING_C3_TARGET_CELLS!r}, got "
                f"{actual_target_cells!r}"
            )
        claimed_steps: dict[int, int] = {}
        for cell_index, cell in enumerate(cells):
            for step in cell.relevant_action_steps:
                previous_owner = claimed_steps.get(step)
                if previous_owner is not None:
                    raise ValueError(
                        "relevant_action_steps must be disjoint across cells: "
                        f"step {step} is claimed by cells {previous_owner} and "
                        f"{cell_index}"
                    )
                claimed_steps[step] = cell_index
        object.__setattr__(self, "cells", cells)
        object.__setattr__(
            self, "evidence", _freeze_json_value(self.evidence, "evidence")
        )


@dataclass(frozen=True)
class ContinuousCastingEvaluationResult:
    """Typed, frozen, JSON-serializable result.

    ``outcome`` is one of :data:`CONTINUOUS_OUTCOMES`. ``success`` is
    derived (``outcome == OUTCOME_SUCCESS``).

    ``per_cell_outcomes`` is a tuple with one entry per cell, in the
    same order as :attr:`ContinuousCastingEvaluationState.cells`.
    The entry is the per-cell verdict string; it is one of
    :data:`CONTINUOUS_OUTCOMES` (or one of the per-cell sentinel ids
    :data:`PER_CELL_NOT_EVALUATED` for the cases where the overall
    verdict outranks the per-cell decision). ``first_failed_cell`` is
    the index of the lowest cell that did not end on
    :data:`PER_CELL_SUCCESS`, or ``None`` when every cell succeeded.
    """

    episode_id: str
    step_id: int
    success: bool
    outcome: str
    completed_cells: int
    total_cells: int
    per_cell_outcomes: tuple[str, ...]
    first_failed_cell: int | None
    blocking_conditions: tuple[str, ...]
    evidence: Mapping[str, Any]
    failure_type: str | None = None
    failure_step: int | None = None
    episode_terminated: bool = False
    terminated_step: int | None = None
    terminated_reason: str | None = None

    def __post_init__(self) -> None:
        if self.outcome not in CONTINUOUS_OUTCOMES:
            raise ValueError(f"unknown outcome: {self.outcome!r}")
        if (
            self.failure_type is not None
            and self.failure_type not in _TERMINAL_FAILURE_OUTCOMES
        ):
            raise ValueError(f"unknown failure_type: {self.failure_type!r}")
        _require_non_negative_int(self.completed_cells, "completed_cells")
        _require_positive_int(self.total_cells, "total_cells")
        if self.completed_cells > self.total_cells:
            raise ValueError("completed_cells cannot exceed total_cells")
        if not isinstance(self.per_cell_outcomes, tuple):
            raise ValueError("per_cell_outcomes must be a tuple")
        if len(self.per_cell_outcomes) != self.total_cells:
            raise ValueError(
                "per_cell_outcomes length must equal total_cells"
            )
        for index, verdict in enumerate(self.per_cell_outcomes):
            if verdict not in _PER_CELL_VERDICTS:
                raise ValueError(
                    f"per_cell_outcomes[{index}] = {verdict!r} is not a "
                    f"valid per-cell verdict"
                )
        if self.first_failed_cell is not None:
            _require_non_negative_int(
                self.first_failed_cell, "first_failed_cell"
            )
            if self.first_failed_cell >= self.total_cells:
                raise ValueError(
                    "first_failed_cell must be < total_cells when set"
                )
        for condition in self.blocking_conditions:
            if not isinstance(condition, str) or not condition.strip():
                raise ValueError("blocking_conditions must be stable strings")
        object.__setattr__(
            self, "evidence", _freeze_json_value(self.evidence, "evidence")
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "success": self.success,
            "outcome": self.outcome,
            "completed_cells": self.completed_cells,
            "total_cells": self.total_cells,
            "per_cell_outcomes": list(self.per_cell_outcomes),
            "first_failed_cell": self.first_failed_cell,
            "blocking_conditions": list(self.blocking_conditions),
            "evidence": _thaw_json_value(self.evidence),
            "failure_type": self.failure_type,
            "failure_step": self.failure_step,
            "episode_terminated": self.episode_terminated,
            "terminated_step": self.terminated_step,
            "terminated_reason": self.terminated_reason,
        }


# ----------------------------------------------------------------------
# Per-cell verdict sentinels
# ----------------------------------------------------------------------

PER_CELL_SUCCESS: str = "cell_success"
PER_CELL_NOT_EVALUATED: str = "cell_not_evaluated"
PER_CELL_TRUTH_MISSING: str = "cell_truth_missing"
PER_CELL_CAUSALITY_MISSING: str = "cell_causality_missing"
PER_CELL_WRONG_BLOCK: str = "cell_wrong_block"

_PER_CELL_VERDICTS: frozenset[str] = frozenset(
    {
        PER_CELL_SUCCESS,
        PER_CELL_NOT_EVALUATED,
        PER_CELL_TRUTH_MISSING,
        PER_CELL_CAUSALITY_MISSING,
        PER_CELL_WRONG_BLOCK,
    }
)


# ----------------------------------------------------------------------
# Per-cell helpers
# ----------------------------------------------------------------------


def _per_cell_missing(cell: ContinuousCastingCellTruth) -> tuple[str, ...]:
    missing: list[str] = []
    if cell.initial_block is None:
        missing.append("initial_block")
    if cell.current_block is None:
        missing.append("current_block")
    if cell.water_truth is None or cell.water_truth.present is None:
        missing.append("water_truth")
    if cell.lava_truth is None or cell.lava_truth.present is None:
        missing.append("lava_truth")
    update = cell.transition_evidence
    if (
        update is None
        or update.update_step is None
        or update.before_block is None
        or update.after_block is None
    ):
        missing.append("transition_evidence")
    if not cell.relevant_action_steps:
        missing.append("relevant_action_steps")
    return tuple(missing)


def _classify_cell(
    cell: ContinuousCastingCellTruth,
    *,
    causality_window_steps: int,
    episode_terminated: bool,
    terminated_step: int | None,
) -> tuple[str, int | None, dict[str, Any]]:
    """Return ``(per_cell_verdict, failure_step, evidence)``.

    The per-cell verdict is one of the per-cell sentinels in
    :data:`_PER_CELL_VERDICTS`. The overall verdict for the episode
    may still outrank the per-cell verdict (e.g. ``step_budget``
    fires before any per-cell rule).
    """
    evidence: dict[str, Any] = {
        "target_cell": list(cell.target_cell),
        "initial_block": cell.initial_block,
        "current_block": cell.current_block,
        "relevant_action_steps": list(cell.relevant_action_steps),
    }
    if cell.water_truth is not None:
        evidence["water_truth"] = {
            "present": cell.water_truth.present,
            "evidence_step": cell.water_truth.evidence_step,
        }
    if cell.lava_truth is not None:
        evidence["lava_truth"] = {
            "present": cell.lava_truth.present,
            "evidence_step": cell.lava_truth.evidence_step,
        }
    if cell.transition_evidence is not None:
        evidence["transition_evidence"] = {
            "before_block": cell.transition_evidence.before_block,
            "after_block": cell.transition_evidence.after_block,
            "update_step": cell.transition_evidence.update_step,
        }
    missing = _per_cell_missing(cell)
    if missing:
        evidence["missing_truth"] = list(missing)
        return (PER_CELL_TRUTH_MISSING, terminated_step, evidence)
    if not episode_terminated:
        return (PER_CELL_NOT_EVALUATED, None, evidence)
    if cell.current_block != "obsidian":
        evidence["expected_block"] = "obsidian"
        evidence["actual_block"] = cell.current_block
        return (PER_CELL_WRONG_BLOCK, terminated_step, evidence)
    update = cell.transition_evidence
    assert update is not None  # truth_missing would have caught it
    if update.after_block != "obsidian":
        evidence["causality_reason"] = "transition_did_not_produce_obsidian"
        return (PER_CELL_CAUSALITY_MISSING, terminated_step, evidence)
    if cell.water_truth is None or cell.water_truth.present is not True:
        evidence["causality_reason"] = "water_not_present"
        return (PER_CELL_CAUSALITY_MISSING, terminated_step, evidence)
    if cell.lava_truth is None or cell.lava_truth.present is not True:
        evidence["causality_reason"] = "lava_not_present"
        return (PER_CELL_CAUSALITY_MISSING, terminated_step, evidence)
    if update.update_step is None:
        evidence["causality_reason"] = "update_step_missing"
        return (PER_CELL_CAUSALITY_MISSING, terminated_step, evidence)
    last_action = _last_relevant_action_before(
        cell.relevant_action_steps, update.update_step + 1
    )
    if last_action is None:
        evidence["causality_reason"] = "update_before_any_action"
        evidence["update_step"] = update.update_step
        return (PER_CELL_CAUSALITY_MISSING, terminated_step, evidence)
    delta = update.update_step - last_action
    evidence["causality_delta_steps"] = delta
    evidence["causality_action_step"] = last_action
    if delta < 0 or delta > causality_window_steps:
        evidence["causality_reason"] = "outside_window"
        return (PER_CELL_CAUSALITY_MISSING, terminated_step, evidence)
    return (PER_CELL_SUCCESS, None, evidence)


def _aggregate_outcome(
    per_cell: tuple[tuple[str, int | None, dict[str, Any]], ...],
    *,
    episode_terminated: bool,
    terminated_step: int | None,
) -> tuple[str, int | None, dict[str, Any]]:
    """Reduce the per-cell verdicts to a single closed-set outcome.

    The reduction respects the locked priority order: truth_missing
    and causality failures are reported first; a truth-complete
    successful prefix becomes partial_completion; out-of-order or
    zero-progress wrong blocks remain wrong_block.
    """
    truth_missing: list[int] = []
    wrong_block: list[int] = []
    causality_missing: list[int] = []
    success_cells: list[int] = []
    for index, (verdict, _step, _evidence) in enumerate(per_cell):
        if verdict == PER_CELL_TRUTH_MISSING:
            truth_missing.append(index)
        elif verdict == PER_CELL_WRONG_BLOCK:
            wrong_block.append(index)
        elif verdict == PER_CELL_CAUSALITY_MISSING:
            causality_missing.append(index)
        elif verdict == PER_CELL_SUCCESS:
            success_cells.append(index)
    if not episode_terminated:
        # Episode still in progress; we only know truth_missing +
        # in_progress. If any cell has truth_missing it still
        # outranks in_progress, so the per-cell verdict chain still
        # gives the right answer when we re-aggregate later.
        if truth_missing:
            return (
                OUTCOME_TRUTH_MISSING,
                terminated_step,
                {"missing_cells": truth_missing},
            )
        return (OUTCOME_IN_PROGRESS, None, {})
    if truth_missing:
        return (
            OUTCOME_TRUTH_MISSING,
            terminated_step,
            {"missing_cells": truth_missing},
        )
    if causality_missing:
        return (
            OUTCOME_CAUSALITY_MISSING,
            terminated_step,
            {"causality_missing_cells": causality_missing},
        )
    total = len(per_cell)
    completed = len(success_cells)
    if completed == total:
        return (OUTCOME_SUCCESS, None, {"success_cells": success_cells})
    # The task is ordered. A non-empty strict prefix of successful
    # cells followed only by truth-complete wrong blocks is a genuine
    # partial completion. A hole in the prefix (for example cells 0
    # and 2 succeeded but cell 1 did not) is a wrong-block failure,
    # not valid progress through the ordered segment.
    expected_prefix = list(range(completed))
    remaining = list(range(completed, total))
    if (
        0 < completed < total
        and success_cells == expected_prefix
        and wrong_block == remaining
    ):
        return (
            OUTCOME_PARTIAL_COMPLEMENT,
            terminated_step,
            {
                "success_cells": success_cells,
                "non_success_cells": total - completed,
            },
        )
    return (
        OUTCOME_WRONG_BLOCK,
        terminated_step,
        {"wrong_block_cells": wrong_block},
    )


# ----------------------------------------------------------------------
# Evaluator entry point
# ----------------------------------------------------------------------


class ContinuousCastingEvaluator:
    """Deterministic, offline evaluator for ``casting_c3_fixed``.

    The evaluator is a *pure* object: ``evaluate()`` has no side
    effects, reads no global state, and never inspects Agent
    prompts, images, or memory. Its single input is a
    :class:`ContinuousCastingEvaluationState`; its single output is
    a :class:`ContinuousCastingEvaluationResult`.

    Failure classification priority (most specific first)
    ---------------------------------------------------

    1. :data:`OUTCOME_STEP_BUDGET_EXCEEDED` — the episode's step
       budget was exceeded before the per-cell verdicts could be
       established.
    2. :data:`OUTCOME_TIME_BUDGET_EXCEEDED` — the episode's time
       budget was exceeded.
    3. :data:`OUTCOME_INVALID_INITIAL_STATE` — at least one cell was
       already obsidian at reset. Continuous casting requires a
       non-obsidian initial state for *every* cell; this is a
       fail-closed verdict.
    4. :data:`OUTCOME_ABNORMAL_TERMINATION` — the episode was
       terminated for a reason that is not in
       :data:`NORMAL_TERMINATION_REASONS`.
    5. :data:`OUTCOME_TRUTH_MISSING` — at least one cell is missing
       a required evaluator truth *or* the cells tuple has the wrong
       length.
    6. :data:`OUTCOME_IN_PROGRESS` — the episode has not been
       terminated yet. The evaluator must not return
       :data:`OUTCOME_SUCCESS` until the termination signal is set.
    7. :data:`OUTCOME_CAUSALITY_MISSING` — at least one cell became
       obsidian but the block update is *not* within the finite
       causality window after one of its own relevant Agent actions
       (or no relevant Agent action is recorded, or the update
       precedes the actions).
    8. :data:`OUTCOME_PARTIAL_COMPLEMENT` — a non-empty strict prefix
       of the ordered cells succeeded and all remaining cells have
       truth-complete wrong-block verdicts.
    9. :data:`OUTCOME_WRONG_BLOCK` — no cell succeeded, or success
       contains an out-of-order hole rather than a valid prefix.
    10. :data:`OUTCOME_SUCCESS` — every cell succeeded and the
        episode terminated normally within budget.
    """

    def evaluate(
        self, state: ContinuousCastingEvaluationState
    ) -> ContinuousCastingEvaluationResult:
        outcome, failure_step, completed_cells, per_cell_outcomes, first_failed, evidence = (
            _classify_outcome(state)
        )
        success = outcome == OUTCOME_SUCCESS
        if outcome in _TERMINAL_FAILURE_OUTCOMES:
            failure_type: str | None = outcome
        else:
            failure_type = None
        return ContinuousCastingEvaluationResult(
            episode_id=state.episode_id,
            step_id=state.step_id,
            success=success,
            outcome=outcome,
            completed_cells=completed_cells,
            total_cells=len(state.cells),
            per_cell_outcomes=per_cell_outcomes,
            first_failed_cell=first_failed,
            blocking_conditions=_blocking_conditions(
                outcome, state, per_cell_outcomes, evidence
            ),
            evidence=evidence,
            failure_type=failure_type,
            failure_step=failure_step,
            episode_terminated=state.episode_terminated,
            terminated_step=state.terminated_step,
            terminated_reason=state.terminated_reason,
        )


def _classify_outcome(
    state: ContinuousCastingEvaluationState,
) -> tuple[str, int | None, int, tuple[str, ...], int | None, Mapping[str, Any]]:
    """Apply the priority rules and return the result payload."""
    total_cells = len(state.cells)
    evidence: dict[str, Any] = {
        "episode_id": state.episode_id,
        "step_id": state.step_id,
        "agent_id": state.agent_id,
        "max_environment_steps": state.max_environment_steps,
        "max_game_time_seconds": state.max_game_time_seconds,
        "current_time_seconds": state.current_time_seconds,
        "causality_window_steps": state.causality_window_steps,
        "target_cells": [list(cell.target_cell) for cell in state.cells],
    }

    # 1. step budget
    step_budget_step: int | None = None
    observed = [state.step_id]
    if state.terminated_step is not None:
        observed.append(state.terminated_step)
    latest_observed = max(observed)
    if latest_observed > state.max_environment_steps:
        step_budget_step = latest_observed
    if step_budget_step is not None:
        evidence["budget_exceeded_kind"] = "step"
        evidence["budget_exceeded_value"] = step_budget_step
        evidence["budget_limit"] = state.max_environment_steps
        return (
            OUTCOME_STEP_BUDGET_EXCEEDED,
            step_budget_step,
            0,
            (PER_CELL_NOT_EVALUATED,) * total_cells,
            None,
            MappingProxyType(evidence),
        )

    # 2. time budget
    if state.current_time_seconds > state.max_game_time_seconds:
        evidence["budget_exceeded_kind"] = "time"
        evidence["budget_exceeded_value"] = state.current_time_seconds
        evidence["budget_limit"] = state.max_game_time_seconds
        return (
            OUTCOME_TIME_BUDGET_EXCEEDED,
            state.terminated_step if state.episode_terminated else state.step_id,
            0,
            (PER_CELL_NOT_EVALUATED,) * total_cells,
            None,
            MappingProxyType(evidence),
        )

    # 3. invalid initial state — any cell starting as obsidian fails closed
    for index, cell in enumerate(state.cells):
        if cell.initial_block == "obsidian":
            evidence["invalid_initial_cell"] = index
            return (
                OUTCOME_INVALID_INITIAL_STATE,
                state.step_id,
                0,
                (PER_CELL_NOT_EVALUATED,) * total_cells,
                index,
                MappingProxyType(evidence),
            )

    # 4. truth_missing before in_progress: even an in-progress
    # episode already has cells whose truth is missing. We check
    # this *before* the in_progress branch so a partially
    # injected state still fails closed.
    if state.episode_terminated:
        # 5. abnormal_termination: episode ended for a non-normal
        # reason. This outranks per-cell verdicts because the
        # episode was already corrupted.
        if (
            state.terminated_reason is not None
            and state.terminated_reason not in NORMAL_TERMINATION_REASONS
        ):
            evidence["terminated_reason"] = state.terminated_reason
            return (
                OUTCOME_ABNORMAL_TERMINATION,
                state.terminated_step,
                0,
                (PER_CELL_NOT_EVALUATED,) * total_cells,
                None,
                MappingProxyType(evidence),
            )

    # 6-10: per-cell verdicts + aggregation
    per_cell: list[tuple[str, int | None, dict[str, Any]]] = []
    for cell in state.cells:
        per_cell.append(
            _classify_cell(
                cell,
                causality_window_steps=state.causality_window_steps,
                episode_terminated=state.episode_terminated,
                terminated_step=state.terminated_step,
            )
        )
    per_cell_outcomes = tuple(verdict for verdict, _step, _evidence in per_cell)
    completed_cells = sum(
        1 for verdict, _step, _evidence in per_cell if verdict == PER_CELL_SUCCESS
    )
    first_failed: int | None = None
    for index, verdict in enumerate(per_cell_outcomes):
        if verdict != PER_CELL_SUCCESS and verdict != PER_CELL_NOT_EVALUATED:
            first_failed = index
            break
    # When the episode is in progress we still want first_failed to
    # point at the first cell that has a definitive issue
    # (truth_missing / wrong_block / causality_missing), not the
    # first not_evaluated cell. The loop above already enforces that
    # ordering: it scans the verdicts in cell order and stops at the
    # first one that is not a clean success / not_evaluated.

    aggregate_evidence: dict[str, Any] = dict(evidence)
    aggregate_evidence["per_cell_evidence"] = [
        _thaw_json_value(cell_evidence)
        for _verdict, _step, cell_evidence in per_cell
    ]
    outcome, failure_step, agg_extras = _aggregate_outcome(
        per_cell,
        episode_terminated=state.episode_terminated,
        terminated_step=state.terminated_step,
    )
    aggregate_evidence.update(agg_extras)
    return (
        outcome,
        failure_step,
        completed_cells,
        per_cell_outcomes,
        first_failed,
        MappingProxyType(aggregate_evidence),
    )


def _blocking_conditions(
    outcome: str,
    state: ContinuousCastingEvaluationState,
    per_cell_outcomes: tuple[str, ...],
    evidence: Mapping[str, Any],
) -> tuple[str, ...]:
    if outcome == OUTCOME_SUCCESS:
        return ()
    if outcome == OUTCOME_IN_PROGRESS:
        return ("episode_not_terminated",)
    if outcome == OUTCOME_STEP_BUDGET_EXCEEDED:
        return ("step_budget_exceeded",)
    if outcome == OUTCOME_TIME_BUDGET_EXCEEDED:
        return ("time_budget_exceeded",)
    if outcome == OUTCOME_INVALID_INITIAL_STATE:
        cell_index = evidence.get("invalid_initial_cell")
        if isinstance(cell_index, int):
            return (f"invalid_initial_state:cell_{cell_index}",)
        return ("invalid_initial_state",)
    if outcome == OUTCOME_TRUTH_MISSING:
        cells_evidence = evidence.get("per_cell_evidence", [])
        conditions: list[str] = []
        for index, cell_evidence in enumerate(cells_evidence):
            missing = cell_evidence.get("missing_truth") if isinstance(
                cell_evidence, Mapping
            ) else None
            if isinstance(missing, (list, tuple)):
                for name in missing:
                    conditions.append(f"missing_truth:cell_{index}.{name}")
        if not conditions:
            conditions.append("missing_truth")
        return tuple(conditions)
    if outcome == OUTCOME_CAUSALITY_MISSING:
        cells_evidence = evidence.get("per_cell_evidence", [])
        conditions = []
        for index, cell_evidence in enumerate(cells_evidence):
            if per_cell_outcomes[index] != PER_CELL_CAUSALITY_MISSING:
                continue
            reason = cell_evidence.get("causality_reason") if isinstance(
                cell_evidence, Mapping
            ) else None
            if isinstance(reason, str) and reason:
                conditions.append(f"causality_missing:cell_{index}:{reason}")
            else:
                conditions.append(f"causality_missing:cell_{index}")
        if not conditions:
            return ("causality_missing",)
        return tuple(conditions)
    if outcome == OUTCOME_WRONG_BLOCK:
        cells_evidence = evidence.get("per_cell_evidence", [])
        conditions = []
        for index, cell_evidence in enumerate(cells_evidence):
            if per_cell_outcomes[index] != PER_CELL_WRONG_BLOCK:
                continue
            actual = cell_evidence.get("actual_block") if isinstance(
                cell_evidence, Mapping
            ) else None
            if isinstance(actual, str) and actual:
                conditions.append(
                    f"wrong_block:cell_{index}:expected_obsidian_got_{actual}"
                )
            else:
                conditions.append(f"wrong_block:cell_{index}")
        if not conditions:
            return ("wrong_block",)
        return tuple(conditions)
    if outcome == OUTCOME_PARTIAL_COMPLEMENT:
        success_cells = evidence.get("success_cells", [])
        non_success_cells = evidence.get("non_success_cells", [])
        if isinstance(success_cells, (list, tuple)) and isinstance(
            non_success_cells, int
        ):
            return (
                f"partial_completion:completed_{len(success_cells)}_of_"
                f"{non_success_cells + len(success_cells)}",
            )
        return ("partial_completion",)
    if outcome == OUTCOME_ABNORMAL_TERMINATION:
        return ("abnormal_termination",)
    return ()


__all__ = [
    "ContinuousCastingCellTruth",
    "ContinuousCastingEvaluationResult",
    "ContinuousCastingEvaluationState",
    "ContinuousCastingEvaluator",
    "CASTING_C3_TARGET_CELLS",
    "CONTINUOUS_OUTCOMES",
    "DEFAULT_C3_CELL_BUDGET",
    "DEFAULT_CELL_COUNT",
    "MAX_CELL_COUNT",
    "MIN_CELL_COUNT",
    "NORMAL_TERMINATION_REASONS",
    "OUTCOME_ABNORMAL_TERMINATION",
    "OUTCOME_CAUSALITY_MISSING",
    "OUTCOME_IN_PROGRESS",
    "OUTCOME_INVALID_INITIAL_STATE",
    "OUTCOME_PARTIAL_COMPLEMENT",
    "OUTCOME_STEP_BUDGET_EXCEEDED",
    "OUTCOME_SUCCESS",
    "OUTCOME_TIME_BUDGET_EXCEEDED",
    "OUTCOME_TRUTH_MISSING",
    "OUTCOME_WRONG_BLOCK",
    "PER_CELL_CAUSALITY_MISSING",
    "PER_CELL_NOT_EVALUATED",
    "PER_CELL_SUCCESS",
    "PER_CELL_TRUTH_MISSING",
    "PER_CELL_WRONG_BLOCK",
    "_PER_CELL_VERDICTS",
    "_TERMINAL_FAILURE_OUTCOMES",
]

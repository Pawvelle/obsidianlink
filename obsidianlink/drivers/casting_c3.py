"""R5 deterministic continuous-casting driver for ``casting_c3_fixed``.

This module extends the R4 single-block driver to a fixed, short
multi-cell straight line segment (the minimum viable "almost a portal
side" slice). The driver never grows the world: it still uses a
single :class:`MacroAction` per backend step and never executes
model-generated code.

Design contract
---------------

* The driver only consumes Agent-visible
  :class:`~obsidianlink.core.types.Observation` data
  (``visible_inventory`` / ``workflow_stage`` / ``step_id``). It
  never reads casting truth, target-cell truth, fluid truth, or any
  evaluator-only field. The AST / spy / ``__getattribute__`` tests
  in :mod:`tests.test_continuous_casting_driver` pin this contract
  with the same three gates the R4 suite uses, plus a fourth gate
  that scans for the ``ContinuousCasting*`` surface.
* The driver never calls
  :meth:`FakeEnvironmentBackend.set_continuous_casting_evaluation_state`
  or :meth:`FakeEnvironmentBackend.get_continuous_casting_evaluation_state`.
  Truth injection lives in the test orchestrator
  (:mod:`tests.test_continuous_casting_driver`); the driver surface
  has no access to those methods.
* The driver only emits :class:`MacroAction` values from the
  project's public action protocol: ``equip_item`` / ``use_item`` /
  ``place_block`` / ``wait``. The driver's
  :func:`build_continuous_casting_action_plan` is the single source
  of truth for the action sequence; no other action sequence is
  allowed at runtime.
* The plan is a fixed, ordered, finite tuple of plan steps. Each
  step carries the target :attr:`cell_index`, the workflow
  :attr:`phase`, a semantic :attr:`label`, a :class:`MacroAction`,
  and a ``relevant_action`` boolean. The plan builder is the only
  place that constructs plan steps; the driver only walks them.
* Every step / time / wait / plan length / recovery budget has a
  hard, type-explicit cap. The driver refuses to start a step that
  would exceed any cap; budget exhaustion is reported as
  ``status="blocked"`` with a descriptive ``blocked_reason``.
* The driver's recovery protocol is a **deterministic, finite,
  public-signal-only** retry loop. The recovery signal is the typed
  :class:`obsidianlink.core.types.RecoverableBackendError` exception
  raised by :meth:`EnvironmentBackend.step`. The driver catches the
  specific subclass (not the bare :class:`RuntimeError`), counts the
  attempt, and either re-submits the same action (if the per-step
  recovery budget is not exhausted and the total recovery budget is
  not exhausted) or fails closed with ``status="blocked"``. The
  driver never reads evaluator truth to decide whether to retry.

Termination contract
--------------------

The driver does not terminate the episode by itself. It always
returns a :class:`CastingC3DriverResult` and relies on the calling
orchestrator to mark the episode terminated and feed the final
state into the :class:`ContinuousCastingEvaluator`. This mirrors the
R3 / R4 evaluator contract: the ``episode_terminated`` field is the
only thing that flips the evaluator from ``in_progress`` to a
terminal outcome, and only the orchestrator / environment should
flip it.

The :class:`CastingC3DriverResult` carries:

* ``status`` — one of ``"completed"`` / ``"blocked"`` / ``"failed"``;
* ``steps_executed`` / ````wait_steps`` / ``planned_steps`` /
  ``recovery_attempts`` / ``recovery_total`` — bounded counters
  useful for replay evidence;
* ``per_cell_relevant_action_steps`` — a mapping from ``cell_index``
  to the tuple of step ids at which a relevant action was submitted
  for that cell. The orchestrator uses this mapping to build
  per-cell ``relevant_action_steps`` lists without ever reading
  evaluator truth;
* ``events`` — a tuple of structured event mappings
  (``episode_id`` / ``agent_id`` / ``step_id`` / ``cell_index`` /
  ``label`` / ``phase`` / ``action_type`` / ``target`` /
  ``relevant_action`` / ``attempt``);
* ``action_label_for_step`` — a mapping from ``step_id`` to the
  final semantic label produced at that step;
* ``terminated`` / ``truncated`` — the final flags reported by the
  backend. The driver never fabricates either flag.
* ``final_observation`` — the most recent Observation the driver
  received (used by the orchestrator for evidence).
* ``as_dict`` — returns a detached snapshot for evidence logging.

The driver never returns ``status == "passed"`` / ``"success"``.
The driver reports whether it *reached* the end of the bounded
plan; the orchestrator owns the evaluator verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

from obsidianlink.core.types import (
    BackendStep,
    MacroAction,
    Observation,
    RecoverableBackendError,
    TaskInstance,
)


AGENT_ID = "agent_1"


# Closed R5 action allowlist. Mirrors the R4 allowlist because the
# per-cell casting sequence uses the same public action protocol.
ALLOWED_C3_ACTION_TYPES: frozenset[str] = frozenset(
    {"equip_item", "use_item", "place_block", "wait"}
)


# Targets the driver is allowed to use. ``place_block`` is allowed
# only for the cobblestone support block; ``use_item`` / ``equip_item``
# are allowed only for the two buckets.
ALLOWED_C3_TARGETS: frozenset[str] = frozenset(
    {
        "water_bucket",
        "lava_bucket",
        "cobblestone",
    }
)


# Phase labels used in the structured event log. Phases are not
# read by the driver; they are emitted for evidence only.
PHASE_PREPARE = "prepare"
PHASE_PLACE_SUPPORT = "place_support"
PHASE_PLACE_LAVA = "place_lava"
PHASE_PLACE_WATER = "place_water"
PHASE_WAIT_FOR_OBSIDIAN = "wait_for_obsidian"
PHASE_RECOVERY = "recovery"

PHASE_VALUES: frozenset[str] = frozenset(
    {
        PHASE_PREPARE,
        PHASE_PLACE_SUPPORT,
        PHASE_PLACE_LAVA,
        PHASE_PLACE_WATER,
        PHASE_WAIT_FOR_OBSIDIAN,
        PHASE_RECOVERY,
    }
)


# Defaults for the bounded plan. Each value is a *hard* upper
# bound; smaller values are accepted, larger values are rejected.
DEFAULT_CELL_COUNT: int = 3
MAX_CELL_COUNT: int = 8
MIN_CELL_COUNT: int = 1

DEFAULT_MAX_WAIT_STEPS: int = 96
# ``MAX_C3_PLAN_WAIT_STEPS`` is the hard cap for the *total* wait
# count of the *entire* R5 plan. The default plan's wait count is
# 3 cells × 16 waits = 48, well under the cap; the cap still fires
# if a caller injects an unbounded loop.
MAX_C3_PLAN_WAIT_STEPS: int = DEFAULT_MAX_WAIT_STEPS

DEFAULT_SUPPORT_BLOCK_WAIT_STEPS: int = 1
DEFAULT_FLUID_SETTLE_WAIT_STEPS: int = 4
DEFAULT_OBSIDIAN_WAIT_STEPS: int = 4

# Step / time budget defaults match the ``casting_c3_fixed`` task
# contract. The driver consults them to decide when to give up
# waiting even before the environment reports termination.
DEFAULT_MAX_ENVIRONMENT_STEPS: int = 240
DEFAULT_MAX_GAME_TIME_SECONDS: float = 180.0

# Recovery budget. The driver may retry the same action at most
# ``RECOVERIES_PER_ACTION_DEFAULT`` times, and the total number of
# recoveries across the whole plan is bounded by
# ``TOTAL_RECOVERY_BUDGET_DEFAULT``. Both caps are hard.
RECOVERIES_PER_ACTION_DEFAULT: int = 1
TOTAL_RECOVERY_BUDGET_DEFAULT: int = 3
MAX_RECOVERIES_PER_ACTION: int = 2
MAX_TOTAL_RECOVERY_BUDGET: int = 8


# Terminal driver statuses. The driver never returns
# ``"passed"`` / ``"success"``; those are reserved for the
# evaluator.
DRIVER_STATUS_COMPLETED = "completed"
DRIVER_STATUS_BLOCKED = "blocked"
DRIVER_STATUS_FAILED = "failed"
DRIVER_STATUSES: frozenset[str] = frozenset(
    {DRIVER_STATUS_COMPLETED, DRIVER_STATUS_BLOCKED, DRIVER_STATUS_FAILED}
)


# ----------------------------------------------------------------------
# Validation helpers (private, shared by plan builder and driver)
# ----------------------------------------------------------------------


def _require_positive_int(value: int, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_non_negative_int(value: int, name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _require_positive_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _freeze_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("driver evidence numbers must be finite")
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    raise ValueError(
        f"driver evidence must be JSON-compatible, got {type(value).__name__}"
    )


def _thaw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return value


def _require_c3_action(action: MacroAction, *, context: str) -> MacroAction:
    """Validate that ``action`` belongs to the closed C3 allowlist.

    The driver is the only place where this check is enforced.
    """
    if not isinstance(action, MacroAction):
        raise ValueError(f"{context}: action must be a MacroAction")
    if action.action_type not in ALLOWED_C3_ACTION_TYPES:
        raise ValueError(
            f"{context}: action_type {action.action_type!r} is outside the "
            f"C3 allowlist {sorted(ALLOWED_C3_ACTION_TYPES)}"
        )
    if action.target is not None and action.target not in ALLOWED_C3_TARGETS:
        raise ValueError(
            f"{context}: target {action.target!r} is outside the C3 allowlist "
            f"{sorted(ALLOWED_C3_TARGETS)}"
        )
    if action.action_type == "equip_item" and action.target not in {
        "water_bucket",
        "lava_bucket",
    }:
        raise ValueError(
            f"{context}: equip_item target must be 'water_bucket' or "
            f"'lava_bucket', got {action.target!r}"
        )
    if action.action_type == "wait" and action.target is not None:
        raise ValueError(f"{context}: wait action cannot have a target")
    if action.action_type == "place_block" and action.target != "cobblestone":
        raise ValueError(
            f"{context}: place_block target must be 'cobblestone', got "
            f"{action.target!r}"
        )
    if action.action_type == "use_item" and action.target not in {
        "water_bucket",
        "lava_bucket",
    }:
        raise ValueError(
            f"{context}: use_item target must be 'water_bucket' or "
            f"'lava_bucket', got {action.target!r}"
        )
    if not 1 <= action.duration_ticks <= 40:
        raise ValueError(f"{context}: duration_ticks must be between 1 and 40")
    if action.parameters:
        raise ValueError(f"{context}: C3 actions cannot contain parameters")
    return action


# ----------------------------------------------------------------------
# Plan step
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ContinuousCastingPlanStep:
    """One step in the bounded R5 multi-cell plan.

    ``cell_index`` selects the target cell this step works on.
    ``phase`` is the workflow stage. ``action`` is the (already
    whitelisted) :class:`MacroAction` to submit. ``relevant_action``
    is ``True`` when this step is a candidate for the
    per-cell ``relevant_action_steps`` list (i.e. it places a fluid
    or a support block that could be causally linked to that cell's
    target-cell block update). ``recoveries_allowed`` is the
    per-step recovery budget; the driver never consults the
    evaluator to decide whether to retry.
    """

    cell_index: int
    label: str
    phase: str
    action: MacroAction
    relevant_action: bool = False
    recoveries_allowed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("plan step label must be a non-empty string")
        _require_non_negative_int(self.cell_index, "cell_index")
        if self.phase not in PHASE_VALUES:
            raise ValueError(f"unknown continuous casting plan phase: {self.phase!r}")
        if type(self.relevant_action) is not bool:
            raise ValueError("relevant_action must be a boolean")
        if (
            type(self.recoveries_allowed) is not int
            or isinstance(self.recoveries_allowed, bool)
            or self.recoveries_allowed < 0
            or self.recoveries_allowed > MAX_RECOVERIES_PER_ACTION
        ):
            raise ValueError(
                "recoveries_allowed must be an int between 0 and "
                f"{MAX_RECOVERIES_PER_ACTION}"
            )
        _require_c3_action(self.action, context=f"plan[{self.label!r}]")
        expected_relevant = self.action.action_type in {
            "place_block",
            "use_item",
        }
        if self.relevant_action is not expected_relevant:
            raise ValueError(
                "relevant_action must be true exactly for place_block/use_item"
            )
        if self.phase == PHASE_RECOVERY:
            raise ValueError(
                "PHASE_RECOVERY is reserved for driver-internal events; "
                "the plan builder must not emit it"
            )


# ----------------------------------------------------------------------
# Driver result
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CastingC3DriverResult:
    """Public result of :func:`run_casting_c3_driver`.

    The driver never returns a casting verdict; the orchestrator
    owns the evaluator call. This object only reports whether the
    driver reached the end of the bounded plan, which steps it
    executed, and the event log.
    """

    status: str
    steps_executed: int
    wait_steps: int
    planned_steps: int
    recovery_attempts: int
    recovery_budget: int
    per_cell_relevant_action_steps: Mapping[int, tuple[int, ...]]
    final_observation: Observation
    events: tuple[Mapping[str, Any], ...]
    action_label_for_step: Mapping[int, str]
    terminated: bool
    truncated: bool
    blocked_reason: str | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        if self.status not in DRIVER_STATUSES:
            raise ValueError(
                f"driver status must be one of {sorted(DRIVER_STATUSES)}, "
                f"got {self.status!r}"
            )
        _require_non_negative_int(self.steps_executed, "steps_executed")
        _require_non_negative_int(self.wait_steps, "wait_steps")
        _require_positive_int(self.planned_steps, "planned_steps")
        if self.steps_executed > self.planned_steps:
            raise ValueError("steps_executed cannot exceed planned_steps")
        if self.wait_steps > self.steps_executed:
            raise ValueError("wait_steps cannot exceed steps_executed")
        _require_non_negative_int(self.recovery_attempts, "recovery_attempts")
        _require_non_negative_int(self.recovery_budget, "recovery_budget")
        # ``recovery_attempts`` counts every backend raise the
        # driver caught. On a successful run it is bounded by
        # ``recovery_budget``; on a blocked run the last attempt
        # that exhausted the budget is included, so
        # ``recovery_attempts`` can be at most
        # ``recovery_budget + 1``. The bound is enforced by the
        # driver runtime, not by the result validator.
        if self.recovery_attempts > self.recovery_budget + 1:
            raise ValueError(
                "recovery_attempts cannot exceed recovery_budget + 1"
            )
        if not isinstance(self.final_observation, Observation):
            raise ValueError("final_observation must be an Observation")
        if not isinstance(self.events, tuple):
            raise ValueError("events must be a tuple")
        if not isinstance(self.per_cell_relevant_action_steps, Mapping):
            raise ValueError("per_cell_relevant_action_steps must be a mapping")
        for key, value in self.per_cell_relevant_action_steps.items():
            _require_non_negative_int(key, "per_cell_relevant_action_steps key")
            if not isinstance(value, tuple) or any(
                type(step) is not int or isinstance(step, bool) or step < 0
                for step in value
            ):
                raise ValueError(
                    "per_cell_relevant_action_steps values must be tuples of "
                    "non-negative ints"
                )
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise ValueError("terminated and truncated must be booleans")
        if self.status == DRIVER_STATUS_COMPLETED:
            if self.steps_executed != self.planned_steps:
                raise ValueError("completed driver must execute the full plan")
            if self.blocked_reason is not None:
                raise ValueError("completed driver cannot have blocked_reason")
        else:
            if (
                not isinstance(self.blocked_reason, str)
                or not self.blocked_reason.strip()
            ):
                raise ValueError(
                    "blocked/failed driver requires blocked_reason"
                )
            if self.error_type is not None and not isinstance(self.error_type, str):
                raise ValueError("error_type must be a string or None")
        if not isinstance(self.action_label_for_step, Mapping):
            raise ValueError("action_label_for_step must be a mapping")
        frozen_events = tuple(_freeze_value(event) for event in self.events)
        frozen_labels = _freeze_value(self.action_label_for_step)
        frozen_cells = _freeze_value(self.per_cell_relevant_action_steps)
        object.__setattr__(self, "events", frozen_events)
        object.__setattr__(self, "action_label_for_step", frozen_labels)
        object.__setattr__(self, "per_cell_relevant_action_steps", frozen_cells)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "steps_executed": self.steps_executed,
            "wait_steps": self.wait_steps,
            "planned_steps": self.planned_steps,
            "recovery_attempts": self.recovery_attempts,
            "recovery_budget": self.recovery_budget,
            "per_cell_relevant_action_steps": {
                int(cell_index): list(steps)
                for cell_index, steps in self.per_cell_relevant_action_steps.items()
            },
            "terminated": self.terminated,
            "truncated": self.truncated,
            "blocked_reason": self.blocked_reason,
            "error_type": self.error_type,
            "events": _thaw_value(self.events),
            "action_label_for_step": _thaw_value(self.action_label_for_step),
        }


# ----------------------------------------------------------------------
# Plan builder
# ----------------------------------------------------------------------


def _wait_step(
    cell_index: int, label: str, phase: str
) -> ContinuousCastingPlanStep:
    return ContinuousCastingPlanStep(
        cell_index=cell_index,
        label=label,
        phase=phase,
        action=MacroAction.wait(),
    )


def _select_step(
    cell_index: int, target: str, label: str, phase: str
) -> ContinuousCastingPlanStep:
    return ContinuousCastingPlanStep(
        cell_index=cell_index,
        label=label,
        phase=phase,
        action=MacroAction(action_type="equip_item", target=target),
    )


def _place_support_step(
    cell_index: int, label: str, phase: str
) -> ContinuousCastingPlanStep:
    return ContinuousCastingPlanStep(
        cell_index=cell_index,
        label=label,
        phase=phase,
        action=MacroAction(action_type="place_block", target="cobblestone"),
        relevant_action=True,
    )


def _use_bucket_step(
    cell_index: int,
    target: str,
    label: str,
    phase: str,
    *,
    recoveries_allowed: int = 0,
) -> ContinuousCastingPlanStep:
    return ContinuousCastingPlanStep(
        cell_index=cell_index,
        label=label,
        phase=phase,
        action=MacroAction(action_type="use_item", target=target),
        relevant_action=True,
        recoveries_allowed=recoveries_allowed,
    )


def build_continuous_casting_action_plan(
    *,
    cell_count: int = DEFAULT_CELL_COUNT,
    support_block_wait_steps: int = DEFAULT_SUPPORT_BLOCK_WAIT_STEPS,
    fluid_settle_wait_steps: int = DEFAULT_FLUID_SETTLE_WAIT_STEPS,
    obsidian_wait_steps: int = DEFAULT_OBSIDIAN_WAIT_STEPS,
    recoveries_per_use_item: int = RECOVERIES_PER_ACTION_DEFAULT,
) -> tuple[ContinuousCastingPlanStep, ...]:
    """Build the bounded R5 multi-cell casting plan.

    The plan is fully deterministic. For each cell (in fixed index
    order, 0 .. ``cell_count - 1``) the sequence is:

    1. Select lava bucket + brief wait (hotbar equip is not
       instant).
    2. Place a cobblestone support block + settle wait.
    3. Place a second cobblestone support block + settle wait.
    4. Re-select lava bucket + brief wait.
    5. Use lava bucket on the target cell + settle wait. Each
       ``use_item`` step carries a ``recoveries_allowed`` budget so
       the driver can deterministically retry a transient error.
    6. Select water bucket + brief wait.
    7. Use water bucket against the lava + bounded wait for the
       fluid update to complete.
    8. Bounded extra wait so the casting evaluator has a fair
       chance to observe the obsidian transition.

    All wait counts are parameterised but bounded by
    :func:`run_casting_c3_driver` so a caller cannot ask the driver
    to run forever.
    """
    cells = _require_positive_int(cell_count, "cell_count")
    if cells < MIN_CELL_COUNT or cells > MAX_CELL_COUNT:
        raise ValueError(
            f"cell_count must be between {MIN_CELL_COUNT} and {MAX_CELL_COUNT}"
        )
    support_waits = _require_non_negative_int(
        support_block_wait_steps, "support_block_wait_steps"
    )
    fluid_waits = _require_non_negative_int(
        fluid_settle_wait_steps, "fluid_settle_wait_steps"
    )
    obsidian_waits = _require_non_negative_int(
        obsidian_wait_steps, "obsidian_wait_steps"
    )
    per_step_recoveries = _require_non_negative_int(
        recoveries_per_use_item, "recoveries_per_use_item"
    )
    if per_step_recoveries > MAX_RECOVERIES_PER_ACTION:
        raise ValueError(
            "recoveries_per_use_item cannot exceed "
            f"{MAX_RECOVERIES_PER_ACTION}"
        )
    waits_per_cell = (
        3
        + (2 * support_waits)
        + (2 * fluid_waits)
        + obsidian_waits
    )
    total_waits = waits_per_cell * cells
    if total_waits > MAX_C3_PLAN_WAIT_STEPS:
        raise ValueError(
            "continuous casting plan wait steps exceed the hard limit: "
            f"{total_waits} > {MAX_C3_PLAN_WAIT_STEPS}"
        )
    plan: list[ContinuousCastingPlanStep] = []
    for cell_index in range(cells):
        plan.extend(
            [
                _select_step(
                    cell_index, "lava_bucket", f"cell_{cell_index}.prepare.select_lava",
                    PHASE_PREPARE,
                ),
                _wait_step(
                    cell_index,
                    f"cell_{cell_index}.prepare.select_lava.release",
                    PHASE_PREPARE,
                ),
                _place_support_step(
                    cell_index,
                    f"cell_{cell_index}.support.block_1",
                    PHASE_PLACE_SUPPORT,
                ),
            ]
        )
        plan.extend(
            _wait_step(
                cell_index,
                f"cell_{cell_index}.support.block_1.settle.{i + 1}",
                PHASE_PLACE_SUPPORT,
            )
            for i in range(support_waits)
        )
        plan.append(
            _place_support_step(
                cell_index,
                f"cell_{cell_index}.support.block_2",
                PHASE_PLACE_SUPPORT,
            )
        )
        plan.extend(
            _wait_step(
                cell_index,
                f"cell_{cell_index}.support.block_2.settle.{i + 1}",
                PHASE_PLACE_SUPPORT,
            )
            for i in range(support_waits)
        )
        plan.extend(
            [
                _select_step(
                    cell_index,
                    "lava_bucket",
                    f"cell_{cell_index}.casting.select_lava",
                    PHASE_PLACE_LAVA,
                ),
                _wait_step(
                    cell_index,
                    f"cell_{cell_index}.casting.select_lava.release",
                    PHASE_PLACE_LAVA,
                ),
                _use_bucket_step(
                    cell_index,
                    "lava_bucket",
                    f"cell_{cell_index}.casting.use_lava",
                    PHASE_PLACE_LAVA,
                    recoveries_allowed=per_step_recoveries,
                ),
            ]
        )
        plan.extend(
            _wait_step(
                cell_index,
                f"cell_{cell_index}.casting.lava.settle.{i + 1}",
                PHASE_PLACE_LAVA,
            )
            for i in range(fluid_waits)
        )
        plan.extend(
            [
                _select_step(
                    cell_index,
                    "water_bucket",
                    f"cell_{cell_index}.casting.select_water",
                    PHASE_PLACE_WATER,
                ),
                _wait_step(
                    cell_index,
                    f"cell_{cell_index}.casting.select_water.release",
                    PHASE_PLACE_WATER,
                ),
                _use_bucket_step(
                    cell_index,
                    "water_bucket",
                    f"cell_{cell_index}.casting.use_water",
                    PHASE_PLACE_WATER,
                    recoveries_allowed=per_step_recoveries,
                ),
            ]
        )
        plan.extend(
            _wait_step(
                cell_index,
                f"cell_{cell_index}.casting.water.settle.{i + 1}",
                PHASE_PLACE_WATER,
            )
            for i in range(fluid_waits)
        )
        plan.extend(
            _wait_step(
                cell_index,
                f"cell_{cell_index}.casting.obsidian.wait.{i + 1}",
                PHASE_WAIT_FOR_OBSIDIAN,
            )
            for i in range(obsidian_waits)
        )
    for step in plan:
        _require_c3_action(step.action, context=f"plan[{step.label!r}]")
    return tuple(plan)


# ----------------------------------------------------------------------
# Driver implementation
# ----------------------------------------------------------------------


def _visible_inventory_has(observation: Observation, item: str) -> bool:
    if item not in ALLOWED_C3_TARGETS:
        raise ValueError(f"driver cannot inspect item {item!r}")
    inventory = observation.visible_inventory
    if not inventory:
        return False
    quantity = inventory.get(item, 0)
    if type(quantity) is not int or quantity < 0:
        raise ValueError(
            "visible_inventory quantities must be non-negative integers"
        )
    return quantity > 0


def _assert_workflow_stage(observation: Observation) -> None:
    if not isinstance(observation, Observation):
        raise ValueError("expected an Observation")
    if observation.workflow_stage != "casting_c3_fixed":
        raise ValueError(
            "driver only supports workflow 'casting_c3_fixed', got "
            f"{observation.workflow_stage!r}"
        )


def run_casting_c3_driver(
    backend: Any,
    task: TaskInstance,
    *,
    plan: tuple[ContinuousCastingPlanStep, ...] | None = None,
    max_wait_steps: int = DEFAULT_MAX_WAIT_STEPS,
    max_environment_steps: int | None = None,
    max_game_time_seconds: float | None = None,
    total_recovery_budget: int = TOTAL_RECOVERY_BUDGET_DEFAULT,
    recoveries_per_use_item: int = RECOVERIES_PER_ACTION_DEFAULT,
    event_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> CastingC3DriverResult:
    """Run the bounded R5 multi-cell casting plan on ``backend``.

    The driver:

    1. Calls ``backend.reset(task)`` and uses the returned
       ``Observation`` only for the initial ``visible_inventory`` /
       ``workflow_stage`` check. It never reads casting truth.
    2. Walks the plan step-by-step, calling
       ``backend.step({AGENT_ID: action})`` once per step. Each
       step is validated against the closed C3 allowlist before
       submission.
    3. Refuses to start a step that requires an item the Agent is
       not carrying in its visible inventory. The check uses
       ``visible_inventory`` only; the driver never reads
       ``Observation.frame`` or any other field.
    4. Catches the typed :class:`RecoverableBackendError` exception
       raised by ``backend.step`` and applies the deterministic,
       bounded recovery protocol described in the module docstring.
       Any other exception (``RuntimeError`` / ``OSError`` /
       ``TypeError`` not subclassing
       :class:`RecoverableBackendError`) fails closed immediately.
    5. Bounded by ``max_environment_steps`` and
       ``max_game_time_seconds``; when either is exceeded the
       driver returns ``status="blocked"`` with a descriptive
       ``blocked_reason``.

    The driver does *not* call
    ``set_continuous_casting_evaluation_state`` /
    ``get_continuous_casting_evaluation_state`` and does *not*
    invoke :class:`ContinuousCastingEvaluator`. Both live in the
    test orchestrator.
    """
    if not isinstance(task, TaskInstance):
        raise ValueError("task must be a TaskInstance")
    if task.workflow != "casting_c3_fixed":
        raise ValueError(
            f"driver only supports workflow 'casting_c3_fixed', got "
            f"{task.workflow!r}"
        )
    _require_positive_int(max_wait_steps, "max_wait_steps")
    if max_wait_steps > MAX_C3_PLAN_WAIT_STEPS:
        raise ValueError(
            f"max_wait_steps must be <= {MAX_C3_PLAN_WAIT_STEPS}"
        )
    task_step_limit = task.limits["max_environment_steps"]
    task_time_limit = float(task.limits["max_game_time_seconds"])
    if max_environment_steps is None:
        max_environment_steps = task_step_limit
    else:
        _require_positive_int(max_environment_steps, "max_environment_steps")
        if max_environment_steps > task_step_limit:
            raise ValueError(
                "max_environment_steps cannot exceed the task limit "
                f"{task_step_limit}"
            )
    if max_game_time_seconds is None:
        max_game_time_seconds = task_time_limit
    else:
        _require_positive_number(
            max_game_time_seconds, "max_game_time_seconds"
        )
        if max_game_time_seconds > task_time_limit:
            raise ValueError(
                "max_game_time_seconds cannot exceed the task limit "
                f"{task_time_limit}"
            )
    if plan is None:
        plan = build_continuous_casting_action_plan(
            recoveries_per_use_item=recoveries_per_use_item,
        )
    if not isinstance(plan, tuple) or not plan:
        raise ValueError("plan must be non-empty")
    if any(
        not isinstance(step, ContinuousCastingPlanStep) for step in plan
    ):
        raise ValueError("plan must contain only ContinuousCastingPlanStep values")
    if len(plan) > task_step_limit:
        raise ValueError(
            "plan length cannot exceed the task step limit "
            f"{task_step_limit}"
        )
    plan_wait_steps = sum(
        step.action.action_type == "wait" for step in plan
    )
    if plan_wait_steps > MAX_C3_PLAN_WAIT_STEPS:
        raise ValueError(
            "plan wait steps cannot exceed the hard limit "
            f"{MAX_C3_PLAN_WAIT_STEPS}"
        )
    _require_non_negative_int(total_recovery_budget, "total_recovery_budget")
    if total_recovery_budget > MAX_TOTAL_RECOVERY_BUDGET:
        raise ValueError(
            "total_recovery_budget cannot exceed "
            f"{MAX_TOTAL_RECOVERY_BUDGET}"
        )
    if not hasattr(backend, "reset") or not hasattr(backend, "step"):
        raise ValueError("backend must implement reset/step")

    observations = backend.reset(task)
    final_observation = observations[AGENT_ID]
    if not isinstance(final_observation, Observation):
        raise ValueError("backend.reset must return Observation values")
    _assert_workflow_stage(final_observation)
    reset_timestamp = (
        float(final_observation.timestamp)
        if math.isfinite(final_observation.timestamp)
        else None
    )

    events: list[Mapping[str, Any]] = []
    action_label_for_step: dict[int, str] = {}
    per_cell_relevant: dict[int, list[int]] = {}
    wait_steps = 0
    steps_executed = 0
    recovery_attempts = 0
    blocked_reason: str | None = None
    error_type: str | None = None
    status = DRIVER_STATUS_COMPLETED
    backend_terminated = False
    backend_truncated = False

    def record_event(event: Mapping[str, Any]) -> None:
        identified = {
            "episode_id": task.task_id,
            "agent_id": AGENT_ID,
            **dict(event),
        }
        events.append(identified)
        if event_sink is not None:
            event_sink(_thaw_value(_freeze_value(identified)))

    record_event(
        {
            "step_id": final_observation.step_id,
            "cell_index": -1,
            "label": "environment.reset",
            "phase": PHASE_PREPARE,
            "action_type": "wait",
            "target": None,
            "relevant_action": False,
            "attempt": 0,
            "visible_inventory": dict(
                final_observation.visible_inventory or {}
            ),
        }
    )

    def mark_last_event_budget(kind: str) -> None:
        current = events[-1]
        if isinstance(current, dict):
            current["budget_exceeded"] = kind
        else:
            updated = dict(current)
            updated["budget_exceeded"] = kind
            events[-1] = updated

    for plan_index, plan_step in enumerate(plan):
        _require_c3_action(
            plan_step.action,
            context=f"plan[{plan_index}]={plan_step.label!r}",
        )
        if final_observation.step_id >= max_environment_steps:
            blocked_reason = (
                f"step budget exhausted before {plan_step.label}: "
                f"step_id={final_observation.step_id} >= {max_environment_steps}"
            )
            status = DRIVER_STATUS_BLOCKED
            mark_last_event_budget("step")
            break
        if plan_step.action.action_type == "wait" and wait_steps >= max_wait_steps:
            blocked_reason = (
                f"wait budget exhausted before {plan_step.label}: "
                f"wait_steps={wait_steps} >= {max_wait_steps}"
            )
            status = DRIVER_STATUS_BLOCKED
            mark_last_event_budget("wait")
            break
        if plan_step.action.action_type == "equip_item":
            item = plan_step.action.target
            if item is None or not _visible_inventory_has(
                final_observation, item
            ):
                blocked_reason = (
                    f"missing required item {item!r} at {plan_step.label}"
                )
                status = DRIVER_STATUS_BLOCKED
                break
        elif plan_step.action.action_type == "use_item":
            item = plan_step.action.target
            if item is None or not _visible_inventory_has(
                final_observation, item
            ):
                blocked_reason = (
                    f"missing required item {item!r} at {plan_step.label}"
                )
                status = DRIVER_STATUS_BLOCKED
                break
        elif plan_step.action.action_type == "place_block":
            if not _visible_inventory_has(final_observation, "cobblestone"):
                blocked_reason = (
                    "missing required item 'cobblestone' at "
                    f"{plan_step.label}"
                )
                status = DRIVER_STATUS_BLOCKED
                break

        attempt = 0
        submitted = False
        step_attempts_allowed = max(0, plan_step.recoveries_allowed)
        while not submitted:
            previous_step_id = final_observation.step_id
            try:
                step = backend.step({AGENT_ID: plan_step.action})
            except RecoverableBackendError as error:
                recovery_attempts += 1
                attempt += 1
                record_event(
                    {
                        "step_id": previous_step_id,
                        "cell_index": plan_step.cell_index,
                        "label": plan_step.label,
                        "phase": PHASE_RECOVERY,
                        "action_type": plan_step.action.action_type,
                        "target": plan_step.action.target,
                        "relevant_action": False,
                        "attempt": attempt,
                        "recoverable_kind": error.recoverable_kind,
                        "recoverable_message": str(error),
                    }
                )
                if recovery_attempts > total_recovery_budget:
                    blocked_reason = (
                        f"recovery budget exhausted at {plan_step.label}: "
                        f"recovery_attempts={recovery_attempts} > "
                        f"total_recovery_budget={total_recovery_budget}"
                    )
                    status = DRIVER_STATUS_BLOCKED
                    break
                if attempt > step_attempts_allowed:
                    blocked_reason = (
                        f"per-step recovery budget exhausted at {plan_step.label}: "
                        f"attempt={attempt} > recoveries_allowed={step_attempts_allowed}"
                    )
                    status = DRIVER_STATUS_BLOCKED
                    break
                continue
            except (RuntimeError, OSError, TypeError) as error:
                blocked_reason = (
                    f"{type(error).__name__} at {plan_step.label}: {error}"
                )
                status = DRIVER_STATUS_FAILED
                error_type = type(error).__name__
                record_event(
                    {
                        "step_id": getattr(error, "step_id", None),
                        "cell_index": plan_step.cell_index,
                        "label": plan_step.label,
                        "phase": plan_step.phase,
                        "action_type": plan_step.action.action_type,
                        "target": plan_step.action.target,
                        "relevant_action": plan_step.relevant_action,
                        "attempt": attempt,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                break
            if not isinstance(step, BackendStep):
                raise ValueError("backend.step must return a BackendStep")
            if step.episode_id != task.task_id:
                raise ValueError(
                    "BackendStep episode_id must match the current task"
                )
            if step.step_id != previous_step_id + 1:
                raise ValueError(
                    "BackendStep step_id must advance exactly once: "
                    f"expected {previous_step_id + 1}, got {step.step_id}"
                )
            steps_executed += 1
            attempt += 1
            if plan_step.action.action_type == "wait":
                wait_steps += 1
            next_observation = step.observations[AGENT_ID]
            if not isinstance(next_observation, Observation):
                raise ValueError("backend.step must return Observation values")
            _assert_workflow_stage(next_observation)
            backend_terminated = step.terminated
            backend_truncated = step.truncated

            if (
                reset_timestamp is not None
                and math.isfinite(next_observation.timestamp)
            ):
                elapsed = next_observation.timestamp - reset_timestamp
                if elapsed > max_game_time_seconds:
                    blocked_reason = (
                        f"time budget exceeded at {plan_step.label}: "
                        f"elapsed={elapsed:.3f}s > "
                        f"max_game_time_seconds={max_game_time_seconds}"
                    )
                    status = DRIVER_STATUS_BLOCKED
                    final_observation = next_observation
                    action_label_for_step[step.step_id] = plan_step.label
                    if plan_step.relevant_action:
                        per_cell_relevant.setdefault(
                            plan_step.cell_index, []
                        ).append(step.step_id)
                    record_event(
                        {
                            "step_id": step.step_id,
                            "cell_index": plan_step.cell_index,
                            "label": plan_step.label,
                            "phase": plan_step.phase,
                            "action_type": plan_step.action.action_type,
                            "target": plan_step.action.target,
                            "relevant_action": plan_step.relevant_action,
                            "attempt": attempt,
                            "budget_exceeded": "time",
                        }
                    )
                    submitted = True
                    break
            final_observation = next_observation
            action_label_for_step[step.step_id] = plan_step.label
            if plan_step.relevant_action:
                per_cell_relevant.setdefault(
                    plan_step.cell_index, []
                ).append(step.step_id)
            record_event(
                {
                    "step_id": step.step_id,
                    "cell_index": plan_step.cell_index,
                    "label": plan_step.label,
                    "phase": plan_step.phase,
                    "action_type": plan_step.action.action_type,
                    "target": plan_step.action.target,
                    "relevant_action": plan_step.relevant_action,
                    "attempt": attempt,
                    "visible_inventory": dict(
                        final_observation.visible_inventory or {}
                    ),
                }
            )
            submitted = True
            if step.terminated or step.truncated:
                if plan_index + 1 < len(plan):
                    status = DRIVER_STATUS_BLOCKED
                    reason = "termination" if step.terminated else "truncation"
                    blocked_reason = (
                        f"plan interrupted by backend {reason} at {plan_step.label}"
                    )
                break
        if status != DRIVER_STATUS_COMPLETED:
            break

    if blocked_reason is None and steps_executed < len(plan):
        blocked_reason = "plan interrupted by backend termination"
        status = DRIVER_STATUS_BLOCKED

    frozen_per_cell = MappingProxyType(
        {
            int(cell_index): tuple(steps)
            for cell_index, steps in per_cell_relevant.items()
        }
    )
    return CastingC3DriverResult(
        status=status,
        steps_executed=steps_executed,
        wait_steps=wait_steps,
        planned_steps=len(plan),
        recovery_attempts=recovery_attempts,
        recovery_budget=total_recovery_budget,
        per_cell_relevant_action_steps=frozen_per_cell,
        final_observation=final_observation,
        events=tuple(events),
        action_label_for_step=MappingProxyType(action_label_for_step),
        terminated=backend_terminated,
        truncated=backend_truncated,
        blocked_reason=blocked_reason,
        error_type=error_type,
    )


__all__ = [
    "AGENT_ID",
    "ALLOWED_C3_ACTION_TYPES",
    "ALLOWED_C3_TARGETS",
    "DEFAULT_CELL_COUNT",
    "DEFAULT_FLUID_SETTLE_WAIT_STEPS",
    "DEFAULT_MAX_ENVIRONMENT_STEPS",
    "DEFAULT_MAX_GAME_TIME_SECONDS",
    "DEFAULT_MAX_WAIT_STEPS",
    "DEFAULT_OBSIDIAN_WAIT_STEPS",
    "DEFAULT_SUPPORT_BLOCK_WAIT_STEPS",
    "MAX_C3_PLAN_WAIT_STEPS",
    "MAX_CELL_COUNT",
    "MAX_RECOVERIES_PER_ACTION",
    "MAX_TOTAL_RECOVERY_BUDGET",
    "MIN_CELL_COUNT",
    "RECOVERIES_PER_ACTION_DEFAULT",
    "TOTAL_RECOVERY_BUDGET_DEFAULT",
    "DRIVER_STATUS_BLOCKED",
    "DRIVER_STATUS_COMPLETED",
    "DRIVER_STATUS_FAILED",
    "DRIVER_STATUSES",
    "PHASE_PLACE_LAVA",
    "PHASE_PLACE_SUPPORT",
    "PHASE_PLACE_WATER",
    "PHASE_PREPARE",
    "PHASE_RECOVERY",
    "PHASE_WAIT_FOR_OBSIDIAN",
    "PHASE_VALUES",
    "CastingC3DriverResult",
    "ContinuousCastingPlanStep",
    "build_continuous_casting_action_plan",
    "run_casting_c3_driver",
]

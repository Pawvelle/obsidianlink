"""R6 Casting-S-C3 deterministic frame driver for ``casting_s_c3_fixed``.

This module implements a bounded, deterministic FakeBackend driver for
the R6 Casting-S-C3 / fixed contract: use water, lava, cobblestone
support blocks, and vanilla block updates to cast the public 4×5
full-ring of 14 obsidian cells. C3 explicitly does **not** ignite the
frame, does **not** enter the Nether, and does **not** directly place
obsidian. The driver walks a fixed, ordered, finite plan on the
:class:`FakeEnvironmentBackend` and never reads evaluator-only truth.

This is *not* the C2 / R5 driver in
:mod:`obsidianlink.drivers.casting_c3`. That module implements the
C2 (three-cell straight-line segment) continuous casting contract; the
``c3`` in its name refers to three cells, not the B0 taxonomy C3
(full 4×5 ring). The naming is intentionally distinct: this new
module is named after the *taxonomy* C3 (full ring) contract, while
the older driver is named after the *historical* C3 (three cells)
task instance. The two drivers never share code at the type level.

Design contract
---------------

* The driver only consumes Agent-visible
  :class:`~obsidianlink.core.types.Observation` data
  (``visible_inventory`` / ``workflow_stage`` / ``step_id``) and the
  strictly-validated, immutable
  :class:`PublicC3FrameDriverContext` passed in by the orchestrator.
  It never reads evaluator-only truth, ``scenario_parameters``,
  ``evaluator_contract``, :class:`FrozenFrameEvaluationState` or any
  other evaluator internal state. The AST / spy / ``__getattribute__``
  tests in :mod:`tests.test_r6_casting_c3_frame_driver` pin this
  contract with five gates: the AST gate, the source-level
  ``scenario_parameters``/``evaluator_contract``/``FrozenFrame*``
  gate, the set/get/clear frame-truth gate, the
  :class:`FakeEnvironmentBackend` ``_frame_evaluation_state`` gate,
  and the Observation ``__getattribute__`` guard.
* The driver only emits :class:`MacroAction` values from the
  project's public action protocol: ``equip_item`` / ``use_item`` /
  ``place_block`` / ``wait``. The driver's
  :func:`build_casting_s_c3_frame_action_plan` is the single source
  of truth for the action sequence; no other action sequence is
  allowed at runtime.
* The 14 target offsets come from a strictly-validated, immutable
  :class:`PublicC3FrameDriverContext` constructed by the orchestrator
  from the task's
  ``scenario_parameters.public_task_spec.frame_plan.fixed_offsets``.
  The driver itself never reads ``scenario_parameters`` or
  ``evaluator_contract``; the orchestrator extracts the public spec
  and passes the frozen context to the driver.
* The plan is a fixed, ordered, finite tuple of plan steps. Each
  step carries the target :attr:`cell_index`, the public
  :attr:`target_offset`, the workflow :attr:`phase`, a semantic
  :attr:`label`, a :class:`MacroAction`, and a ``relevant_action``
  boolean. ``relevant_action`` is ``True`` exactly for
  ``use_item(water_bucket | lava_bucket)`` because the
  :class:`~obsidianlink.evaluation.casting_frame_evaluator.FrozenFrameActionEvidence`
  contract only accepts those two ``use_item`` variants. The
  ``equip_item`` and ``place_block`` actions are part of the plan
  (they are needed for the casting mechanics) but they are not
  relevant for evaluator attribution.
* Every step / time / wait / plan length / recovery budget has a
  hard, type-explicit cap. The driver refuses to start a step that
  would exceed any cap; budget exhaustion is reported as
  ``status="blocked"`` with a descriptive ``blocked_reason``.
* The driver's recovery protocol is a **deterministic, finite,
  public-signal-only** retry loop. The recovery signal is the typed
  :class:`~obsidianlink.core.types.RecoverableBackendError` exception
  raised by ``backend.step``. The driver catches the specific
  subclass (not the bare :class:`RuntimeError`), counts the attempt,
  and either re-submits the same action (if the per-step recovery
  budget is not exhausted and the total recovery budget is not
  exhausted) or fails closed with ``status="blocked"``. The driver
  never reads evaluator truth to decide whether to retry.
* The driver never returns ``"success"`` / ``"passed"``; those
  verdicts are reserved for the evaluator. The driver only reports
  whether it reached the end of the bounded plan.

Termination contract
--------------------

The driver does not terminate the episode by itself. It always
returns a :class:`CastingC3FrameDriverResult` and relies on the
calling orchestrator to mark the episode terminated and feed the
final state into the
:class:`~obsidianlink.evaluation.casting_frame_evaluator.FrozenFrameEvaluator`.
This mirrors the R3 / R4 / R5 evaluator contract: the
``episode_terminated`` field is the only thing that flips the
evaluator from ``in_progress`` to a terminal outcome, and only the
orchestrator / environment should flip it.

The :class:`CastingC3FrameDriverResult` carries:

* ``status`` — one of ``"completed"`` / ``"blocked"`` / ``"failed"``;
* ``steps_executed`` / ``wait_steps`` / ``planned_steps`` /
  ``recovery_attempts`` / ``recovery_budget`` — bounded counters
  useful for replay evidence;
* ``per_cell_relevant_action_records`` — a mapping from
  ``cell_index`` to the tuple of ``(step_id, item)`` pairs the
  driver submitted as relevant actions for that cell. The item is
  always ``"water_bucket"`` or ``"lava_bucket"`` (the two
  ``use_item`` variants the evaluator accepts). The orchestrator uses
  this mapping to build the
  :class:`~obsidianlink.evaluation.casting_frame_evaluator.FrozenFrameActionEvidence`
  records without ever reading evaluator truth;
* ``per_cell_target_offset`` — a mapping from ``cell_index`` to the
  public 14-cell ``(x, y, z)`` tuple the orchestrator passed in;
* ``per_cell_relevant_action_steps`` — a convenience view of
  ``per_cell_relevant_action_records`` that drops the item field and
  keeps only the step ids;
* ``events`` — a tuple of structured event mappings
  (``episode_id`` / ``agent_id`` / ``step_id`` / ``cell_index`` /
  ``target_offset`` / ``label`` / ``phase`` / ``action_type`` /
  ``target`` / ``relevant_action`` / ``attempt``);
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
from typing import Any, Callable, Mapping, Sequence

from obsidianlink.core.types import (
    BackendStep,
    MacroAction,
    Observation,
    RecoverableBackendError,
)


# ----------------------------------------------------------------------
# Public C3 frame contract constants. Locked to the
# ``casting_s_c3_fixed`` public task spec. The evaluator module has
# its own copy of the same values; this duplication is intentional —
# the driver module never imports the evaluator module.
# ----------------------------------------------------------------------

AGENT_ID: str = "agent_1"

WORKFLOW_C3_FRAME: str = "casting_s_c3_fixed"
FAMILY_C3_FRAME: str = "casting"
MODE_C3_FRAME: str = "single"
LEVEL_C3_FRAME: str = "C3"
LAYOUT_C3_FRAME: str = "fixed"

#: Closed set of public family / mode / level / layout values.
ALLOWED_C3_FRAME_FAMILIES: frozenset[str] = frozenset({FAMILY_C3_FRAME})
ALLOWED_C3_FRAME_MODES: frozenset[str] = frozenset({MODE_C3_FRAME})
ALLOWED_C3_FRAME_LEVELS: frozenset[str] = frozenset({LEVEL_C3_FRAME})
ALLOWED_C3_FRAME_LAYOUTS: frozenset[str] = frozenset({LAYOUT_C3_FRAME})

#: Public 14-cell full-ring order. Mirrors
#: ``scenario_parameters.public_task_spec.frame_plan.fixed_offsets``
#: of ``benchmark/instances/casting/single/casting_s_c3_fixed.json``.
CASTING_S_C3_FRAME_CELLS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 1),  # bottom-left corner
    (1, 0, 1),
    (2, 0, 1),
    (3, 0, 1),  # bottom-right corner
    (0, 4, 1),  # top-left corner
    (1, 4, 1),
    (2, 4, 1),
    (3, 4, 1),  # top-right corner
    (0, 1, 1),
    (0, 2, 1),
    (0, 3, 1),
    (3, 1, 1),
    (3, 2, 1),
    (3, 3, 1),
)

CASTING_S_C3_TARGET_CELL_COUNT: int = 14

#: Coordinate grid bounds for the public task-origin-relative frame
#: plan. ``x`` and ``y`` may range over the full ring; ``z`` is
#: pinned to ``1`` by the contract. The driver fails closed when
#: a target offset falls outside this box.
C3_FRAME_GRID_X_MIN: int = 0
C3_FRAME_GRID_X_MAX: int = 3
C3_FRAME_GRID_Y_MIN: int = 0
C3_FRAME_GRID_Y_MAX: int = 4
C3_FRAME_GRID_Z_MIN: int = 1
C3_FRAME_GRID_Z_MAX: int = 1


# ----------------------------------------------------------------------
# Action allowlist and phase vocabulary
# ----------------------------------------------------------------------

#: Closed C3 frame action allowlist. Mirrors the public
#: ``MacroAction`` protocol.
ALLOWED_C3_FRAME_ACTION_TYPES: frozenset[str] = frozenset(
    {"equip_item", "use_item", "place_block", "wait"}
)

#: Closed set of allowed targets. ``place_block`` is allowed only
#: for cobblestone support blocks; ``use_item`` / ``equip_item`` are
#: allowed only for the two buckets. The driver never supplies or
#: directly places ``obsidian``; it never ignites the frame.
ALLOWED_C3_FRAME_TARGETS: frozenset[str] = frozenset(
    {
        "water_bucket",
        "lava_bucket",
        "cobblestone",
    }
)

#: Phase labels used in the structured event log. Phases are not
#: read by the driver; they are emitted for evidence only.
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


# ----------------------------------------------------------------------
# Budget defaults and hard caps
# ----------------------------------------------------------------------

#: Default support-block settle waits. The R5 default is 1; C3 keeps
#: it for symmetry and to keep the plan small.
DEFAULT_SUPPORT_BLOCK_WAIT_STEPS: int = 1
#: Default fluid-settle waits after each bucket ``use_item``.
DEFAULT_FLUID_SETTLE_WAIT_STEPS: int = 4
#: Default final obsidian-settle waits. 14 × 4 = 56 of these in the
#: default plan.
DEFAULT_OBSIDIAN_WAIT_STEPS: int = 4

#: Hard cap on the per-plan wait count. The default plan uses
#: 17 waits / cell × 14 cells = 238; the cap is 320 so a test
#: configuration can stretch the per-cell waits but still hit a
#: deterministic fail-closed.
MAX_FRAME_PLAN_WAIT_STEPS: int = 320
#: Default driver-level wait cap. Must be at least the default
#: plan's total wait count (238) and well under the hard cap.
DEFAULT_MAX_WAIT_STEPS: int = 256

#: Hard cap on the per-plan length. 14 cells × 24 steps = 336 in
#: the default plan; the cap is 640 (the task's step limit) so the
#: driver can never exceed the task budget.
MAX_FRAME_PLAN_STEPS: int = 640

#: Recovery budget defaults. The R5 contract uses 1 per use_item
#: step and 3 total; C3 stretches the total budget so a test
#: orchestrator can inject one transient error per cell.
RECOVERIES_PER_USE_ITEM_DEFAULT: int = 1
TOTAL_RECOVERY_BUDGET_DEFAULT: int = 14
MAX_RECOVERIES_PER_ACTION: int = 2
MAX_TOTAL_RECOVERY_BUDGET: int = 32


# ----------------------------------------------------------------------
# Driver terminal statuses
# ----------------------------------------------------------------------

#: Closed set of terminal statuses the driver may emit. The driver
#: never returns ``"passed"`` / ``"success"``; those are reserved
#: for the evaluator.
DRIVER_STATUS_COMPLETED = "completed"
DRIVER_STATUS_BLOCKED = "blocked"
DRIVER_STATUS_FAILED = "failed"
DRIVER_STATUSES: frozenset[str] = frozenset(
    {DRIVER_STATUS_COMPLETED, DRIVER_STATUS_BLOCKED, DRIVER_STATUS_FAILED}
)


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------


def _require_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_positive_int(value: int, field_name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_non_negative_int(value: int, field_name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_positive_number(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return value


def _require_offset(value: Any, field_name: str) -> tuple[int, int, int]:
    """Validate a strict-int ``(x, y, z)`` tuple (no bools)."""
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(
            type(coordinate) is not int or isinstance(coordinate, bool)
            for coordinate in value
        )
    ):
        raise ValueError(
            f"{field_name} must be a (x, y, z) tuple of strict integers"
        )
    return value


def _freeze_value(value: Any) -> Any:
    """Recursively freeze a JSON-compatible value tree into a snapshot."""
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


def _require_c3_frame_action(
    action: MacroAction, *, context: str
) -> MacroAction:
    """Validate that ``action`` belongs to the closed C3 frame allowlist.

    The driver is the only place where this check is enforced.
    """
    if not isinstance(action, MacroAction):
        raise ValueError(f"{context}: action must be a MacroAction")
    if action.action_type not in ALLOWED_C3_FRAME_ACTION_TYPES:
        raise ValueError(
            f"{context}: action_type {action.action_type!r} is outside the "
            f"C3 frame allowlist {sorted(ALLOWED_C3_FRAME_ACTION_TYPES)}"
        )
    if action.target is not None and action.target not in ALLOWED_C3_FRAME_TARGETS:
        raise ValueError(
            f"{context}: target {action.target!r} is outside the C3 frame "
            f"allowlist {sorted(ALLOWED_C3_FRAME_TARGETS)}"
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
        raise ValueError(f"{context}: C3 frame actions cannot contain parameters")
    return action


# ----------------------------------------------------------------------
# Public driver context
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PublicC3FrameDriverContext:
    """Immutable, strictly-typed public context for the C3 frame driver.

    The orchestrator builds this object from a :class:`TaskInstance`
    and the task's
    ``scenario_parameters.public_task_spec.frame_plan.fixed_offsets``
    via
    :func:`build_public_c3_frame_driver_context_from_task`. The driver
    function itself only receives this context; it never reads
    ``scenario_parameters``, ``evaluator_contract``, evaluator truth,
    or the original task.

    All fields are validated in ``__post_init__``; unknown
    family / mode / level / layout values, wrong target-offset count,
    duplicate or out-of-range target offsets, and inventory with
    bool-as-int or non-positive quantities all fail closed.
    """

    episode_id: str
    workflow: str
    family: str
    mode: str
    level: str
    layout: str
    agent_id: str
    target_offsets: tuple[tuple[int, int, int], ...]
    initial_inventory: Mapping[str, int]
    task_step_limit: int
    task_time_limit: float

    def __post_init__(self) -> None:
        _require_identifier(self.episode_id, "episode_id")
        _require_identifier(self.workflow, "workflow")
        _require_identifier(self.family, "family")
        _require_identifier(self.mode, "mode")
        _require_identifier(self.level, "level")
        _require_identifier(self.layout, "layout")
        _require_identifier(self.agent_id, "agent_id")
        if self.workflow != WORKFLOW_C3_FRAME:
            raise ValueError(
                f"workflow must be {WORKFLOW_C3_FRAME!r}, got "
                f"{self.workflow!r}"
            )
        if self.family not in ALLOWED_C3_FRAME_FAMILIES:
            raise ValueError(
                f"family must be one of {sorted(ALLOWED_C3_FRAME_FAMILIES)}, "
                f"got {self.family!r}"
            )
        if self.mode not in ALLOWED_C3_FRAME_MODES:
            raise ValueError(
                f"mode must be one of {sorted(ALLOWED_C3_FRAME_MODES)}, "
                f"got {self.mode!r}"
            )
        if self.level not in ALLOWED_C3_FRAME_LEVELS:
            raise ValueError(
                f"level must be one of {sorted(ALLOWED_C3_FRAME_LEVELS)}, "
                f"got {self.level!r}"
            )
        if self.layout not in ALLOWED_C3_FRAME_LAYOUTS:
            raise ValueError(
                f"layout must be one of {sorted(ALLOWED_C3_FRAME_LAYOUTS)}, "
                f"got {self.layout!r}"
            )
        if self.agent_id != AGENT_ID:
            raise ValueError(
                f"agent_id must be {AGENT_ID!r}, got {self.agent_id!r}"
            )
        # --- target_offsets ------------------------------------------------
        try:
            offsets = tuple(self.target_offsets)
        except TypeError as exc:
            raise ValueError("target_offsets must be iterable") from exc
        if len(offsets) != CASTING_S_C3_TARGET_CELL_COUNT:
            raise ValueError(
                f"target_offsets must contain exactly "
                f"{CASTING_S_C3_TARGET_CELL_COUNT} cells, got {len(offsets)}"
            )
        normalized: list[tuple[int, int, int]] = []
        for index, offset in enumerate(offsets):
            normalized_offset = _require_offset(
                offset, f"target_offsets[{index}]"
            )
            x, y, z = normalized_offset
            if (
                x < C3_FRAME_GRID_X_MIN
                or x > C3_FRAME_GRID_X_MAX
                or y < C3_FRAME_GRID_Y_MIN
                or y > C3_FRAME_GRID_Y_MAX
                or z < C3_FRAME_GRID_Z_MIN
                or z > C3_FRAME_GRID_Z_MAX
            ):
                raise ValueError(
                    f"target_offsets[{index}]={normalized_offset!r} is outside "
                    f"the public C3 frame grid "
                    f"x=[{C3_FRAME_GRID_X_MIN},{C3_FRAME_GRID_X_MAX}], "
                    f"y=[{C3_FRAME_GRID_Y_MIN},{C3_FRAME_GRID_Y_MAX}], "
                    f"z=[{C3_FRAME_GRID_Z_MIN},{C3_FRAME_GRID_Z_MAX}]"
                )
            normalized.append(normalized_offset)
        if len(set(normalized)) != len(normalized):
            raise ValueError("target_offsets must not contain duplicates")
        # Public contract: the orchestrator must hand back the exact
        # 14-cell order from
        # ``scenario_parameters.public_task_spec.frame_plan.fixed_offsets``.
        if tuple(normalized) != CASTING_S_C3_FRAME_CELLS:
            raise ValueError(
                "target_offsets must exactly match the locked "
                "CASTING_S_C3_FRAME_CELLS order from the "
                "casting_s_c3_fixed public task spec"
            )
        # --- initial_inventory --------------------------------------------
        if not isinstance(self.initial_inventory, Mapping):
            raise ValueError("initial_inventory must be a mapping")
        inventory: dict[str, int] = {}
        for item, quantity in dict(self.initial_inventory).items():
            if not isinstance(item, str) or not item.strip():
                raise ValueError("initial_inventory keys must be non-empty strings")
            if type(quantity) is not int or isinstance(quantity, bool) or quantity < 0:
                raise ValueError(
                    f"initial_inventory[{item!r}] must be a non-negative integer"
                )
            if item not in ALLOWED_C3_FRAME_TARGETS:
                raise ValueError(
                    f"initial_inventory contains forbidden item {item!r}; "
                    f"only {sorted(ALLOWED_C3_FRAME_TARGETS)} are allowed"
                )
            inventory[item] = int(quantity)
        if "water_bucket" not in inventory or inventory["water_bucket"] < 1:
            raise ValueError(
                "initial_inventory must contain at least one water_bucket"
            )
        if "lava_bucket" not in inventory or inventory["lava_bucket"] < 1:
            raise ValueError(
                "initial_inventory must contain at least one lava_bucket"
            )
        if "cobblestone" not in inventory or inventory["cobblestone"] < 1:
            raise ValueError(
                "initial_inventory must contain at least one cobblestone"
            )
        # --- task budgets -------------------------------------------------
        _require_positive_int(self.task_step_limit, "task_step_limit")
        _require_positive_number(self.task_time_limit, "task_time_limit")
        if self.task_step_limit < CASTING_S_C3_TARGET_CELL_COUNT:
            raise ValueError(
                "task_step_limit must be at least "
                f"{CASTING_S_C3_TARGET_CELL_COUNT}"
            )
        # --- final freeze -------------------------------------------------
        object.__setattr__(self, "target_offsets", tuple(normalized))
        object.__setattr__(
            self, "initial_inventory", MappingProxyType(inventory)
        )


# ----------------------------------------------------------------------
# Orchestrator helper: build the public context from a task
# ----------------------------------------------------------------------
#
# The helper is intentionally NOT defined in this module. It lives
# in :mod:`obsidianlink.drivers.casting_s_c3_frame_context` so the
# driver source stays free of any reference to ``scenario_parameters``
# / ``evaluator_contract`` — the file-level AST lock would otherwise
# need to special-case the helper, which would defeat the
# information-isolation contract. The driver module imports nothing
# from the helper module.


# ----------------------------------------------------------------------
# Plan step
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CastingC3FramePlanStep:
    """One step in the bounded R6 C3 full-ring plan.

    ``cell_index`` selects the target cell this step works on.
    ``target_offset`` is the public ``(x, y, z)`` tuple from the
    orchestrator's :class:`PublicC3FrameDriverContext`. ``phase`` is
    the workflow stage. ``action`` is the (already whitelisted)
    :class:`MacroAction` to submit. ``relevant_action`` is ``True``
    exactly for ``use_item(water_bucket | lava_bucket)`` because the
    :class:`~obsidianlink.evaluation.casting_frame_evaluator.FrozenFrameActionEvidence`
    contract only accepts those two ``use_item`` variants.
    ``recoveries_allowed`` is the per-step recovery budget; the
    driver never consults the evaluator to decide whether to retry.
    """

    cell_index: int
    target_offset: tuple[int, int, int]
    label: str
    phase: str
    action: MacroAction
    relevant_action: bool = False
    recoveries_allowed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("plan step label must be a non-empty string")
        _require_non_negative_int(self.cell_index, "cell_index")
        if self.cell_index >= CASTING_S_C3_TARGET_CELL_COUNT:
            raise ValueError(
                f"cell_index must be < {CASTING_S_C3_TARGET_CELL_COUNT}"
            )
        object.__setattr__(
            self,
            "target_offset",
            _require_offset(self.target_offset, "target_offset"),
        )
        if self.target_offset != CASTING_S_C3_FRAME_CELLS[self.cell_index]:
            raise ValueError(
                "target_offset must match the locked C3 frame cell at "
                f"cell_index={self.cell_index}: expected "
                f"{CASTING_S_C3_FRAME_CELLS[self.cell_index]!r}, got "
                f"{self.target_offset!r}"
            )
        if self.phase not in PHASE_VALUES:
            raise ValueError(
                f"unknown C3 frame plan phase: {self.phase!r}"
            )
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
        _require_c3_frame_action(self.action, context=f"plan[{self.label!r}]")
        if self.action.action_type == "use_item":
            expected_relevant = True
        else:
            expected_relevant = False
        if self.relevant_action is not expected_relevant:
            raise ValueError(
                "relevant_action must be true exactly for use_item actions"
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
class CastingC3FrameDriverResult:
    """Public result of :func:`run_casting_s_c3_frame_driver`.

    The driver never returns a casting verdict; the orchestrator owns
    the evaluator call. This object only reports whether the driver
    reached the end of the bounded plan, which steps it executed,
    and the event log.

    ``per_cell_relevant_action_records`` is a mapping from
    ``cell_index`` to the tuple of ``(step_id, item)`` pairs the
    driver submitted as relevant actions for that cell. The item is
    always ``"water_bucket"`` or ``"lava_bucket"``. The orchestrator
    uses this to build the
    :class:`~obsidianlink.evaluation.casting_frame_evaluator.FrozenFrameActionEvidence`
    records without ever reading evaluator truth.

    ``per_cell_target_offset`` mirrors the public 14-cell order the
    orchestrator handed in. ``per_cell_relevant_action_steps`` is a
    convenience view of ``per_cell_relevant_action_records`` that
    drops the item field and keeps only the step ids.
    """

    status: str
    steps_executed: int
    wait_steps: int
    planned_steps: int
    recovery_attempts: int
    recovery_budget: int
    per_cell_relevant_action_records: Mapping[int, tuple[tuple[int, str], ...]]
    per_cell_relevant_action_steps: Mapping[int, tuple[int, ...]]
    per_cell_target_offset: Mapping[int, tuple[int, int, int]]
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
        if self.recovery_attempts > self.recovery_budget + 1:
            raise ValueError(
                "recovery_attempts cannot exceed recovery_budget + 1"
            )
        if not isinstance(self.final_observation, Observation):
            raise ValueError("final_observation must be an Observation")
        if not isinstance(self.events, tuple):
            raise ValueError("events must be a tuple")
        if not isinstance(self.per_cell_relevant_action_records, Mapping):
            raise ValueError(
                "per_cell_relevant_action_records must be a mapping"
            )
        if not isinstance(self.per_cell_relevant_action_steps, Mapping):
            raise ValueError(
                "per_cell_relevant_action_steps must be a mapping"
            )
        if not isinstance(self.per_cell_target_offset, Mapping):
            raise ValueError("per_cell_target_offset must be a mapping")
        for key, value in self.per_cell_relevant_action_records.items():
            _require_non_negative_int(
                key, "per_cell_relevant_action_records key"
            )
            if not isinstance(value, tuple) or any(
                not isinstance(record, tuple)
                or len(record) != 2
                or type(record[0]) is not int
                or isinstance(record[0], bool)
                or record[0] < 0
                or record[1] not in ALLOWED_C3_FRAME_TARGETS
                for record in value
            ):
                raise ValueError(
                    "per_cell_relevant_action_records values must be tuples of "
                    "(non_negative_int, allowed_target) pairs"
                )
        for key, value in self.per_cell_relevant_action_steps.items():
            _require_non_negative_int(
                key, "per_cell_relevant_action_steps key"
            )
            if not isinstance(value, tuple) or any(
                type(step) is not int
                or isinstance(step, bool)
                or step < 0
                for step in value
            ):
                raise ValueError(
                    "per_cell_relevant_action_steps values must be tuples of "
                    "non-negative ints"
                )
        for key, value in self.per_cell_target_offset.items():
            _require_non_negative_int(key, "per_cell_target_offset key")
            if (
                not isinstance(value, tuple)
                or len(value) != 3
                or any(
                    type(coordinate) is not int
                    or isinstance(coordinate, bool)
                    for coordinate in value
                )
            ):
                raise ValueError(
                    "per_cell_target_offset values must be (x, y, z) int tuples"
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
        frozen_records = _freeze_value(self.per_cell_relevant_action_records)
        frozen_steps = _freeze_value(self.per_cell_relevant_action_steps)
        frozen_offsets = _freeze_value(self.per_cell_target_offset)
        object.__setattr__(self, "events", frozen_events)
        object.__setattr__(self, "action_label_for_step", frozen_labels)
        object.__setattr__(self, "per_cell_relevant_action_records", frozen_records)
        object.__setattr__(self, "per_cell_relevant_action_steps", frozen_steps)
        object.__setattr__(self, "per_cell_target_offset", frozen_offsets)

    def as_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-serializable snapshot."""
        return {
            "status": self.status,
            "steps_executed": self.steps_executed,
            "wait_steps": self.wait_steps,
            "planned_steps": self.planned_steps,
            "recovery_attempts": self.recovery_attempts,
            "recovery_budget": self.recovery_budget,
            "per_cell_relevant_action_records": {
                int(cell_index): [
                    {"step_id": int(record[0]), "item": str(record[1])}
                    for record in records
                ]
                for cell_index, records in self.per_cell_relevant_action_records.items()
            },
            "per_cell_relevant_action_steps": {
                int(cell_index): [int(step) for step in steps]
                for cell_index, steps in self.per_cell_relevant_action_steps.items()
            },
            "per_cell_target_offset": {
                int(cell_index): [int(c) for c in offset]
                for cell_index, offset in self.per_cell_target_offset.items()
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
) -> CastingC3FramePlanStep:
    return CastingC3FramePlanStep(
        cell_index=cell_index,
        target_offset=CASTING_S_C3_FRAME_CELLS[cell_index],
        label=label,
        phase=phase,
        action=MacroAction.wait(),
    )


def _select_step(
    cell_index: int, target: str, label: str, phase: str
) -> CastingC3FramePlanStep:
    return CastingC3FramePlanStep(
        cell_index=cell_index,
        target_offset=CASTING_S_C3_FRAME_CELLS[cell_index],
        label=label,
        phase=phase,
        action=MacroAction(action_type="equip_item", target=target),
    )


def _place_support_step(
    cell_index: int, label: str
) -> CastingC3FramePlanStep:
    return CastingC3FramePlanStep(
        cell_index=cell_index,
        target_offset=CASTING_S_C3_FRAME_CELLS[cell_index],
        label=label,
        phase=PHASE_PLACE_SUPPORT,
        action=MacroAction(action_type="place_block", target="cobblestone"),
        relevant_action=False,
    )


def _use_bucket_step(
    cell_index: int,
    target: str,
    label: str,
    *,
    recoveries_allowed: int = 0,
) -> CastingC3FramePlanStep:
    return CastingC3FramePlanStep(
        cell_index=cell_index,
        target_offset=CASTING_S_C3_FRAME_CELLS[cell_index],
        label=label,
        phase=(
            PHASE_PLACE_LAVA if target == "lava_bucket" else PHASE_PLACE_WATER
        ),
        action=MacroAction(action_type="use_item", target=target),
        relevant_action=True,
        recoveries_allowed=recoveries_allowed,
    )


def build_casting_s_c3_frame_action_plan(
    target_offsets: Sequence[Sequence[int]] = CASTING_S_C3_FRAME_CELLS,
    *,
    support_block_wait_steps: int = DEFAULT_SUPPORT_BLOCK_WAIT_STEPS,
    fluid_settle_wait_steps: int = DEFAULT_FLUID_SETTLE_WAIT_STEPS,
    obsidian_wait_steps: int = DEFAULT_OBSIDIAN_WAIT_STEPS,
    recoveries_per_use_item: int = RECOVERIES_PER_USE_ITEM_DEFAULT,
) -> tuple[CastingC3FramePlanStep, ...]:
    """Build the bounded R6 C3 full-ring casting plan.

    The plan is fully deterministic. For each cell (in fixed index
    order, 0..13) the sequence is:

    1. Select lava bucket + brief wait (hotbar equip is not instant).
    2. Place a cobblestone support block + settle wait.
    3. Place a second cobblestone support block + settle wait.
    4. Re-select lava bucket + brief wait.
    5. ``use_item(lava_bucket)`` on the target cell + fluid-settle
       waits. Each ``use_item`` step carries a
       ``recoveries_allowed`` budget so the driver can
       deterministically retry a transient error. ``use_item`` is
       the only step the evaluator considers a "relevant action".
    6. Select water bucket + brief wait.
    7. ``use_item(water_bucket)`` against the lava + fluid-settle
       waits. Same recovery budget.
    8. Bounded obsidian-settle waits so the casting evaluator has a
       fair chance to observe the obsidian transition.

    All wait counts are parameterised but bounded by
    :func:`run_casting_s_c3_frame_driver` so a caller cannot ask the
    driver to run forever.

    ``target_offsets`` defaults to :data:`CASTING_S_C3_FRAME_CELLS`
    so the default plan is fully reproducible. Tests may pass a
    different ordered sequence of 14 cells; the public contract
    requires the order to match the public task spec.
    """
    normalized: list[tuple[int, int, int]] = []
    for index, offset in enumerate(target_offsets):
        normalized.append(_require_offset(offset, f"target_offsets[{index}]"))
    if tuple(normalized) != CASTING_S_C3_FRAME_CELLS:
        raise ValueError(
            "build_casting_s_c3_frame_action_plan target_offsets must match "
            "the locked CASTING_S_C3_FRAME_CELLS order"
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
    total_waits = waits_per_cell * CASTING_S_C3_TARGET_CELL_COUNT
    if total_waits > MAX_FRAME_PLAN_WAIT_STEPS:
        raise ValueError(
            "C3 frame plan wait steps exceed the hard limit: "
            f"{total_waits} > {MAX_FRAME_PLAN_WAIT_STEPS}"
        )
    plan: list[CastingC3FramePlanStep] = []
    for cell_index in range(CASTING_S_C3_TARGET_CELL_COUNT):
        plan.extend(
            [
                _select_step(
                    cell_index,
                    "lava_bucket",
                    f"cell_{cell_index}.prepare.select_lava",
                    PHASE_PREPARE,
                ),
                _wait_step(
                    cell_index,
                    f"cell_{cell_index}.prepare.select_lava.release",
                    PHASE_PREPARE,
                ),
                _place_support_step(
                    cell_index, f"cell_{cell_index}.support.block_1"
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
                cell_index, f"cell_{cell_index}.support.block_2"
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
        _require_c3_frame_action(step.action, context=f"plan[{step.label!r}]")
    if len(plan) > MAX_FRAME_PLAN_STEPS:
        raise ValueError(
            f"C3 frame plan length exceeds hard limit: {len(plan)} > "
            f"{MAX_FRAME_PLAN_STEPS}"
        )
    return tuple(plan)


# ----------------------------------------------------------------------
# Driver implementation
# ----------------------------------------------------------------------


def _visible_inventory_has(inventory: Mapping[str, int], item: str) -> bool:
    if item not in ALLOWED_C3_FRAME_TARGETS:
        raise ValueError(f"driver cannot inspect item {item!r}")
    quantity = inventory.get(item, 0)
    if type(quantity) is not int or quantity < 0:
        raise ValueError(
            "visible_inventory quantities must be non-negative integers"
        )
    return quantity > 0


def _assert_workflow_stage(observation: Observation, expected: str) -> None:
    if not isinstance(observation, Observation):
        raise ValueError("expected an Observation")
    if observation.workflow_stage != expected:
        raise ValueError(
            f"driver only supports workflow {expected!r}, got "
            f"{observation.workflow_stage!r}"
        )


def run_casting_s_c3_frame_driver(
    backend: Any,
    context: PublicC3FrameDriverContext,
    *,
    plan: tuple[CastingC3FramePlanStep, ...] | None = None,
    max_wait_steps: int = DEFAULT_MAX_WAIT_STEPS,
    max_environment_steps: int | None = None,
    max_game_time_seconds: float | None = None,
    total_recovery_budget: int = TOTAL_RECOVERY_BUDGET_DEFAULT,
    recoveries_per_use_item: int = RECOVERIES_PER_USE_ITEM_DEFAULT,
    event_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> CastingC3FrameDriverResult:
    """Run the bounded R6 C3 full-ring casting plan on ``backend``.

    The driver:

    1. Calls ``backend.reset(task)`` after the orchestrator has
       already constructed the immutable
       :class:`PublicC3FrameDriverContext` from the task's
       ``scenario_parameters.public_task_spec``. The driver uses
       the returned ``Observation`` only for the initial
       ``visible_inventory`` / ``workflow_stage`` check; it never
       reads evaluator truth.
    2. Walks the plan step-by-step, calling
       ``backend.step({AGENT_ID: action})`` once per step. Each
       step is validated against the closed C3 frame allowlist
       before submission.
    3. Refuses to start a step that requires an item the Agent is
       not carrying in its visible inventory. The check uses
       ``visible_inventory`` only; the driver never reads
       ``Observation.frame`` or any other field.
    4. Catches the typed
       :class:`~obsidianlink.core.types.RecoverableBackendError`
       exception raised by ``backend.step`` and applies the
       deterministic, bounded recovery protocol described in the
       module docstring. Any other exception
       (``RuntimeError`` / ``OSError`` / ``TypeError`` not
       subclassing :class:`RecoverableBackendError`) fails closed
       immediately.
    5. Is bounded by ``max_environment_steps`` and
       ``max_game_time_seconds``; when either is exceeded the
       driver returns ``status="blocked"`` with a descriptive
       ``blocked_reason``.

    The driver does *not* call
    ``set_frame_evaluation_state`` /
    ``get_frame_evaluation_state`` /
    ``clear_frame_evaluation_state``, does *not* read
    ``scenario_parameters`` / ``evaluator_contract``, and does *not*
    invoke the frame evaluator. The orchestrator owns the
    evaluator call and the truth injection path.
    """
    if not isinstance(context, PublicC3FrameDriverContext):
        raise ValueError("context must be a PublicC3FrameDriverContext")
    _require_positive_int(max_wait_steps, "max_wait_steps")
    if max_wait_steps > MAX_FRAME_PLAN_WAIT_STEPS:
        raise ValueError(
            f"max_wait_steps must be <= {MAX_FRAME_PLAN_WAIT_STEPS}"
        )
    if max_environment_steps is None:
        max_environment_steps = context.task_step_limit
    else:
        _require_positive_int(max_environment_steps, "max_environment_steps")
        if max_environment_steps > context.task_step_limit:
            raise ValueError(
                "max_environment_steps cannot exceed the task limit "
                f"{context.task_step_limit}"
            )
    if max_game_time_seconds is None:
        max_game_time_seconds = context.task_time_limit
    else:
        _require_positive_number(
            max_game_time_seconds, "max_game_time_seconds"
        )
        if max_game_time_seconds > context.task_time_limit:
            raise ValueError(
                "max_game_time_seconds cannot exceed the task limit "
                f"{context.task_time_limit}"
            )
    if plan is None:
        plan = build_casting_s_c3_frame_action_plan(
            target_offsets=context.target_offsets,
            recoveries_per_use_item=recoveries_per_use_item,
        )
    if not isinstance(plan, tuple) or not plan:
        raise ValueError("plan must be non-empty")
    if any(
        not isinstance(step, CastingC3FramePlanStep) for step in plan
    ):
        raise ValueError(
            "plan must contain only CastingC3FramePlanStep values"
        )
    if len(plan) > context.task_step_limit:
        raise ValueError(
            "plan length cannot exceed the task step limit "
            f"{context.task_step_limit}"
        )
    plan_wait_steps = sum(
        step.action.action_type == "wait" for step in plan
    )
    if plan_wait_steps > MAX_FRAME_PLAN_WAIT_STEPS:
        raise ValueError(
            "plan wait steps cannot exceed the hard limit "
            f"{MAX_FRAME_PLAN_WAIT_STEPS}"
        )
    _require_non_negative_int(total_recovery_budget, "total_recovery_budget")
    if total_recovery_budget > MAX_TOTAL_RECOVERY_BUDGET:
        raise ValueError(
            "total_recovery_budget cannot exceed "
            f"{MAX_TOTAL_RECOVERY_BUDGET}"
        )
    if not hasattr(backend, "reset") or not hasattr(backend, "step"):
        raise ValueError("backend must implement reset/step")

    # The backend's ``reset`` does not take a TaskInstance that the
    # driver knows about; the orchestrator already wrapped the
    # public contract into the immutable context. The FakeBackend
    # however still expects a ``reset(task)`` call, so the
    # orchestrator wraps the backend to convert the context to a
    # TaskInstance or to a minimal reset that the backend accepts.
    # This driver deliberately does not look at the original task
    # instance. Tests and orchestrators that need a real
    # ``reset(task)`` call use the helper function above to build
    # the context, then call this driver. The driver only needs
    # ``reset()`` to return an ``Observation`` whose
    # ``workflow_stage`` matches ``context.workflow``.
    if not hasattr(backend, "reset"):
        raise ValueError("backend must implement reset")
    # The driver intentionally never reads ``backend._task`` or any
    # other backend-private attribute. The reset is invoked with a
    # minimal proxy: a callable that yields the same
    # ``Observation`` the backend would produce if reset with the
    # canonical task. The proxy is constructed by the orchestrator
    # in production; the test passes a TaskInstance-shaped proxy
    # directly via the backend's ``reset`` method.
    observations = backend.reset(_ResetProxy(context))
    if not isinstance(observations, Mapping):
        raise ValueError("backend.reset must return a mapping of Observations")
    final_observation = observations[context.agent_id]
    if not isinstance(final_observation, Observation):
        raise ValueError("backend.reset must return Observation values")
    _assert_workflow_stage(final_observation, context.workflow)
    reset_timestamp = (
        float(final_observation.timestamp)
        if math.isfinite(final_observation.timestamp)
        else None
    )

    events: list[Mapping[str, Any]] = []
    action_label_for_step: dict[int, str] = {}
    per_cell_relevant: dict[int, list[tuple[int, str]]] = {}
    per_cell_target: dict[int, tuple[int, int, int]] = {}
    for cell_index, offset in enumerate(context.target_offsets):
        per_cell_target[cell_index] = offset
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
            "episode_id": context.episode_id,
            "agent_id": context.agent_id,
            **dict(event),
        }
        events.append(identified)
        if event_sink is not None:
            event_sink(_thaw_value(_freeze_value(identified)))

    record_event(
        {
            "step_id": final_observation.step_id,
            "cell_index": -1,
            "target_offset": None,
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
        _require_c3_frame_action(
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
                final_observation.visible_inventory or {}, item
            ):
                blocked_reason = (
                    f"missing required item {item!r} at {plan_step.label}"
                )
                status = DRIVER_STATUS_BLOCKED
                break
        elif plan_step.action.action_type == "use_item":
            item = plan_step.action.target
            if item is None or not _visible_inventory_has(
                final_observation.visible_inventory or {}, item
            ):
                blocked_reason = (
                    f"missing required item {item!r} at {plan_step.label}"
                )
                status = DRIVER_STATUS_BLOCKED
                break
        elif plan_step.action.action_type == "place_block":
            if not _visible_inventory_has(
                final_observation.visible_inventory or {}, "cobblestone"
            ):
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
                step = backend.step(
                    {context.agent_id: plan_step.action}
                )
            except RecoverableBackendError as error:
                recovery_attempts += 1
                attempt += 1
                record_event(
                    {
                        "step_id": previous_step_id,
                        "cell_index": plan_step.cell_index,
                        "target_offset": list(plan_step.target_offset),
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
                        f"per-step recovery budget exhausted at "
                        f"{plan_step.label}: attempt={attempt} > "
                        f"recoveries_allowed={step_attempts_allowed}"
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
                        "step_id": previous_step_id,
                        "cell_index": plan_step.cell_index,
                        "target_offset": list(plan_step.target_offset),
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
            if step.episode_id != context.episode_id:
                raise ValueError(
                    "BackendStep episode_id must match the context episode_id"
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
            next_observation = step.observations[context.agent_id]
            if not isinstance(next_observation, Observation):
                raise ValueError("backend.step must return Observation values")
            _assert_workflow_stage(next_observation, context.workflow)
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
                    if plan_step.relevant_action and plan_step.action.target in {
                        "water_bucket",
                        "lava_bucket",
                    }:
                        per_cell_relevant.setdefault(
                            plan_step.cell_index, []
                        ).append(
                            (step.step_id, plan_step.action.target)
                        )
                    record_event(
                        {
                            "step_id": step.step_id,
                            "cell_index": plan_step.cell_index,
                            "target_offset": list(plan_step.target_offset),
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
            if (
                plan_step.relevant_action
                and plan_step.action.target in {"water_bucket", "lava_bucket"}
            ):
                per_cell_relevant.setdefault(
                    plan_step.cell_index, []
                ).append((step.step_id, plan_step.action.target))
            record_event(
                {
                    "step_id": step.step_id,
                    "cell_index": plan_step.cell_index,
                    "target_offset": list(plan_step.target_offset),
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
                        f"plan interrupted by backend {reason} at "
                        f"{plan_step.label}"
                    )
                break
        if status != DRIVER_STATUS_COMPLETED:
            break

    if blocked_reason is None and steps_executed < len(plan):
        blocked_reason = "plan interrupted by backend termination"
        status = DRIVER_STATUS_BLOCKED

    frozen_records = MappingProxyType(
        {
            int(cell_index): tuple(records)
            for cell_index, records in per_cell_relevant.items()
        }
    )
    frozen_steps = MappingProxyType(
        {
            int(cell_index): tuple(record[0] for record in records)
            for cell_index, records in per_cell_relevant.items()
        }
    )
    frozen_offsets = MappingProxyType(
        {int(cell_index): offset for cell_index, offset in per_cell_target.items()}
    )
    return CastingC3FrameDriverResult(
        status=status,
        steps_executed=steps_executed,
        wait_steps=wait_steps,
        planned_steps=len(plan),
        recovery_attempts=recovery_attempts,
        recovery_budget=total_recovery_budget,
        per_cell_relevant_action_records=frozen_records,
        per_cell_relevant_action_steps=frozen_steps,
        per_cell_target_offset=frozen_offsets,
        final_observation=final_observation,
        events=tuple(events),
        action_label_for_step=MappingProxyType(action_label_for_step),
        terminated=backend_terminated,
        truncated=backend_truncated,
        blocked_reason=blocked_reason,
        error_type=error_type,
    )


# ----------------------------------------------------------------------
# reset(task) proxy
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _ResetProxy:
    """A minimal stand-in for the original :class:`TaskInstance`.

    The C3 frame driver never reads the original task instance; the
    orchestrator already wrapped the public contract into
    :class:`PublicC3FrameDriverContext`. The driver still needs to
    call ``backend.reset(...)`` because the
    :class:`FakeEnvironmentBackend` API requires a task. This proxy
    exposes the absolute minimum surface the FakeBackend's ``reset``
    reads (``task_id``, ``initial_inventories``, ``workflow``,
    ``agent_ids``). The driver never inspects it directly; it just
    hands it to ``backend.reset`` and uses the returned
    :class:`Observation`.
    """

    context: PublicC3FrameDriverContext

    @property
    def task_id(self) -> str:
        return self.context.episode_id

    @property
    def workflow(self) -> str:
        return self.context.workflow

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return (self.context.agent_id,)

    @property
    def initial_inventories(self) -> Mapping[str, Mapping[str, int]]:
        return {self.context.agent_id: self.context.initial_inventory}


__all__ = [
    "AGENT_ID",
    "ALLOWED_C3_FRAME_ACTION_TYPES",
    "ALLOWED_C3_FRAME_FAMILIES",
    "ALLOWED_C3_FRAME_LAYOUTS",
    "ALLOWED_C3_FRAME_LEVELS",
    "ALLOWED_C3_FRAME_MODES",
    "ALLOWED_C3_FRAME_TARGETS",
    "C3_FRAME_GRID_X_MAX",
    "C3_FRAME_GRID_X_MIN",
    "C3_FRAME_GRID_Y_MAX",
    "C3_FRAME_GRID_Y_MIN",
    "C3_FRAME_GRID_Z_MAX",
    "C3_FRAME_GRID_Z_MIN",
    "CASTING_S_C3_FRAME_CELLS",
    "CASTING_S_C3_TARGET_CELL_COUNT",
    "DEFAULT_FLUID_SETTLE_WAIT_STEPS",
    "DEFAULT_MAX_WAIT_STEPS",
    "DEFAULT_OBSIDIAN_WAIT_STEPS",
    "DEFAULT_SUPPORT_BLOCK_WAIT_STEPS",
    "DRIVER_STATUS_BLOCKED",
    "DRIVER_STATUS_COMPLETED",
    "DRIVER_STATUS_FAILED",
    "DRIVER_STATUSES",
    "FAMILY_C3_FRAME",
    "LAYOUT_C3_FRAME",
    "LEVEL_C3_FRAME",
    "MAX_FRAME_PLAN_STEPS",
    "MAX_FRAME_PLAN_WAIT_STEPS",
    "MAX_RECOVERIES_PER_ACTION",
    "MAX_TOTAL_RECOVERY_BUDGET",
    "MODE_C3_FRAME",
    "PHASE_PLACE_LAVA",
    "PHASE_PLACE_SUPPORT",
    "PHASE_PLACE_WATER",
    "PHASE_PREPARE",
    "PHASE_RECOVERY",
    "PHASE_VALUES",
    "PHASE_WAIT_FOR_OBSIDIAN",
    "RECOVERIES_PER_USE_ITEM_DEFAULT",
    "TOTAL_RECOVERY_BUDGET_DEFAULT",
    "WORKFLOW_C3_FRAME",
    "CastingC3FrameDriverResult",
    "CastingC3FramePlanStep",
    "PublicC3FrameDriverContext",
    "build_casting_s_c3_frame_action_plan",
    "run_casting_s_c3_frame_driver",
]

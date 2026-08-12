"""R4 deterministic single-block casting driver.

This module contains the :func:`run_casting_c1_driver` entry point
and the :func:`build_casting_action_plan` plan builder used by the
:mod:`tests.test_casting_driver` test suite to drive the
``casting_c1_fixed`` task on the :class:`FakeEnvironmentBackend`.

Design contract
---------------

The driver is the *public* deterministic single-block policy for
R4. It obeys the same constraints as the legacy Route A0
``run_scripted_a0``:

* The driver only consumes Agent-visible
  :class:`~obsidianlink.core.types.Observation` data
  (``visible_inventory`` / ``workflow_stage`` / ``step_id``). It
  never reads casting truth, target-cell truth, fluid truth, or
  any evaluator-only field.
* The driver never calls
  :meth:`FakeEnvironmentBackend.set_casting_evaluation_state` or
  :meth:`FakeEnvironmentBackend.get_casting_evaluation_state`.
  Truth injection and the :class:`CastingEvaluator` call live in
  the test orchestrator (``tests/test_casting_driver.py``); the
  driver surface has no access to those methods.
* The driver only emits :class:`MacroAction` values from the
  project's public action protocol:
  ``equip_item`` / ``use_item`` / ``place_block`` / ``wait`` /
  ``look`` / ``move``. The driver's
  :func:`build_casting_action_plan` is the single source of truth
  for the action sequence; no other action sequence is allowed at
  runtime.
* Every wait and total-step budget has a hard, type-explicit
  cap (``max_wait_steps`` / ``max_environment_steps`` /
  ``max_game_time_seconds``). The sole ``for`` traversal is bounded
  by the validated plan length and by the backend's step counter.
* The driver never executes model-generated code, never opens a
  shell, and never imports the VLM surface. It is safe to call
  from a unit test, an offline replay, or (once authorised) a real
  MineRL backend.

Termination contract
--------------------

The driver does *not* terminate the episode by itself. It always
returns a :class:`CastingC1DriverResult` and relies on the calling
orchestrator to mark the episode terminated and feed the final
state into the casting evaluator. This mirrors the R3 evaluator
contract: ``CastingEvaluationState.episode_terminated`` is the
*only* field that flips the evaluator from ``in_progress`` to a
terminal outcome, and only the orchestrator / environment should
flip it.

The :class:`CastingC1DriverResult` carries:

* ``status`` — one of ``"completed"`` / ``"blocked"`` / ``"failed"``;
* ``steps_executed`` / ``wait_steps`` / ``planned_steps`` — bounded
  counters useful for replay evidence;
* ``events`` — a tuple of structured event mappings
  (``episode_id`` / ``agent_id`` / ``step_id`` / ``label`` /
  ``phase`` / ``action_type`` / ``target`` / ``relevant_action``).
  A label is marked ``relevant_action=True`` when it represents a
  semantic fluid action (use_item on a bucket, or place_block on
  cobblestone) so the orchestrator can build
  ``relevant_action_steps`` from the event log without reading
  evaluator truth.
* ``action_label_for_step`` — a mapping from ``step_id`` to the
  final semantic label produced at that step.
* ``relevant_action_steps`` — the tuple of step ids at which a
  relevant action was submitted.
* ``terminated`` / ``truncated`` — the final flags reported by the
  backend. The driver never fabricates either flag.
* ``final_observation`` — the most recent Observation the driver
  received (used by the orchestrator for evidence).
* ``as_dict`` — returns a detached snapshot for evidence logging.

The driver never returns ``status == "passed"`` / ``"success"`` /
``"failed_casting"``. The driver reports whether it *reached* the
end of the bounded plan; the orchestrator owns the evaluator
verdict.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from obsidianlink.core.types import BackendStep, MacroAction, Observation, TaskInstance


AGENT_ID = "agent_1"


# Closed R4 action allowlist. Every step the driver emits must be
# one of these semantic action types. ``place_block`` is only used
# for the cobblestone support block; the casting fluid is placed
# exclusively through ``use_item`` with ``water_bucket`` /
# ``lava_bucket`` as the target.
ALLOWED_R4_ACTION_TYPES: frozenset[str] = frozenset(
    {"equip_item", "use_item", "place_block", "wait", "look", "move"}
)


# Targets the driver is allowed to use. ``place_block`` is allowed
# only for the cobblestone support block; ``use_item`` is allowed
# only for the two buckets. ``equip_item`` may select either bucket
# or cobblestone before placement.
ALLOWED_R4_TARGETS: frozenset[str] = frozenset(
    {
        "water_bucket",
        "lava_bucket",
        "cobblestone",
    }
)


# Phase labels used in the structured event log. Phases are not
# read by the driver; they are emitted for evidence only.
PHASE_PREPARE = "prepare"
PHASE_AIM = "aim"
PHASE_PLACE_SUPPORT = "place_support"
PHASE_PLACE_LAVA = "place_lava"
PHASE_PLACE_WATER = "place_water"
PHASE_WAIT_FOR_OBSIDIAN = "wait_for_obsidian"

# Frozen public interaction geometry for target cell ``[2, 4, 3]`` from
# spawn ``[0, 4, 0]``.  Camera actions are *relative deltas*, matching
# MineRL's HumanLevelCommands transport.  The click points are the centres
# of solid top faces at y=4: two nearby support cells, the lava target, and
# the adjacent water source whose flow reaches the lava target.
FROZEN_PLAYER_EYE: tuple[float, float, float] = (0.5, 5.62, 0.5)
FROZEN_SUPPORT_FACE_POINTS: tuple[tuple[float, float, float], ...] = (
    (1.5, 4.0, 3.5),
    # This face exists only after support.block_1 succeeds. Building upward
    # avoids the real-scene ray interception observed when aiming at a second
    # grass face behind/beside the first placed block.
    (1.5, 5.0, 3.5),
)
FROZEN_LAVA_FACE_POINT: tuple[float, float, float] = (2.5, 4.0, 3.5)
FROZEN_WATER_FACE_POINT: tuple[float, float, float] = (2.5, 4.0, 2.5)
LOOK_PARAMETER_ABS_MAX: float = 30.0
MOVE_FORWARD_ABS_MAX: float = 1.0


# Defaults for the bounded plan. Each value is a *hard* upper
# bound; smaller values are accepted, larger values are rejected.
# The default ``max_wait_steps`` is intentionally larger than
# the plan's total wait count so the default plan fits without a
# hard cap, but the cap still fires if a caller injects an
# unbounded loop.
DEFAULT_MAX_WAIT_STEPS: int = 40
MAX_PLAN_WAIT_STEPS: int = DEFAULT_MAX_WAIT_STEPS
DEFAULT_SUPPORT_BLOCK_WAIT_STEPS: int = 4
DEFAULT_FLUID_SETTLE_WAIT_STEPS: int = 4
DEFAULT_OBSIDIAN_WAIT_STEPS: int = 4
# A relevant action may be acknowledged late by MineRL, but the driver must
# never wait without a fixed bound.  Only consecutive, already-planned no-op
# ticks may be used for confirmation; any intervening action fails closed.
MAX_INVENTORY_CONFIRMATION_WAIT_STEPS: int = 4


# Step / time budget defaults match the ``casting_c1_fixed`` task
# contract. The driver consults them to decide when to give up
# waiting even before the environment reports termination.
DEFAULT_MAX_ENVIRONMENT_STEPS: int = 160
DEFAULT_MAX_GAME_TIME_SECONDS: float = 120.0


# Terminal driver statuses. The driver never returns
# ``"passed"`` / ``"success"``; those are reserved for the
# evaluator. ``"completed"`` means the plan finished without a
# hard-budget stop, ``"blocked"`` means a budget cap fired,
# ``"failed"`` means the backend raised.
DRIVER_STATUS_COMPLETED = "completed"
DRIVER_STATUS_BLOCKED = "blocked"
DRIVER_STATUS_FAILED = "failed"
DRIVER_STATUSES: frozenset[str] = frozenset(
    {DRIVER_STATUS_COMPLETED, DRIVER_STATUS_BLOCKED, DRIVER_STATUS_FAILED}
)


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
    """Recursively freeze the JSON-compatible evidence tree."""
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
    """Return a detached JSON-serializable evidence tree."""
    if isinstance(value, Mapping):
        return {key: _thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return value


def _require_look_parameters(parameters: Mapping[str, Any], *, context: str) -> None:
    allowed = {"yaw", "pitch"}
    unknown = set(parameters) - allowed
    if unknown:
        raise ValueError(
            f"{context}: look parameters may only include yaw/pitch, "
            f"got {sorted(unknown)}"
        )
    if "yaw" not in parameters or "pitch" not in parameters:
        raise ValueError(f"{context}: look requires yaw and pitch")
    for name in ("yaw", "pitch"):
        value = parameters[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{context}: look {name} must be a finite number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{context}: look {name} must be finite")
        if abs(number) > LOOK_PARAMETER_ABS_MAX:
            raise ValueError(
                f"{context}: look {name} must be within "
                f"±{LOOK_PARAMETER_ABS_MAX}"
            )


def _require_move_parameters(parameters: Mapping[str, Any], *, context: str) -> None:
    allowed = {"forward", "strafe", "sprint", "jump"}
    unknown = set(parameters) - allowed
    if unknown:
        raise ValueError(
            f"{context}: move parameters may only include "
            f"forward/strafe/sprint/jump, got {sorted(unknown)}"
        )
    if "forward" not in parameters:
        raise ValueError(f"{context}: move requires forward")
    forward = parameters["forward"]
    if isinstance(forward, bool) or not isinstance(forward, (int, float)):
        raise ValueError(f"{context}: move forward must be a finite number")
    forward_value = float(forward)
    if not math.isfinite(forward_value):
        raise ValueError(f"{context}: move forward must be finite")
    if forward_value < 0.0 or forward_value > MOVE_FORWARD_ABS_MAX:
        raise ValueError(
            f"{context}: move forward must be in "
            f"[0, {MOVE_FORWARD_ABS_MAX}]"
        )
    strafe = parameters.get("strafe", 0.0)
    if isinstance(strafe, bool) or not isinstance(strafe, (int, float)):
        raise ValueError(f"{context}: move strafe must be a finite number")
    if float(strafe) != 0.0:
        raise ValueError(f"{context}: move strafe must be exactly 0.0")
    if parameters.get("sprint", False) is not False:
        raise ValueError(f"{context}: move sprint must be False")
    if parameters.get("jump", False) is not False:
        raise ValueError(f"{context}: move jump must be False")


def _require_r4_action(action: MacroAction, *, context: str) -> MacroAction:
    """Validate that ``action`` belongs to the closed R4 allowlist.

    The driver is the only place where this check is enforced; the
    test orchestrator must not bypass it.
    """
    if not isinstance(action, MacroAction):
        raise ValueError(f"{context}: action must be a MacroAction")
    if action.action_type not in ALLOWED_R4_ACTION_TYPES:
        raise ValueError(
            f"{context}: action_type {action.action_type!r} is outside the "
            f"R4 allowlist {sorted(ALLOWED_R4_ACTION_TYPES)}"
        )
    if action.target is not None and action.target not in ALLOWED_R4_TARGETS:
        raise ValueError(
            f"{context}: target {action.target!r} is outside the R4 allowlist "
            f"{sorted(ALLOWED_R4_TARGETS)}"
        )
    if action.action_type == "equip_item" and action.target not in {
        "water_bucket",
        "lava_bucket",
        "cobblestone",
    }:
        raise ValueError(
            f"{context}: equip_item target must be 'water_bucket', "
            f"'lava_bucket', or 'cobblestone', got {action.target!r}"
        )
    if action.action_type in {"wait", "look", "move"} and action.target is not None:
        raise ValueError(
            f"{context}: {action.action_type} action cannot have a target"
        )
    if action.action_type == "place_block" and action.target != "cobblestone":
        # Drivers may only place the cobblestone support block.
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
    if action.action_type == "look":
        _require_look_parameters(action.parameters, context=context)
    elif action.action_type == "move":
        _require_move_parameters(action.parameters, context=context)
    elif action.parameters:
        raise ValueError(
            f"{context}: {action.action_type} actions cannot contain parameters"
        )
    return action


@dataclass(frozen=True)
class CastingPlanStep:
    """One step in the bounded R4 plan.

    ``label`` is the semantic label used in the event log. ``phase``
    is the workflow stage. ``action`` is the (already whitelisted)
    :class:`MacroAction` to submit. ``relevant_action`` is ``True``
    when this step is a candidate for ``relevant_action_steps``
    (i.e. it places a fluid or a support block that could be
    causally linked to a target-cell block update).
    """

    label: str
    phase: str
    action: MacroAction
    relevant_action: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("plan step label must be a non-empty string")
        if self.phase not in {
            PHASE_PREPARE,
            PHASE_AIM,
            PHASE_PLACE_SUPPORT,
            PHASE_PLACE_LAVA,
            PHASE_PLACE_WATER,
            PHASE_WAIT_FOR_OBSIDIAN,
        }:
            raise ValueError(f"unknown casting plan phase: {self.phase!r}")
        if type(self.relevant_action) is not bool:
            raise ValueError("relevant_action must be a boolean")
        _require_r4_action(self.action, context=f"plan[{self.label!r}]")
        expected_relevant = self.action.action_type in {
            "place_block",
            "use_item",
        }
        if self.relevant_action is not expected_relevant:
            raise ValueError(
                "relevant_action must be true exactly for place_block/use_item"
            )


@dataclass(frozen=True)
class CastingC1DriverResult:
    """Public result of :func:`run_casting_c1_driver`.

    The driver never returns a casting verdict; the orchestrator
    owns the evaluator call. This object only reports whether the
    driver reached the end of the bounded plan, which steps it
    executed, and the event log.
    """

    status: str
    steps_executed: int
    wait_steps: int
    planned_steps: int
    final_observation: Observation
    events: tuple[Mapping[str, Any], ...]
    action_label_for_step: Mapping[int, str]
    relevant_action_steps: tuple[int, ...]
    terminated: bool
    truncated: bool
    blocked_reason: str | None

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
        if not isinstance(self.final_observation, Observation):
            raise ValueError("final_observation must be an Observation")
        if not isinstance(self.events, tuple):
            raise ValueError("events must be a tuple")
        if not isinstance(self.relevant_action_steps, tuple):
            raise ValueError("relevant_action_steps must be a tuple")
        for step in self.relevant_action_steps:
            _require_non_negative_int(step, "relevant_action_steps")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise ValueError("terminated and truncated must be booleans")
        if self.status == DRIVER_STATUS_COMPLETED:
            if self.steps_executed != self.planned_steps:
                raise ValueError("completed driver must execute the full plan")
            if self.blocked_reason is not None:
                raise ValueError("completed driver cannot have blocked_reason")
        elif not isinstance(self.blocked_reason, str) or not self.blocked_reason.strip():
            raise ValueError("blocked/failed driver requires blocked_reason")
        if not isinstance(self.action_label_for_step, Mapping):
            raise ValueError("action_label_for_step must be a mapping")
        frozen_events = tuple(_freeze_value(event) for event in self.events)
        frozen_labels = _freeze_value(self.action_label_for_step)
        object.__setattr__(self, "events", frozen_events)
        object.__setattr__(self, "action_label_for_step", frozen_labels)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "steps_executed": self.steps_executed,
            "wait_steps": self.wait_steps,
            "planned_steps": self.planned_steps,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "blocked_reason": self.blocked_reason,
            "events": _thaw_value(self.events),
            "action_label_for_step": _thaw_value(self.action_label_for_step),
            "relevant_action_steps": list(self.relevant_action_steps),
        }


def _wait_step(label: str, phase: str) -> CastingPlanStep:
    return CastingPlanStep(
        label=label,
        phase=phase,
        action=MacroAction.wait(),
    )


def _select_step(target: str, label: str, phase: str) -> CastingPlanStep:
    return CastingPlanStep(
        label=label,
        phase=phase,
        action=MacroAction(action_type="equip_item", target=target),
    )


def _aim_angles(
    target: tuple[float, float, float],
    *,
    eye: tuple[float, float, float] = FROZEN_PLAYER_EYE,
) -> tuple[float, float]:
    """Return absolute Minecraft yaw/pitch for one public click point."""
    delta_x = target[0] - eye[0]
    delta_y = target[1] - eye[1]
    delta_z = target[2] - eye[2]
    horizontal = math.hypot(delta_x, delta_z)
    return (
        -math.degrees(math.atan2(delta_x, delta_z)),
        -math.degrees(math.atan2(delta_y, horizontal)),
    )


def _relative_look_steps(
    *,
    label: str,
    phase: str,
    current: tuple[float, float],
    target_point: tuple[float, float, float],
) -> tuple[list[CastingPlanStep], tuple[float, float]]:
    """Split an absolute aim into bounded MineRL camera deltas."""
    target_yaw, target_pitch = _aim_angles(target_point)
    yaw_delta = target_yaw - current[0]
    pitch_delta = target_pitch - current[1]
    count = max(
        1,
        math.ceil(abs(yaw_delta) / LOOK_PARAMETER_ABS_MAX),
        math.ceil(abs(pitch_delta) / LOOK_PARAMETER_ABS_MAX),
    )
    steps = [
        CastingPlanStep(
            label=f"{label}.aim.{index + 1}",
            phase=phase,
            action=MacroAction(
                action_type="look",
                parameters={
                    "yaw": yaw_delta / count,
                    "pitch": pitch_delta / count,
                },
            ),
        )
        for index in range(count)
    ]
    return steps, (target_yaw, target_pitch)


def _place_support_step(label: str, phase: str) -> CastingPlanStep:
    return CastingPlanStep(
        label=label,
        phase=phase,
        action=MacroAction(action_type="place_block", target="cobblestone"),
        relevant_action=True,
    )


def _use_bucket_step(target: str, label: str, phase: str) -> CastingPlanStep:
    return CastingPlanStep(
        label=label,
        phase=phase,
        action=MacroAction(action_type="use_item", target=target),
        relevant_action=True,
    )


def build_casting_action_plan(
    *,
    support_block_wait_steps: int = DEFAULT_SUPPORT_BLOCK_WAIT_STEPS,
    fluid_settle_wait_steps: int = DEFAULT_FLUID_SETTLE_WAIT_STEPS,
    obsidian_wait_steps: int = DEFAULT_OBSIDIAN_WAIT_STEPS,
) -> tuple[CastingPlanStep, ...]:
    """Build the bounded R4 single-block casting plan.

    The plan is fully deterministic. The fixed sequence is:

    1. Equip cobblestone + brief wait, aim at two distinct solid top
       faces, and place two support blocks.
    2. Equip lava bucket + brief wait, aim at the target cell's support
       face, use lava, and settle.
    3. Equip water bucket + brief wait, aim at an adjacent support face,
       use water, and settle.
    4. Bounded extra wait so the casting evaluator can observe the
       obsidian transition.

    Look / move parameters are closed and bounded. Place / use steps
    are relevant actions; the driver fail-closes if they produce no
    inventory change on a semantics-aware backend.

    All wait counts are parameterised but bounded by
    :func:`run_casting_c1_driver` so a caller cannot ask the driver
    to run forever.
    """
    support_waits = _require_non_negative_int(
        support_block_wait_steps, "support_block_wait_steps"
    )
    fluid_waits = _require_non_negative_int(
        fluid_settle_wait_steps, "fluid_settle_wait_steps"
    )
    obsidian_waits = _require_non_negative_int(
        obsidian_wait_steps, "obsidian_wait_steps"
    )
    # Equip-release and settle waits only. Camera actions are counted as
    # environment steps but are not wait-budget steps.
    total_waits = (
        2  # cobble equip release waits
        + (2 * support_waits)
        + 2  # lava/water equip release waits
        + (2 * fluid_waits)
        + obsidian_waits
    )
    if total_waits > MAX_PLAN_WAIT_STEPS:
        raise ValueError(
            "casting plan wait steps exceed the hard limit: "
            f"{total_waits} > {MAX_PLAN_WAIT_STEPS}"
        )
    plan: list[CastingPlanStep] = [
        _select_step(
            "cobblestone", "support.select_cobble_1", PHASE_PLACE_SUPPORT
        ),
        _wait_step("support.select_cobble_1.release", PHASE_PLACE_SUPPORT),
    ]
    camera = (0.0, 0.0)
    look, camera = _relative_look_steps(
        label="support.block_1",
        phase=PHASE_PLACE_SUPPORT,
        current=camera,
        target_point=FROZEN_SUPPORT_FACE_POINTS[0],
    )
    plan.extend(look)
    plan.append(_place_support_step("support.block_1", PHASE_PLACE_SUPPORT))
    plan.extend(
        _wait_step(
            f"support.block_1.settle.{index + 1}", PHASE_PLACE_SUPPORT
        )
        for index in range(support_waits)
    )
    plan.extend(
        [
            _select_step(
                "cobblestone",
                "support.select_cobble_2",
                PHASE_PLACE_SUPPORT,
            ),
            _wait_step(
                "support.select_cobble_2.release", PHASE_PLACE_SUPPORT
            ),
        ]
    )
    look, camera = _relative_look_steps(
        label="support.block_2",
        phase=PHASE_PLACE_SUPPORT,
        current=camera,
        target_point=FROZEN_SUPPORT_FACE_POINTS[1],
    )
    plan.extend(look)
    plan.append(_place_support_step("support.block_2", PHASE_PLACE_SUPPORT))
    plan.extend(
        _wait_step(
            f"support.block_2.settle.{index + 1}", PHASE_PLACE_SUPPORT
        )
        for index in range(support_waits)
    )
    plan.extend(
        [
            _select_step(
                "lava_bucket", "casting.select_lava", PHASE_PLACE_LAVA
            ),
            _wait_step("casting.select_lava.release", PHASE_PLACE_LAVA),
        ]
    )
    look, camera = _relative_look_steps(
        label="casting.use_lava",
        phase=PHASE_PLACE_LAVA,
        current=camera,
        target_point=FROZEN_LAVA_FACE_POINT,
    )
    plan.extend(look)
    plan.append(
        _use_bucket_step("lava_bucket", "casting.use_lava", PHASE_PLACE_LAVA)
    )
    plan.extend(
        _wait_step(
            f"casting.lava.settle.{index + 1}", PHASE_PLACE_LAVA
        )
        for index in range(fluid_waits)
    )
    plan.extend(
        [
            _select_step(
                "water_bucket", "casting.select_water", PHASE_PLACE_WATER
            ),
            _wait_step("casting.select_water.release", PHASE_PLACE_WATER),
        ]
    )
    look, camera = _relative_look_steps(
        label="casting.use_water",
        phase=PHASE_PLACE_WATER,
        current=camera,
        target_point=FROZEN_WATER_FACE_POINT,
    )
    plan.extend(look)
    plan.append(
        _use_bucket_step("water_bucket", "casting.use_water", PHASE_PLACE_WATER)
    )
    plan.extend(
        _wait_step(
            f"casting.water.settle.{index + 1}", PHASE_PLACE_WATER
        )
        for index in range(fluid_waits)
    )
    plan.extend(
        _wait_step(
            f"casting.obsidian.wait.{index + 1}", PHASE_WAIT_FOR_OBSIDIAN
        )
        for index in range(obsidian_waits)
    )
    for step in plan:
        _require_r4_action(step.action, context=f"plan[{step.label!r}]")
    return tuple(plan)


# ----------------------------------------------------------------------
# Driver implementation
# ----------------------------------------------------------------------


def _visible_inventory_has(observation: Observation, item: str) -> bool:
    """Return ``True`` iff ``observation.visible_inventory`` has
    a positive quantity of ``item``.

    This is the only inventory predicate the driver uses; it is
    intentionally restricted to the closed R4 target list. The
    driver never inspects ``Observation.frame`` or any other field.
    """
    if item not in ALLOWED_R4_TARGETS:
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


def _visible_inventory_quantity(observation: Observation, item: str) -> int:
    """Return one closed-set visible item quantity with strict typing."""
    if item not in ALLOWED_R4_TARGETS:
        raise ValueError(f"driver cannot inspect item {item!r}")
    inventory = observation.visible_inventory or {}
    quantity = inventory.get(item, 0)
    if type(quantity) is not int or quantity < 0:
        raise ValueError(
            "visible_inventory quantities must be non-negative integers"
        )
    return quantity


def _assert_workflow_stage(observation: Observation) -> None:
    """Verify the Observation advertises the casting workflow.

    The driver is workflow-scoped; refusing to run on a different
    workflow stage is part of the R4 contract.
    """
    if not isinstance(observation, Observation):
        raise ValueError("expected an Observation")
    if observation.workflow_stage != "casting_c1_fixed":
        raise ValueError(
            "driver only supports workflow 'casting_c1_fixed', got "
            f"{observation.workflow_stage!r}"
        )


def _backend_exposes_casting_truth_surface(backend: Any) -> bool:
    """Return ``True`` iff ``backend`` exposes the casting surface.

    This is *informational only*. The driver itself does not refuse
    to run on such a backend (the standard
    :class:`FakeEnvironmentBackend` exposes the surface for the
    orchestrator, and the driver must run on it). Instead, the test
    suite uses this helper to verify isolation in wrapper-style
    tests where the surface is intentionally re-exposed.
    """
    return any(
        hasattr(backend, name)
        for name in (
            "set_casting_evaluation_state",
            "get_casting_evaluation_state",
        )
    )


def run_casting_c1_driver(
    backend: Any,
    task: TaskInstance,
    *,
    plan: tuple[CastingPlanStep, ...] | None = None,
    max_wait_steps: int = DEFAULT_MAX_WAIT_STEPS,
    max_environment_steps: int | None = None,
    max_game_time_seconds: float | None = None,
    event_sink: Callable[[Mapping[str, Any]], None] | None = None,
) -> CastingC1DriverResult:
    """Run the bounded R4 single-block casting plan on ``backend``.

    The driver:

    1. Calls ``backend.reset(task)`` and uses the returned
       ``Observation`` only for the initial ``visible_inventory`` /
       ``workflow_stage`` check. It never reads casting truth.
    2. Walks the plan step-by-step, calling
       ``backend.step({AGENT_ID: action})`` once per step. Each
       step is validated against the closed R4 allowlist before
       submission.
    3. Rejects any plan step that requires an item the Agent is
       not carrying in its visible inventory. The check uses
       ``visible_inventory`` only; the driver never reads
       ``Observation.frame`` or any other field.
    4. Bounded by ``max_environment_steps`` and
       ``max_game_time_seconds``; when either is exceeded the
       driver returns ``status="blocked"`` with a descriptive
       ``blocked_reason``.

    The driver does *not* call ``set_casting_evaluation_state`` /
    ``get_casting_evaluation_state`` and does *not* invoke
    :class:`CastingEvaluator`. Both live in the test orchestrator.
    """
    if not isinstance(task, TaskInstance):
        raise ValueError("task must be a TaskInstance")
    if task.workflow != "casting_c1_fixed":
        raise ValueError(
            f"driver only supports workflow 'casting_c1_fixed', got "
            f"{task.workflow!r}"
        )
    _require_positive_int(max_wait_steps, "max_wait_steps")
    if max_wait_steps > MAX_PLAN_WAIT_STEPS:
        raise ValueError(
            f"max_wait_steps must be <= {MAX_PLAN_WAIT_STEPS}"
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
        plan = build_casting_action_plan()
    if not isinstance(plan, tuple) or not plan:
        raise ValueError("plan must be non-empty")
    if any(not isinstance(step, CastingPlanStep) for step in plan):
        raise ValueError("plan must contain only CastingPlanStep values")
    if len(plan) > task_step_limit:
        raise ValueError(
            "plan length cannot exceed the task step limit "
            f"{task_step_limit}"
        )
    plan_wait_steps = sum(
        step.action.action_type == "wait" for step in plan
    )
    if plan_wait_steps > MAX_PLAN_WAIT_STEPS:
        raise ValueError(
            "plan wait steps cannot exceed the hard limit "
            f"{MAX_PLAN_WAIT_STEPS}"
        )

    if not hasattr(backend, "reset") or not hasattr(backend, "step"):
        raise ValueError("backend must implement reset/step")
    # The driver does not refuse to run on a backend that exposes
    # the casting truth surface (the standard
    # :class:`FakeEnvironmentBackend` does, because the orchestrator
    # needs it). The driver instead guarantees, by construction, that
    # it never *calls* ``set_casting_evaluation_state`` /
    # ``get_casting_evaluation_state``. The
    # :mod:`tests.test_casting_driver` suite pins this with a spy.

    observations = backend.reset(task)
    final_observation = observations[AGENT_ID]
    if not isinstance(final_observation, Observation):
        raise ValueError("backend.reset must return Observation values")
    _assert_workflow_stage(final_observation)
    # The time budget is *elapsed* time from reset, not wall-clock
    # time. Storing the reset timestamp keeps the contract clean
    # even when the backend reports wall-clock ``timestamp`` (e.g.
    # the ``FakeEnvironmentBackend`` uses ``time.time()``).
    reset_timestamp = (
        float(final_observation.timestamp)
        if math.isfinite(final_observation.timestamp)
        else None
    )

    # The plan is a fixed sequence; the driver never inspects the
    # Agent's hotbar to pick the next action. We do, however, refuse
    # to start a step that requires an item the Agent is not
    # carrying. The check uses ``visible_inventory`` only.
    events: list[Mapping[str, Any]] = []
    action_label_for_step: dict[int, str] = {}
    relevant_action_steps: list[int] = []
    wait_steps = 0
    steps_executed = 0
    blocked_reason: str | None = None
    status = DRIVER_STATUS_COMPLETED
    backend_terminated = False
    backend_truncated = False
    # item, before, expected, source_label, source_step, settle_ticks_seen
    pending_inventory_effect: tuple[str, int, int, str, int, int] | None = None

    def record_event(event: Mapping[str, Any]) -> None:
        identified = {
            "episode_id": task.task_id,
            "agent_id": AGENT_ID,
            **dict(event),
        }
        events.append(identified)
        if event_sink is not None:
            # The sink receives a detached snapshot so callback code cannot
            # mutate the driver's evidence before the result freezes it.
            event_sink(_thaw_value(_freeze_value(identified)))

    record_event(
        {
            "step_id": final_observation.step_id,
            "label": "environment.reset",
            "phase": PHASE_PREPARE,
            "action_type": "wait",
            "target": None,
            "relevant_action": False,
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
        _require_r4_action(
            plan_step.action, context=f"plan[{plan_index}]={plan_step.label!r}"
        )
        if (
            pending_inventory_effect is not None
            and plan_step.action.action_type != "wait"
        ):
            item, before, expected, source_label, source_step, settle_ticks_seen = (
                pending_inventory_effect
            )
            blocked_reason = (
                "inventory effect verification requires an immediate wait "
                f"after {source_label}: item={item!r}, "
                f"expected {before}->{expected}, source_step={source_step}"
            )
            status = DRIVER_STATUS_BLOCKED
            break
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

        previous_step_id = final_observation.step_id
        relevant_item_before: int | None = None
        if plan_step.relevant_action:
            relevant_item = plan_step.action.target
            if relevant_item is None:
                raise ValueError("relevant action requires a target item")
            relevant_item_before = _visible_inventory_quantity(
                final_observation, relevant_item
            )
        try:
            step = backend.step({AGENT_ID: plan_step.action})
        except (RuntimeError, OSError, TypeError) as error:
            # ``RuntimeError`` / ``OSError`` / ``TypeError`` are
            # treated as runtime backend failures: the driver
            # records the failure and stops. ``ValueError`` from
            # the backend (a contract violation) is *not* caught
            # here — the post-step contract checks (stale
            # step_id, identity mismatch, allowlist validation)
            # below will raise their own ``ValueError`` on a
            # contract violation, and that one propagates
            # directly to the caller.
            blocked_reason = (
                f"{type(error).__name__} at {plan_step.label}: {error}"
            )
            status = DRIVER_STATUS_FAILED
            record_event(
                {
                    "step_id": getattr(error, "step_id", None),
                    "label": plan_step.label,
                    "phase": plan_step.phase,
                    "action_type": plan_step.action.action_type,
                    "target": plan_step.action.target,
                    "relevant_action": plan_step.relevant_action,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            break

        if not isinstance(step, BackendStep):
            raise ValueError("backend.step must return a BackendStep")
        if step.episode_id != task.task_id:
            raise ValueError("BackendStep episode_id must match the current task")
        if step.step_id != previous_step_id + 1:
            raise ValueError(
                "BackendStep step_id must advance exactly once: "
                f"expected {previous_step_id + 1}, got {step.step_id}"
            )
        # ``BackendStep.__post_init__`` already enforces that
        # every observation's ``step_id`` matches ``step.step_id``
        # and that the ``episode_id`` / ``agent_id`` line up. Any
        # stale-step or identity-mismatch attempt therefore fails
        # at the contract layer, before the driver sees it; the
        # driver relies on the typed contract instead of
        # re-implementing the same checks.
        steps_executed += 1
        if plan_step.action.action_type == "wait":
            wait_steps += 1
        next_observation = step.observations[AGENT_ID]
        if not isinstance(next_observation, Observation):
            raise ValueError("backend.step must return Observation values")
        _assert_workflow_stage(next_observation)
        backend_terminated = step.terminated
        backend_truncated = step.truncated

        # Hard stop on time budget (elapsed time from reset, not
        # wall-clock). The driver only consults ``Observation.
        # timestamp`` and never reads any evaluator truth.
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
                    relevant_action_steps.append(step.step_id)
                record_event(
                    {
                        "step_id": step.step_id,
                        "label": plan_step.label,
                        "phase": plan_step.phase,
                        "action_type": plan_step.action.action_type,
                        "target": plan_step.action.target,
                        "relevant_action": plan_step.relevant_action,
                        "budget_exceeded": "time",
                    }
                )
                break
        final_observation = next_observation
        action_label_for_step[step.step_id] = plan_step.label
        verified_pending_effect: tuple[str, int] | None = None
        waiting_pending_effect: tuple[str, int] | None = None
        if pending_inventory_effect is not None:
            item, before, expected, source_label, source_step, settle_ticks_seen = (
                pending_inventory_effect
            )
            observed = _visible_inventory_quantity(final_observation, item)
            settle_ticks_seen += 1
            if observed == expected:
                verified_pending_effect = (source_label, source_step)
                pending_inventory_effect = None
            elif observed == before and (
                settle_ticks_seen < MAX_INVENTORY_CONFIRMATION_WAIT_STEPS
            ):
                pending_inventory_effect = (
                    item,
                    before,
                    expected,
                    source_label,
                    source_step,
                    settle_ticks_seen,
                )
                waiting_pending_effect = (source_label, source_step)
            else:
                reason = (
                    "expected_inventory_effect_missing"
                    if observed == before
                    else "unexpected_inventory_effect"
                )
                blocked_reason = (
                    f"{reason} at "
                    f"{source_label}: expected {item!r} quantity "
                    f"{before}->{expected} within "
                    f"{MAX_INVENTORY_CONFIRMATION_WAIT_STEPS} settle ticks, "
                    f"observed {observed} after {settle_ticks_seen}"
                )
                status = DRIVER_STATUS_BLOCKED
                record_event(
                    {
                        "step_id": step.step_id,
                        "label": plan_step.label,
                        "phase": plan_step.phase,
                        "action_type": plan_step.action.action_type,
                        "target": plan_step.action.target,
                        "relevant_action": plan_step.relevant_action,
                        "visible_inventory": dict(
                            final_observation.visible_inventory or {}
                        ),
                        "inventory_effect_confirmed": False,
                        "verifies_action_step": source_step,
                        "blocked_reason": blocked_reason,
                    }
                )
                break
        relevant_effect_confirmed: bool | None = None
        if plan_step.relevant_action:
            relevant_action_steps.append(step.step_id)
            inventory_after = dict(final_observation.visible_inventory or {})
            relevant_item = plan_step.action.target
            if relevant_item is None or relevant_item_before is None:
                raise ValueError("relevant action inventory check is unavailable")
            relevant_item_after = _visible_inventory_quantity(
                final_observation, relevant_item
            )
            expected_item_after = relevant_item_before - 1
            if relevant_item_after == expected_item_after:
                relevant_effect_confirmed = True
            elif relevant_item_after == relevant_item_before:
                pending_inventory_effect = (
                    relevant_item,
                    relevant_item_before,
                    expected_item_after,
                    plan_step.label,
                    step.step_id,
                    0,
                )
                relevant_effect_confirmed = None
            else:
                blocked_reason = (
                    "unexpected_inventory_effect at "
                    f"{plan_step.label}: expected {relevant_item!r} "
                    f"quantity {relevant_item_before}->{expected_item_after}, "
                    f"observed {relevant_item_after}"
                )
                status = DRIVER_STATUS_BLOCKED
                record_event(
                    {
                        "step_id": step.step_id,
                        "label": plan_step.label,
                        "phase": plan_step.phase,
                        "action_type": plan_step.action.action_type,
                        "target": plan_step.action.target,
                        "relevant_action": plan_step.relevant_action,
                        "visible_inventory": inventory_after,
                        "inventory_effect_confirmed": False,
                        "blocked_reason": blocked_reason,
                    }
                )
                break
        record_event(
            {
                "step_id": step.step_id,
                "label": plan_step.label,
                "phase": plan_step.phase,
                "action_type": plan_step.action.action_type,
                "target": plan_step.action.target,
                "relevant_action": plan_step.relevant_action,
                "visible_inventory": dict(
                    final_observation.visible_inventory or {}
                ),
                "inventory_effect_confirmed": (
                    relevant_effect_confirmed
                    if plan_step.relevant_action
                    else (True if verified_pending_effect is not None else None)
                ),
                "verifies_action_step": (
                    verified_pending_effect[1]
                    if verified_pending_effect is not None
                    else (
                        waiting_pending_effect[1]
                        if waiting_pending_effect is not None
                        else None
                    )
                ),
                "inventory_confirmation_wait_ticks": (
                    settle_ticks_seen
                    if waiting_pending_effect is not None
                    or verified_pending_effect is not None
                    else None
                ),
            }
        )
        if step.terminated or step.truncated:
            # The driver never flips termination on its own; if
            # the backend pre-terminates (e.g. a real MineRL
            # ``done`` flag) we still return and let the
            # orchestrator build the final state.
            if plan_index + 1 < len(plan):
                status = DRIVER_STATUS_BLOCKED
                reason = "termination" if step.terminated else "truncation"
                blocked_reason = (
                    f"plan interrupted by backend {reason} at {plan_step.label}"
                )
            break

    if blocked_reason is None and steps_executed < len(plan):
        # The plan was not fully executed but we did not block on
        # a hard budget; this happens when the backend reports
        # termination mid-plan. Surface it explicitly.
        blocked_reason = "plan interrupted by backend termination"
        status = DRIVER_STATUS_BLOCKED

    if blocked_reason is None and pending_inventory_effect is not None:
        item, before, expected, source_label, source_step, settle_ticks_seen = (
            pending_inventory_effect
        )
        blocked_reason = (
            "inventory effect remained unverified at plan end after "
            f"{source_label}: item={item!r}, expected {before}->{expected}, "
            f"source_step={source_step}, settle_ticks_seen={settle_ticks_seen}"
        )
        status = DRIVER_STATUS_BLOCKED

    return CastingC1DriverResult(
        status=status,
        steps_executed=steps_executed,
        wait_steps=wait_steps,
        planned_steps=len(plan),
        final_observation=final_observation,
        events=tuple(events),
        action_label_for_step=MappingProxyType(action_label_for_step),
        relevant_action_steps=tuple(relevant_action_steps),
        terminated=backend_terminated,
        truncated=backend_truncated,
        blocked_reason=blocked_reason,
    )


__all__ = [
    "AGENT_ID",
    "ALLOWED_R4_ACTION_TYPES",
    "ALLOWED_R4_TARGETS",
    "CastingC1DriverResult",
    "CastingPlanStep",
    "DEFAULT_FLUID_SETTLE_WAIT_STEPS",
    "DEFAULT_MAX_ENVIRONMENT_STEPS",
    "DEFAULT_MAX_GAME_TIME_SECONDS",
    "DEFAULT_MAX_WAIT_STEPS",
    "MAX_PLAN_WAIT_STEPS",
    "MAX_INVENTORY_CONFIRMATION_WAIT_STEPS",
    "DEFAULT_OBSIDIAN_WAIT_STEPS",
    "DEFAULT_SUPPORT_BLOCK_WAIT_STEPS",
    "DRIVER_STATUS_BLOCKED",
    "DRIVER_STATUS_COMPLETED",
    "DRIVER_STATUS_FAILED",
    "DRIVER_STATUSES",
    "PHASE_AIM",
    "PHASE_PLACE_LAVA",
    "PHASE_PLACE_SUPPORT",
    "PHASE_PLACE_WATER",
    "PHASE_PREPARE",
    "PHASE_WAIT_FOR_OBSIDIAN",
    "FROZEN_PLAYER_EYE",
    "FROZEN_SUPPORT_FACE_POINTS",
    "FROZEN_LAVA_FACE_POINT",
    "FROZEN_WATER_FACE_POINT",
    "build_casting_action_plan",
    "run_casting_c1_driver",
]

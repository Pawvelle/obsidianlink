"""MineRL-independent P1 E5 movement calibration contract.

The position/orientation values here are a temporary P1 evaluator-only
surface. They are not Agent-visible, not the future v2 canonical
Observation, and not the legacy :class:`obsidianlink.core.types.Observation`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


MOVEMENT_OK = "movement_ok"
POSITION_BEFORE_MISSING = "position_before_missing"
POSITION_AFTER_MISSING = "position_after_missing"
POSITION_INVALID = "position_invalid"
MOVEMENT_ORIENTATION_MISSING = "movement_orientation_missing"
MOVEMENT_ORIENTATION_INVALID = "movement_orientation_invalid"
MOVEMENT_ACTION_REJECTED = "movement_action_rejected"
MOVEMENT_WRONG_ACTION_TYPE = "movement_wrong_action_type"
MOVEMENT_CALIBRATION_MISMATCH = "movement_calibration_mismatch"
MOVEMENT_TEST_ACTION_NOT_EXECUTED = "movement_test_action_not_executed"
MOVEMENT_MULTIPLE_TEST_ACTIONS = "movement_multiple_test_actions"
MOVEMENT_STEP_IDENTITY_MISMATCH = "movement_step_identity_mismatch"
MOVEMENT_NO_DISPLACEMENT = "movement_no_displacement"
MOVEMENT_WRONG_DIRECTION = "movement_wrong_direction"
MOVEMENT_LATERAL_DRIFT_EXCESSIVE = "movement_lateral_drift_excessive"
MOVEMENT_TELEPORT_DETECTED = "movement_teleport_detected"
MOVEMENT_VERTICAL_DRIFT_EXCESSIVE = "movement_vertical_drift_excessive"
MOVEMENT_TRUTH_LEAK = "movement_truth_leak"

MOVEMENT_OUTCOMES = frozenset(
    {
        MOVEMENT_OK,
        POSITION_BEFORE_MISSING,
        POSITION_AFTER_MISSING,
        POSITION_INVALID,
        MOVEMENT_ORIENTATION_MISSING,
        MOVEMENT_ORIENTATION_INVALID,
        MOVEMENT_ACTION_REJECTED,
        MOVEMENT_WRONG_ACTION_TYPE,
        MOVEMENT_CALIBRATION_MISMATCH,
        MOVEMENT_TEST_ACTION_NOT_EXECUTED,
        MOVEMENT_MULTIPLE_TEST_ACTIONS,
        MOVEMENT_STEP_IDENTITY_MISMATCH,
        MOVEMENT_NO_DISPLACEMENT,
        MOVEMENT_WRONG_DIRECTION,
        MOVEMENT_LATERAL_DRIFT_EXCESSIVE,
        MOVEMENT_TELEPORT_DETECTED,
        MOVEMENT_VERTICAL_DRIFT_EXCESSIVE,
        MOVEMENT_TRUTH_LEAK,
    }
)


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def finite_number(value: object, field_name: str) -> float:
    """Return a finite scalar number while rejecting bool and arrays."""

    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite int or float")
    return float(value)


@dataclass(frozen=True)
class PlayerPositionSnapshot:
    """Temporary evaluator-only Minecraft player-position truth."""

    episode_id: str
    agent_id: str
    step_id: int
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _identifier(self.episode_id, "episode_id"))
        object.__setattr__(self, "agent_id", _identifier(self.agent_id, "agent_id"))
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative int")
        for field_name in ("x", "y", "z"):
            object.__setattr__(self, field_name, finite_number(getattr(self, field_name), field_name))


@dataclass(frozen=True)
class MovementOrientationSnapshot:
    """Minimal reset-time yaw truth used only to define forward."""

    episode_id: str
    agent_id: str
    step_id: int
    yaw: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _identifier(self.episode_id, "episode_id"))
        object.__setattr__(self, "agent_id", _identifier(self.agent_id, "agent_id"))
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative int")
        object.__setattr__(self, "yaw", finite_number(self.yaw, "yaw"))


@dataclass(frozen=True)
class MovementActionExecution:
    """Observed backend response to the single bounded E5 move action."""

    episode_id: str
    agent_id: str
    step_id: int
    action_type: str
    forward: float
    strafe: float
    sprint: bool
    jump: bool
    duration_ticks: int
    translated_action_accepted: bool
    tested_action_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _identifier(self.episode_id, "episode_id"))
        object.__setattr__(self, "agent_id", _identifier(self.agent_id, "agent_id"))
        object.__setattr__(self, "action_type", _identifier(self.action_type, "action_type"))
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative int")
        object.__setattr__(self, "forward", finite_number(self.forward, "forward"))
        object.__setattr__(self, "strafe", finite_number(self.strafe, "strafe"))
        if type(self.sprint) is not bool or type(self.jump) is not bool:
            raise ValueError("sprint and jump must be bool")
        if type(self.duration_ticks) is not int or self.duration_ticks < 1:
            raise ValueError("duration_ticks must be a positive int")
        if type(self.translated_action_accepted) is not bool:
            raise ValueError("translated_action_accepted must be bool")
        if type(self.tested_action_count) is not int or self.tested_action_count < 0:
            raise ValueError("tested_action_count must be a non-negative int")


@dataclass(frozen=True)
class MovementInspection:
    outcome: str
    error: str | None
    delta_x: float | None = None
    delta_y: float | None = None
    delta_z: float | None = None
    horizontal_distance: float | None = None
    total_distance: float | None = None
    forward_projection: float | None = None
    lateral_projection: float | None = None
    moved: bool | None = None
    direction_match: bool | None = None
    lateral_drift_ok: bool | None = None
    teleport_guard_ok: bool | None = None
    vertical_drift_ok: bool | None = None

    def __post_init__(self) -> None:
        if self.outcome not in MOVEMENT_OUTCOMES:
            raise ValueError(f"unknown movement outcome: {self.outcome!r}")

    @property
    def valid(self) -> bool:
        return self.outcome == MOVEMENT_OK


def inspect_movement(
    before: PlayerPositionSnapshot,
    after: PlayerPositionSnapshot,
    orientation: MovementOrientationSnapshot,
    execution: MovementActionExecution,
    *,
    minimum_horizontal_distance: float,
    minimum_forward_projection: float,
    maximum_lateral_drift: float,
    maximum_horizontal_distance: float,
    maximum_vertical_drift: float,
) -> MovementInspection:
    """Fail closed unless one accepted forward move yields plausible motion."""

    minimum_horizontal_distance = finite_number(minimum_horizontal_distance, "minimum_horizontal_distance")
    minimum_forward_projection = finite_number(minimum_forward_projection, "minimum_forward_projection")
    maximum_lateral_drift = finite_number(maximum_lateral_drift, "maximum_lateral_drift")
    maximum_horizontal_distance = finite_number(maximum_horizontal_distance, "maximum_horizontal_distance")
    maximum_vertical_drift = finite_number(maximum_vertical_drift, "maximum_vertical_drift")
    if min(minimum_horizontal_distance, minimum_forward_projection, maximum_lateral_drift, maximum_horizontal_distance, maximum_vertical_drift) < 0:
        raise ValueError("movement thresholds must be non-negative")
    if maximum_horizontal_distance < minimum_horizontal_distance:
        raise ValueError("maximum horizontal distance must cover the minimum")
    if execution.action_type != "move":
        return MovementInspection(MOVEMENT_WRONG_ACTION_TYPE, "tested action must be move")
    if (execution.forward, execution.strafe, execution.sprint, execution.jump, execution.duration_ticks) != (1.0, 0.0, False, False, 1):
        return MovementInspection(MOVEMENT_CALIBRATION_MISMATCH, "tested move differs from frozen E5 calibration")
    if execution.tested_action_count == 0:
        return MovementInspection(MOVEMENT_TEST_ACTION_NOT_EXECUTED, "movement test action was not executed")
    if execution.tested_action_count != 1:
        return MovementInspection(MOVEMENT_MULTIPLE_TEST_ACTIONS, "exactly one movement test action is required")
    if not execution.translated_action_accepted:
        return MovementInspection(MOVEMENT_ACTION_REJECTED, "movement action translation was rejected")
    if (
        before.episode_id != execution.episode_id
        or after.episode_id != execution.episode_id
        or orientation.episode_id != execution.episode_id
        or before.agent_id != execution.agent_id
        or after.agent_id != execution.agent_id
        or orientation.agent_id != execution.agent_id
        or before.step_id != 0
        or orientation.step_id != before.step_id
        or execution.step_id != 1
        or after.step_id != execution.step_id
    ):
        return MovementInspection(MOVEMENT_STEP_IDENTITY_MISMATCH, "movement identity or step sequence is invalid")

    dx, dy, dz = after.x - before.x, after.y - before.y, after.z - before.z
    horizontal = math.hypot(dx, dz)
    total = math.sqrt(dx * dx + dy * dy + dz * dz)
    yaw_radians = math.radians(orientation.yaw)
    # Minecraft yaw 0 faces +Z and +90 faces -X.
    forward = dx * -math.sin(yaw_radians) + dz * math.cos(yaw_radians)
    lateral = dx * math.cos(yaw_radians) + dz * math.sin(yaw_radians)
    moved = horizontal >= minimum_horizontal_distance
    direction = forward >= minimum_forward_projection
    lateral_ok = abs(lateral) <= maximum_lateral_drift
    teleport_ok = horizontal <= maximum_horizontal_distance
    vertical_ok = abs(dy) <= maximum_vertical_drift
    evidence = dict(
        delta_x=dx,
        delta_y=dy,
        delta_z=dz,
        horizontal_distance=horizontal,
        total_distance=total,
        forward_projection=forward,
        lateral_projection=lateral,
        moved=moved,
        direction_match=direction,
        lateral_drift_ok=lateral_ok,
        teleport_guard_ok=teleport_ok,
        vertical_drift_ok=vertical_ok,
    )
    if not moved:
        return MovementInspection(MOVEMENT_NO_DISPLACEMENT, "observed horizontal displacement is below the minimum", **evidence)
    if not teleport_ok:
        return MovementInspection(MOVEMENT_TELEPORT_DETECTED, "observed horizontal displacement exceeds the teleport guard", **evidence)
    if not direction:
        return MovementInspection(MOVEMENT_WRONG_DIRECTION, "observed displacement lacks forward progress", **evidence)
    if not lateral_ok:
        return MovementInspection(MOVEMENT_LATERAL_DRIFT_EXCESSIVE, "observed lateral drift exceeds the frozen bound", **evidence)
    if not vertical_ok:
        return MovementInspection(MOVEMENT_VERTICAL_DRIFT_EXCESSIVE, "observed vertical drift exceeds the frozen bound", **evidence)
    return MovementInspection(MOVEMENT_OK, None, **evidence)

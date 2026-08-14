"""MineRL-independent P1 E4 camera calibration contract.

The orientation snapshots in this module are a temporary P1 evaluator-only
surface.  They are not Agent-visible, not the future v2 canonical
Observation, and not the legacy :class:`obsidianlink.core.types.Observation`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


CAMERA_OK = "camera_ok"
ORIENTATION_BEFORE_MISSING = "orientation_before_missing"
ORIENTATION_AFTER_MISSING = "orientation_after_missing"
ORIENTATION_INVALID = "orientation_invalid"
ACTION_REJECTED = "action_rejected"
WRONG_ACTION_TYPE = "wrong_action_type"
TEST_ACTION_NOT_EXECUTED = "test_action_not_executed"
MULTIPLE_TEST_ACTIONS = "multiple_test_actions"
STEP_IDENTITY_MISMATCH = "step_identity_mismatch"
YAW_UNCHANGED = "yaw_unchanged"
YAW_WRONG_DIRECTION = "yaw_wrong_direction"
YAW_MAGNITUDE_MISMATCH = "yaw_magnitude_mismatch"
PITCH_DRIFT_EXCESSIVE = "pitch_drift_excessive"
CAMERA_TRUTH_LEAK = "camera_truth_leak"

CAMERA_OUTCOMES = frozenset(
    {
        CAMERA_OK,
        ORIENTATION_BEFORE_MISSING,
        ORIENTATION_AFTER_MISSING,
        ORIENTATION_INVALID,
        ACTION_REJECTED,
        WRONG_ACTION_TYPE,
        TEST_ACTION_NOT_EXECUTED,
        MULTIPLE_TEST_ACTIONS,
        STEP_IDENTITY_MISMATCH,
        YAW_UNCHANGED,
        YAW_WRONG_DIRECTION,
        YAW_MAGNITUDE_MISMATCH,
        PITCH_DRIFT_EXCESSIVE,
        CAMERA_TRUTH_LEAK,
    }
)


def _identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def finite_angle(value: object, field_name: str) -> float:
    """Return an exact finite numeric angle, rejecting bool and arrays."""

    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite int or float")
    return float(value)


def normalized_angular_delta(after: object, before: object) -> float:
    """Return signed shortest ``after - before`` in [-180, 180]."""

    raw = finite_angle(after, "after") - finite_angle(before, "before")
    normalized = (raw + 180.0) % 360.0 - 180.0
    if normalized == -180.0 and raw > 0.0:
        return 180.0
    return normalized


@dataclass(frozen=True)
class CameraOrientationSnapshot:
    """Temporary evaluator/calibration-only Minecraft orientation truth."""

    episode_id: str
    agent_id: str
    step_id: int
    yaw: float
    pitch: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _identifier(self.episode_id, "episode_id"))
        object.__setattr__(self, "agent_id", _identifier(self.agent_id, "agent_id"))
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative int")
        object.__setattr__(self, "yaw", finite_angle(self.yaw, "yaw"))
        object.__setattr__(self, "pitch", finite_angle(self.pitch, "pitch"))


@dataclass(frozen=True)
class CameraActionExecution:
    """Observed backend response to the single bounded E4 test action."""

    episode_id: str
    agent_id: str
    step_id: int
    action_type: str
    requested_yaw: float
    requested_pitch: float
    translated_action_accepted: bool
    tested_action_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _identifier(self.episode_id, "episode_id"))
        object.__setattr__(self, "agent_id", _identifier(self.agent_id, "agent_id"))
        object.__setattr__(self, "action_type", _identifier(self.action_type, "action_type"))
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative int")
        object.__setattr__(self, "requested_yaw", finite_angle(self.requested_yaw, "requested_yaw"))
        object.__setattr__(self, "requested_pitch", finite_angle(self.requested_pitch, "requested_pitch"))
        if type(self.translated_action_accepted) is not bool:
            raise ValueError("translated_action_accepted must be bool")
        if type(self.tested_action_count) is not int or self.tested_action_count < 0:
            raise ValueError("tested_action_count must be a non-negative int")


@dataclass(frozen=True)
class CameraInspection:
    outcome: str
    error: str | None
    normalized_yaw_delta: float | None = None
    pitch_delta: float | None = None
    direction_match: bool | None = None
    magnitude_match: bool | None = None

    def __post_init__(self) -> None:
        if self.outcome not in CAMERA_OUTCOMES:
            raise ValueError(f"unknown camera outcome: {self.outcome!r}")

    @property
    def valid(self) -> bool:
        return self.outcome == CAMERA_OK


def inspect_camera_change(
    before: CameraOrientationSnapshot,
    after: CameraOrientationSnapshot,
    execution: CameraActionExecution,
    *,
    yaw_tolerance: float,
    pitch_tolerance: float,
) -> CameraInspection:
    """Fail closed unless one accepted look produces the requested rotation."""

    yaw_tolerance = finite_angle(yaw_tolerance, "yaw_tolerance")
    pitch_tolerance = finite_angle(pitch_tolerance, "pitch_tolerance")
    if yaw_tolerance < 0 or pitch_tolerance < 0:
        raise ValueError("camera tolerances must be non-negative")
    if execution.action_type != "look":
        return CameraInspection(WRONG_ACTION_TYPE, "tested action must be look")
    if execution.tested_action_count == 0:
        return CameraInspection(TEST_ACTION_NOT_EXECUTED, "camera test action was not executed")
    if execution.tested_action_count != 1:
        return CameraInspection(MULTIPLE_TEST_ACTIONS, "exactly one camera test action is required")
    if not execution.translated_action_accepted:
        return CameraInspection(ACTION_REJECTED, "camera action translation was rejected")
    if (
        before.episode_id != execution.episode_id
        or after.episode_id != execution.episode_id
        or before.agent_id != execution.agent_id
        or after.agent_id != execution.agent_id
        or before.step_id != 0
        or execution.step_id != 1
        or after.step_id != execution.step_id
    ):
        return CameraInspection(STEP_IDENTITY_MISMATCH, "camera identity or step sequence is invalid")
    yaw_delta = normalized_angular_delta(after.yaw, before.yaw)
    pitch_delta = after.pitch - before.pitch
    if abs(yaw_delta) <= 1e-9:
        return CameraInspection(YAW_UNCHANGED, "observed yaw did not change", yaw_delta, pitch_delta, False, False)
    direction = yaw_delta * execution.requested_yaw > 0.0
    magnitude = abs(yaw_delta - execution.requested_yaw) <= yaw_tolerance
    if not direction:
        return CameraInspection(YAW_WRONG_DIRECTION, "observed yaw changed in the wrong direction", yaw_delta, pitch_delta, False, magnitude)
    if not magnitude:
        return CameraInspection(YAW_MAGNITUDE_MISMATCH, "observed yaw delta is outside tolerance", yaw_delta, pitch_delta, True, False)
    if abs(pitch_delta - execution.requested_pitch) > pitch_tolerance:
        return CameraInspection(PITCH_DRIFT_EXCESSIVE, "observed pitch drift is outside tolerance", yaw_delta, pitch_delta, True, True)
    return CameraInspection(CAMERA_OK, None, yaw_delta, pitch_delta, True, True)

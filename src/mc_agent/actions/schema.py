"""Strict, non-executable macro-action protocol."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any


ALLOWED_ACTIONS = {"wait", "look", "turn", "move_forward"}
ALLOWED_KEYS = {
    "action",
    "duration_ticks",
    "camera",
    "attack",
    "jump",
    "sprint",
    "reason",
}
ALLOWED_CAMERA_KEYS = {"pitch", "yaw"}


@dataclass(frozen=True)
class MacroAction:
    action: str = "wait"
    duration_ticks: int = 1
    camera_pitch: float = 0.0
    camera_yaw: float = 0.0
    attack: bool = False
    jump: bool = False
    sprint: bool = False
    reason: str = ""

    @classmethod
    def no_op(cls, reason: str = "") -> "MacroAction":
        return cls(action="wait", duration_ticks=1, reason=reason[:160])

    def to_log_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParseResult:
    action: MacroAction
    accepted: bool
    error: str | None = None


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def limit_macro_action(action: MacroAction) -> MacroAction:
    """Defense-in-depth limiter for actions constructed outside the parser."""
    try:
        if action.action not in ALLOWED_ACTIONS:
            raise ValueError("action outside whitelist")
        if type(action.duration_ticks) is not int:
            raise ValueError("duration must be integer")
        pitch = _clamp(_number(action.camera_pitch, "camera_pitch"), -30, 30)
        yaw = _clamp(_number(action.camera_yaw, "camera_yaw"), -30, 30)
        attack = _boolean(action.attack, "attack")
        jump = _boolean(action.jump, "jump")
        sprint = _boolean(action.sprint, "sprint")
        if not isinstance(action.reason, str):
            raise ValueError("reason must be string")
        return MacroAction(
            action=action.action,
            duration_ticks=int(_clamp(action.duration_ticks, 1, 40)),
            camera_pitch=pitch,
            camera_yaw=yaw,
            attack=attack,
            jump=jump,
            sprint=sprint,
            reason=action.reason[:160],
        )
    except (TypeError, ValueError) as error:
        return MacroAction.no_op(f"limited to no-op: {error}")


def parse_macro_action(raw: str) -> ParseResult:
    """Parse JSON strictly; any structural failure returns a one-tick no-op."""
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("action payload must be one JSON object")
        unknown = set(value) - ALLOWED_KEYS
        if unknown:
            raise ValueError(f"unknown fields: {sorted(unknown)}")

        action_name = value.get("action")
        if action_name not in ALLOWED_ACTIONS:
            raise ValueError("action is missing or outside the whitelist")

        duration = value.get("duration_ticks", 1)
        if type(duration) is not int:
            raise ValueError("duration_ticks must be an integer")
        duration = int(_clamp(duration, 1, 40))

        camera = value.get("camera", {})
        if not isinstance(camera, dict):
            raise ValueError("camera must be an object")
        unknown_camera = set(camera) - ALLOWED_CAMERA_KEYS
        if unknown_camera:
            raise ValueError(f"unknown camera fields: {sorted(unknown_camera)}")
        pitch = _clamp(_number(camera.get("pitch", 0.0), "camera.pitch"), -30, 30)
        yaw = _clamp(_number(camera.get("yaw", 0.0), "camera.yaw"), -30, 30)

        reason = value.get("reason", "")
        if not isinstance(reason, str):
            raise ValueError("reason must be a string")

        parsed = MacroAction(
            action=action_name,
            duration_ticks=duration,
            camera_pitch=pitch,
            camera_yaw=yaw,
            attack=_boolean(value.get("attack", False), "attack"),
            jump=_boolean(value.get("jump", False), "jump"),
            sprint=_boolean(value.get("sprint", False), "sprint"),
            reason=reason[:160],
        )
        return ParseResult(action=limit_macro_action(parsed), accepted=True)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        message = str(error)
        return ParseResult(
            action=MacroAction.no_op(reason=f"rejected: {message}"),
            accepted=False,
            error=message,
        )

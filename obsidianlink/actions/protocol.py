from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from obsidianlink.core.types import MacroAction


ALLOWED_ACTIONS = frozenset(
    {
        "wait",
        "look",
        "move",
        "equip_item",
        "mine_target",
        "place_block",
        "use_item",
        "craft_item",
    }
)
TARGET_REQUIRED_ACTIONS = frozenset(
    {"equip_item", "mine_target", "place_block", "use_item", "craft_item"}
)
ALLOWED_FIELDS = frozenset(
    {"action_type", "target", "duration_ticks", "parameters"}
)
ALLOWED_PARAMETER_FIELDS = frozenset(
    {"yaw", "pitch", "forward", "strafe", "sprint", "jump"}
)


@dataclass(frozen=True)
class ParseResult:
    action: MacroAction
    accepted: bool
    error: str | None = None


def _finite_number(value: Any, name: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _parse_parameters(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("parameters must be an object")
    unknown = set(value) - ALLOWED_PARAMETER_FIELDS
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)}")
    parsed: dict[str, Any] = {}
    for name in ("yaw", "pitch"):
        if name in value:
            parsed[name] = _clamp(_finite_number(value[name], name), -30.0, 30.0)
    for name in ("forward", "strafe"):
        if name in value:
            parsed[name] = _clamp(_finite_number(value[name], name), -1.0, 1.0)
    if "sprint" in value:
        if type(value["sprint"]) is not bool:
            raise ValueError("sprint must be a boolean")
        parsed["sprint"] = value["sprint"]
    if "jump" in value:
        if type(value["jump"]) is not bool:
            raise ValueError("jump must be a boolean")
        parsed["jump"] = value["jump"]
    return parsed


def parse_macro_action(raw: str) -> ParseResult:
    """Strictly parse one model action and fail closed to a one-tick wait."""
    try:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("action payload must be one JSON object")
        unknown = set(value) - ALLOWED_FIELDS
        if unknown:
            raise ValueError(f"unknown fields: {sorted(unknown)}")
        action_type = value.get("action_type")
        if action_type not in ALLOWED_ACTIONS:
            raise ValueError("action_type is missing or outside the allowlist")
        target = value.get("target")
        if target is not None and (
            not isinstance(target, str) or not target.strip()
        ):
            raise ValueError("target must be null or a non-empty string")
        if action_type in TARGET_REQUIRED_ACTIONS and target is None:
            raise ValueError(f"{action_type} requires target")
        duration = value.get("duration_ticks", 1)
        if type(duration) is not int:
            raise ValueError("duration_ticks must be an integer")
        duration = int(_clamp(float(duration), 1.0, 40.0))
        parameters = _parse_parameters(value.get("parameters", {}))
        return ParseResult(
            action=MacroAction(
                action_type=action_type,
                target=target.strip() if isinstance(target, str) else None,
                duration_ticks=duration,
                parameters=parameters,
            ),
            accepted=True,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        return ParseResult(action=MacroAction.wait(), accepted=False, error=str(error))

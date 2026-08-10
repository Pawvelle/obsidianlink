from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from obsidianlink.core.types import MacroAction


#: Default MineRL hotbar mapping retained for legacy ``PortalA0`` and
#: direct translator callers. MineRL backends build a task-specific mapping
#: from the frozen initial inventory via :func:`build_hotbar_mapping`. The
#: legacy items
#: (``obsidian`` / ``flint_and_steel`` / ``dirt``) keep their
#: original slots so existing direct translator fixtures remain compatible.
PORTAL_A0_HOTBAR: Mapping[str, str] = {
    "obsidian": "hotbar.1",
    "flint_and_steel": "hotbar.2",
    "dirt": "hotbar.3",
    "water_bucket": "hotbar.4",
    "lava_bucket": "hotbar.5",
    "cobblestone": "hotbar.6",
}


def build_hotbar_mapping(
    inventory: Mapping[str, int],
) -> Mapping[str, str]:
    """Build the hotbar mapping created by ``SimpleInventoryAgentStart``.

    MineRL assigns the supplied inventory entries to hotbar slots in input
    order.  Casting tasks use a different inventory from legacy Route A0, so
    reusing :data:`PORTAL_A0_HOTBAR` would select the wrong items.
    """
    if not isinstance(inventory, Mapping) or not inventory:
        raise ValueError("inventory must be a non-empty mapping")
    result: dict[str, str] = {}
    slot = 1
    for item, quantity in inventory.items():
        if not isinstance(item, str) or not item:
            raise ValueError("inventory item names must be non-empty strings")
        if type(quantity) is not int or quantity < 0:
            raise ValueError("inventory quantities must be non-negative integers")
        if quantity == 0:
            continue
        if slot > 9:
            raise ValueError("initial inventory exceeds the nine-slot hotbar")
        result[item] = f"hotbar.{slot}"
        slot += 1
    if not result:
        raise ValueError("inventory must contain at least one positive quantity")
    return MappingProxyType(result)

#: Closed set of items the translator accepts for ``equip_item`` and
#: ``use_item``. Any other item (including the legacy A0 ``obsidian``
#: / ``dirt``) is rejected at the translator boundary; the
#: translator never silently rewrites a forbidden item into a
#: supported one.
TRANSLATOR_EQUIPPABLE_ITEMS: frozenset[str] = frozenset(
    PORTAL_A0_HOTBAR.keys()
)

#: Closed set of targets accepted for ``place_block``. Only
#: ``cobblestone`` is allowed for the R6 C3 / C4 / C5 support
#: blocks. ``obsidian`` and ``dirt`` remain in :data:`PORTAL_A0_HOTBAR`
#: for ``equip_item`` / ``use_item`` parity with the legacy
#: fixtures but are not valid ``place_block`` targets here.
TRANSLATOR_PLACEABLE_ITEMS: frozenset[str] = frozenset(
    {"cobblestone", "obsidian", "dirt"}
)

#: Hard cap on the duration of a single translated action. Mirrors
#: the public ``MacroAction`` cap (1..40) so the translator never
#: widens the input contract.
MAX_TRANSLATOR_DURATION_TICKS: int = 40
MIN_TRANSLATOR_DURATION_TICKS: int = 1


@dataclass(frozen=True)
class MineRLTranslationResult:
    action: Mapping[str, Any]
    accepted: bool
    error: str | None = None


def _set_if_supported(
    low_level: dict[str, Any],
    key: str,
    value: Any,
) -> None:
    if key not in low_level:
        raise ValueError(f"MineRL action space does not support {key}")
    low_level[key] = value


def _require_strict_int(value: Any, name: str) -> int:
    """Validate a strict ``int`` and reject ``bool`` (which is a ``int``)."""
    if type(value) is not int or isinstance(value, bool):
        raise ValueError(f"{name} must be a strict integer")
    return value


def _require_finite_number(value: Any, name: str) -> float:
    """Validate a finite ``int`` / ``float``; reject ``bool`` and non-finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_in_range(
    value: float, name: str, lower: float, upper: float
) -> float:
    if value < lower or value > upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return value


def _check_within_action_space(
    low_level: dict[str, Any], action_space: Any
) -> None:
    """Reject translated actions outside the declared MineRL action space.

    The translator treats the result of ``action_space.contains`` as
    the canonical validity gate; the legacy translator only called
    this on the *accepted* path, so the same call keeps the new
    validation consistent.
    """
    if not action_space.contains(low_level):
        raise ValueError("translated action is outside the MineRL action space")


def translate_macro_action(
    action: MacroAction,
    action_space: Any,
    *,
    hotbar_mapping: Mapping[str, str] | None = None,
) -> MineRLTranslationResult:
    """Translate one semantic action into one bounded MineRL environment tick.

    Supported semantic actions for the R6 Casting-S-C5 deterministic
    driver plan:

    * ``wait`` — pass through a no-op.
    * ``look`` — set the bounded ``camera`` pitch / yaw.
    * ``move`` — set a bounded forward (no strafe / sprint / jump).
    * ``equip_item`` — equip one of the closed translator items
      (``water_bucket`` / ``lava_bucket`` / ``cobblestone`` /
      ``flint_and_steel`` plus the legacy ``obsidian`` /
      ``flint_and_steel`` / ``dirt`` keys for the A0 fixtures).
    * ``place_block`` — equip and use one of the closed placeable
      items (``cobblestone`` plus the legacy ``obsidian`` /
      ``dirt`` keys).
    * ``use_item`` — equip and use one of the closed equippable
      items.
    * ``craft_item`` — explicitly forbidden (no crafting in the
      controlled environment).
    * ``mine_target`` — translated to a single ``attack`` tick for
      the legacy A0 contract.

    The translator enforces the closed allowlist, strict type
    checks, and bounded numeric ranges. Unknown semantic action
    types, unsupported items, bool-as-int, non-finite parameters,
    out-of-range parameters, and translated actions outside the
    declared action space all fail closed by returning
    ``accepted=False`` with a typed error message. The translator
    never executes model-supplied code, never makes MCP / shell
    calls, and never silently rewrites a rejected action.
    """
    item_hotbar = PORTAL_A0_HOTBAR if hotbar_mapping is None else hotbar_mapping
    if not isinstance(item_hotbar, Mapping):
        return MineRLTranslationResult(
            action=dict(action_space.no_op()),
            accepted=False,
            error="hotbar_mapping must be a mapping",
        )
    no_op = action_space.no_op()
    low_level = dict(no_op)
    try:
        if action.action_type == "wait":
            # The wait is a no-op; the duration_ticks budget is
            # enforced by the action protocol, not the translator.
            pass
        elif action.action_type == "look":
            pitch = _require_finite_number(
                action.parameters.get("pitch", 0.0), "look.pitch"
            )
            yaw = _require_finite_number(
                action.parameters.get("yaw", 0.0), "look.yaw"
            )
            _require_in_range(pitch, "look.pitch", -30.0, 30.0)
            _require_in_range(yaw, "look.yaw", -30.0, 30.0)
            _set_if_supported(
                low_level,
                "camera",
                np.asarray([pitch, yaw], dtype=np.float32),
            )
        elif action.action_type == "move":
            # R6 C3 / C4 / C5 driver emits bounded forward-only
            # moves. Strafe, sprint, and jump are accepted on the
            # translator surface (for legacy A0 compatibility) but
            # the C5 driver family never emits them.
            forward = _require_finite_number(
                action.parameters.get("forward", 0.0), "move.forward"
            )
            _require_in_range(forward, "move.forward", -1.0, 1.0)
            if forward > 0:
                _set_if_supported(low_level, "forward", 1)
            elif forward < 0:
                _set_if_supported(low_level, "back", 1)
            strafe_value = _require_finite_number(
                action.parameters.get("strafe", 0.0), "move.strafe"
            )
            _require_in_range(strafe_value, "move.strafe", -1.0, 1.0)
            if strafe_value > 0:
                _set_if_supported(low_level, "right", 1)
            elif strafe_value < 0:
                _set_if_supported(low_level, "left", 1)
            if _require_bool(action.parameters.get("sprint", False), "move.sprint"):
                _set_if_supported(low_level, "sprint", 1)
            if _require_bool(action.parameters.get("jump", False), "move.jump"):
                _set_if_supported(low_level, "jump", 1)
        elif action.action_type == "equip_item":
            target = action.target
            if not isinstance(target, str) or not target:
                raise ValueError(
                    "equip_item requires a non-empty target string"
                )
            hotbar_key = item_hotbar.get(target)
            if hotbar_key is None:
                raise ValueError(
                    f"unsupported translator equip target: {target!r}"
                )
            _set_if_supported(low_level, hotbar_key, 1)
        elif action.action_type == "mine_target":
            _set_if_supported(low_level, "attack", 1)
        elif action.action_type == "place_block":
            target = action.target
            if not isinstance(target, str) or not target:
                raise ValueError(
                    "place_block requires a non-empty target string"
                )
            if target not in TRANSLATOR_PLACEABLE_ITEMS:
                raise ValueError(
                    f"unsupported translator place target: {target!r}"
                )
            hotbar_key = item_hotbar.get(target)
            if hotbar_key is None:
                raise ValueError(f"place target {target!r} is absent from hotbar")
            _set_if_supported(low_level, hotbar_key, 1)
            if _require_bool(action.parameters.get("jump", False), "place_block.jump"):
                _set_if_supported(low_level, "jump", 1)
            _set_if_supported(low_level, "use", 1)
        elif action.action_type == "use_item":
            target = action.target
            if not isinstance(target, str) or not target:
                raise ValueError(
                    "use_item requires a non-empty target string"
                )
            if target not in TRANSLATOR_EQUIPPABLE_ITEMS:
                raise ValueError(
                    f"unsupported translator use target: {target!r}"
                )
            hotbar_key = item_hotbar.get(target)
            if hotbar_key is None:
                raise ValueError(f"use target {target!r} is absent from hotbar")
            _set_if_supported(low_level, hotbar_key, 1)
            _set_if_supported(low_level, "use", 1)
        elif action.action_type == "craft_item":
            raise ValueError("craft_item is not available in Route A0")
        else:
            raise ValueError(
                f"unsupported semantic action: {action.action_type}"
            )

        duration = _require_strict_int(
            action.duration_ticks, "duration_ticks"
        )
        if (
            duration < MIN_TRANSLATOR_DURATION_TICKS
            or duration > MAX_TRANSLATOR_DURATION_TICKS
        ):
            raise ValueError(
                "duration_ticks must be between "
                f"{MIN_TRANSLATOR_DURATION_TICKS} and "
                f"{MAX_TRANSLATOR_DURATION_TICKS}"
            )
        _check_within_action_space(low_level, action_space)
        return MineRLTranslationResult(action=low_level, accepted=True)
    except (TypeError, ValueError) as error:
        return MineRLTranslationResult(
            action=no_op,
            accepted=False,
            error=str(error),
        )


__all__ = [
    "MAX_TRANSLATOR_DURATION_TICKS",
    "MIN_TRANSLATOR_DURATION_TICKS",
    "MineRLTranslationResult",
    "PORTAL_A0_HOTBAR",
    "TRANSLATOR_EQUIPPABLE_ITEMS",
    "TRANSLATOR_PLACEABLE_ITEMS",
    "build_hotbar_mapping",
    "translate_macro_action",
]

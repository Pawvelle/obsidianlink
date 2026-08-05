from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from obsidianlink.core.types import MacroAction


PORTAL_A0_HOTBAR = {
    "obsidian": "hotbar.1",
    "flint_and_steel": "hotbar.2",
    "dirt": "hotbar.3",
}


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


def translate_macro_action(
    action: MacroAction,
    action_space: Any,
) -> MineRLTranslationResult:
    """Translate one semantic action into one bounded MineRL environment tick."""
    no_op = action_space.no_op()
    low_level = dict(no_op)
    try:
        if action.action_type == "wait":
            pass
        elif action.action_type == "look":
            _set_if_supported(
                low_level,
                "camera",
                np.asarray(
                    [
                        float(action.parameters.get("pitch", 0.0)),
                        float(action.parameters.get("yaw", 0.0)),
                    ],
                    dtype=np.float32,
                ),
            )
        elif action.action_type == "move":
            forward = float(action.parameters.get("forward", 0.0))
            strafe = float(action.parameters.get("strafe", 0.0))
            if forward > 0:
                _set_if_supported(low_level, "forward", 1)
            elif forward < 0:
                _set_if_supported(low_level, "back", 1)
            if strafe > 0:
                _set_if_supported(low_level, "right", 1)
            elif strafe < 0:
                _set_if_supported(low_level, "left", 1)
            if bool(action.parameters.get("sprint", False)):
                _set_if_supported(low_level, "sprint", 1)
            if bool(action.parameters.get("jump", False)):
                _set_if_supported(low_level, "jump", 1)
        elif action.action_type == "equip_item":
            hotbar_key = PORTAL_A0_HOTBAR.get(action.target or "")
            if hotbar_key is None:
                raise ValueError(f"unsupported A0 inventory target: {action.target}")
            _set_if_supported(low_level, hotbar_key, 1)
        elif action.action_type == "mine_target":
            _set_if_supported(low_level, "attack", 1)
        elif action.action_type == "place_block":
            if action.target not in {"obsidian", "dirt"}:
                raise ValueError(f"unsupported A0 place target: {action.target}")
            hotbar_key = PORTAL_A0_HOTBAR[action.target]
            _set_if_supported(low_level, hotbar_key, 1)
            if bool(action.parameters.get("jump", False)):
                _set_if_supported(low_level, "jump", 1)
            _set_if_supported(low_level, "use", 1)
        elif action.action_type == "use_item":
            hotbar_key = PORTAL_A0_HOTBAR.get(action.target or "")
            if hotbar_key is None:
                raise ValueError(f"unsupported A0 use target: {action.target}")
            _set_if_supported(low_level, hotbar_key, 1)
            _set_if_supported(low_level, "use", 1)
        elif action.action_type == "craft_item":
            raise ValueError("craft_item is not available in Route A0")
        else:
            raise ValueError(f"unsupported semantic action: {action.action_type}")

        if not action_space.contains(low_level):
            raise ValueError("translated action is outside the MineRL action space")
        return MineRLTranslationResult(action=low_level, accepted=True)
    except (TypeError, ValueError) as error:
        return MineRLTranslationResult(
            action=no_op,
            accepted=False,
            error=str(error),
        )

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from obsidianlink.core.types import MacroAction


PORTAL_ENV_NAME = "ObsidianLinkPortalA0-v0"
PORTAL_A1_ENV_NAME = "ObsidianLinkPortalA1-v0"

# Phase 3 A0 inventory: obsidian / flint_and_steel / dirt → hotbar 1..3.
PORTAL_A0_HOTBAR = {
    "obsidian": "hotbar.1",
    "flint_and_steel": "hotbar.2",
    "dirt": "hotbar.3",
}
# Phase 4 A1 inventory: diamond_pickaxe / flint_and_steel / dirt.
# The A1 spec has no free obsidian in the initial inventory; the agent
# must mine it from a fixed nearby deposit. The hotbar order is fixed
# by the SimpleInventoryAgentStart handler in the EnvSpec, so the
# mapping must match that order exactly.
PORTAL_A1_HOTBAR = {
    "diamond_pickaxe": "hotbar.1",
    "flint_and_steel": "hotbar.2",
    "dirt": "hotbar.3",
}

# Items a driver can ``mine_target`` semantically. The current A1
# slice only ever targets "obsidian", but the translator stays
# permissive so tests can opt into "stone" or "dirt" via the same
# allowlist.
PORTAL_A1_MINE_TARGETS = frozenset({"obsidian", "stone", "dirt"})


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


def _hotbar_for_env(env_name: str | None) -> Mapping[str, str]:
    if env_name == PORTAL_A1_ENV_NAME:
        return PORTAL_A1_HOTBAR
    if env_name is None or env_name == PORTAL_ENV_NAME:
        return PORTAL_A0_HOTBAR
    raise ValueError(f"unknown MineRL portal env_name: {env_name!r}")


def translate_macro_action(
    action: MacroAction,
    action_space: Any,
    *,
    env_name: str | None = PORTAL_ENV_NAME,
) -> MineRLTranslationResult:
    """Translate one semantic action into one bounded MineRL environment tick.

    ``env_name`` selects between the A0 inventory (obsidian /
    flint_and_steel / dirt) and the A1 inventory
    (diamond_pickaxe / flint_and_steel / dirt). ``mine_target`` is
    treated the same in both modes — a one-tick ``attack=1`` — but the
    A1 backend additionally records intent when the target is
    ``obsidian``.
    """
    hotbar = _hotbar_for_env(env_name)
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
            hotbar_key = hotbar.get(action.target or "")
            if hotbar_key is None:
                raise ValueError(
                    f"unsupported inventory target for {env_name}: {action.target}"
                )
            _set_if_supported(low_level, hotbar_key, 1)
        elif action.action_type == "mine_target":
            if env_name == PORTAL_A1_ENV_NAME and (
                action.target not in PORTAL_A1_MINE_TARGETS
            ):
                raise ValueError(
                    f"unsupported A1 mine_target: {action.target}"
                )
            _set_if_supported(low_level, "attack", 1)
        elif action.action_type == "place_block":
            if action.target not in hotbar:
                raise ValueError(
                    f"unsupported place target for {env_name}: {action.target}"
                )
            hotbar_key = hotbar[action.target]
            _set_if_supported(low_level, hotbar_key, 1)
            if bool(action.parameters.get("jump", False)):
                _set_if_supported(low_level, "jump", 1)
            _set_if_supported(low_level, "use", 1)
        elif action.action_type == "use_item":
            hotbar_key = hotbar.get(action.target or "")
            if hotbar_key is None:
                raise ValueError(
                    f"unsupported use target for {env_name}: {action.target}"
                )
            _set_if_supported(low_level, hotbar_key, 1)
            _set_if_supported(low_level, "use", 1)
        elif action.action_type == "craft_item":
            raise ValueError("craft_item is not available in Route A0/A1")
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

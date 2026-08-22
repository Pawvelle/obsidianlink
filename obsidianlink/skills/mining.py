"""Resource collection skills."""

from __future__ import annotations

from obsidianlink.agents.memory import AgentMemory
from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.skills.base import SkillResult, bounded_int

_LOG_NAMES = (
    "acacia_log",
    "birch_log",
    "dark_oak_log",
    "jungle_log",
    "oak_log",
    "spruce_log",
    "log",
    "wood",
)


def log_count(inventory: dict[str, int]) -> int:
    total = 0
    for name, qty in (inventory or {}).items():
        key = str(name).strip().lower().split(":", 1)[-1]
        if key in _LOG_NAMES or key.endswith("_log"):
            try:
                total += int(qty)
            except (TypeError, ValueError):
                continue
    return total


def _tree_horizontal_offset(frame: object) -> float | None:
    """Estimate a leafy tree direction from the upper half of an RGB POV.

    This is deliberately a small visual servo, not semantic world truth. It
    favors green vertical mass near the crosshair and ignores the lower grass
    field. Returned values are normalized to ``[-1, 1]``.
    """
    try:
        import numpy as np

        rgb = np.asarray(frame)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return None
        height, width = rgb.shape[:2]
        red = rgb[:, :, 0].astype(float)
        green = rgb[:, :, 1].astype(float)
        blue = rgb[:, :, 2].astype(float)
        mask = (green > red * 1.12) & (green > blue * 1.08) & (green > 45)
        mask[int(height * 0.50) :, :] = False
        window = max(9, width // 42)
        scores = np.convolve(mask.sum(axis=0), np.ones(window), mode="same")
        center = (width - 1) / 2.0
        distance_penalty = 1.0 + np.abs(np.arange(width) - center) / (width * 0.16)
        weighted = scores / distance_penalty
        target = int(np.argmax(weighted))
        if float(scores[target]) < max(18.0, height * 0.12):
            return None
        return max(-1.0, min(1.0, (target - center) / center))
    except (TypeError, ValueError, AttributeError):
        return None


def _trunk_horizontal_offset(frame: object) -> float | None:
    """Estimate the direction of a nearby vertical brown trunk mass."""
    try:
        import numpy as np

        rgb = np.asarray(frame)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return None
        height, width = rgb.shape[:2]
        red = rgb[:, :, 0].astype(float)
        green = rgb[:, :, 1].astype(float)
        blue = rgb[:, :, 2].astype(float)
        mask = (
            (red > green * 1.18)
            & (green > blue * 1.55)
            & (red > 25)
            & (red < 170)
        )
        # Ignore the hotbar/hand and most dirt immediately below the player.
        mask[int(height * 0.76) :, :] = False
        window = max(13, width // 26)
        scores = np.convolve(mask.sum(axis=0), np.ones(window), mode="same")
        center = (width - 1) / 2.0
        distance_penalty = 1.0 + np.abs(np.arange(width) - center) / (width * 0.40)
        weighted = scores / distance_penalty
        target = int(np.argmax(weighted))
        if float(scores[target]) < max(80.0, height * window * 0.07):
            return None
        return max(-1.0, min(1.0, (target - center) / center))
    except (TypeError, ValueError, AttributeError):
        return None


def _lava_ahead(frame: object) -> bool:
    """Conservative near-field lava cue for natural-terrain smoke safety."""
    try:
        import numpy as np

        rgb = np.asarray(frame)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return False
        height, width = rgb.shape[:2]
        red = rgb[:, :, 0].astype(float)
        green = rgb[:, :, 1].astype(float)
        blue = rgb[:, :, 2].astype(float)
        orange = (
            (red > 150)
            & (green > 35)
            & (green < 180)
            & (blue < 60)
            & (red > green * 1.30)
        )
        near = orange[
            int(height * 0.45) :,
            int(width * 0.25) : int(width * 0.75),
        ]
        return bool(near.size and float(near.mean()) > 0.01)
    except (TypeError, ValueError, AttributeError):
        return False


def _trunk_under_crosshair(frame: object) -> bool:
    """Return true when a brown trunk surface fills the aiming region."""
    try:
        import numpy as np

        rgb = np.asarray(frame)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return False
        height, width = rgb.shape[:2]
        red = rgb[:, :, 0].astype(float)
        green = rgb[:, :, 1].astype(float)
        blue = rgb[:, :, 2].astype(float)
        brown = (
            (red > green * 1.18)
            # Bark in the installed texture pack has much less blue than
            # dirt. This prevents a dirt ledge from entering attack lock.
            & (green > blue * 1.55)
            & (red > 25)
            & (red < 170)
        )
        aim_region = brown[
            int(height * 0.48) : int(height * 0.52),
            int(width * 0.49) : int(width * 0.51),
        ]
        return bool(aim_region.size and float(aim_region.mean()) > 0.15)
    except (TypeError, ValueError, AttributeError):
        return False


class CollectWoodSkill:
    name = "collect_wood"
    description = "Find, approach, and break trees until the requested log count is in inventory."

    def execute(
        self,
        controller: MinecraftController,
        memory: AgentMemory,
        arguments: dict[str, object],
    ) -> SkillResult:
        quantity = bounded_int(arguments.get("quantity"), default=3, minimum=1, maximum=16)
        budget = bounded_int(arguments.get("max_steps"), default=240, minimum=8, maximum=800)
        start = controller.steps
        engaged_ticks = 0
        target_acquisitions = 0
        search_turns = 0
        alignment_turns = 0
        approach_attacks = 0
        contact_attacks = 0
        contact_burst_remaining = 0
        hazard_turns = 0
        previous_logs = log_count(dict(controller.observe().inventory or {}))
        while controller.steps - start < budget and not controller.exhausted:
            observation = controller.observe()
            current_logs = log_count(dict(observation.inventory or {}))
            if current_logs >= quantity:
                memory.update_state(observation)
                return SkillResult(
                    True,
                    f"collected at least {quantity} logs",
                    controller.steps - start,
                    metadata={
                        "search_turns": search_turns,
                        "alignment_turns": alignment_turns,
                        "target_acquisitions": target_acquisitions,
                        "approach_attacks": approach_attacks,
                        "contact_attacks": contact_attacks,
                        "hazard_turns": hazard_turns,
                        "logs": current_logs,
                        "target_logs": quantity,
                    },
                )
            if current_logs > previous_logs:
                previous_logs = current_logs
                engaged_ticks = 0
                if current_logs < quantity:
                    controller.step(Action(ActionType.CAMERA, pitch=-15.0))
                continue
            if contact_burst_remaining > 0:
                approach_attacks += 1
                contact_attacks += 1
                contact_burst_remaining -= 1
                controller.step(Action(ActionType.ATTACK, dx=1, jump=True))
                continue
            trunk_offset = _trunk_horizontal_offset(observation.frame)
            offset = (
                trunk_offset
                if trunk_offset is not None
                else _tree_horizontal_offset(observation.frame)
            )
            if _lava_ahead(observation.frame):
                hazard_turns += 1
                engaged_ticks = 0
                controller.step(Action(ActionType.CAMERA, yaw=75.0))
                continue
            if _trunk_under_crosshair(observation.frame):
                engaged_ticks = max(1, engaged_ticks)
                contact_burst_remaining = 19
                approach_attacks += 1
                contact_attacks += 1
                controller.step(Action(ActionType.ATTACK, dx=1, jump=True))
                continue
            if engaged_ticks == 0:
                if observation.frame is None:
                    target_acquisitions += 1
                    engaged_ticks = 1
                    controller.step(Action(ActionType.ATTACK, dx=1))
                    continue
                if offset is None:
                    search_turns += 1
                    controller.step(Action(ActionType.CAMERA, yaw=30.0))
                    continue
                if abs(offset) > 0.04:
                    alignment_turns += 1
                    controller.step(Action(ActionType.CAMERA, yaw=offset * 35.0))
                    continue
                # A trunk (preferred) or foliage candidate is centered. The
                # agent can press forward and attack in the same tick.
                target_acquisitions += 1
                engaged_ticks = 1

            if trunk_offset is not None and abs(trunk_offset) > 0.025:
                alignment_turns += 1
                controller.step(Action(ActionType.CAMERA, yaw=trunk_offset * 28.0))
                continue

            approach_attacks += 1
            controller.step(
                Action(
                    ActionType.ATTACK,
                    dx=1,
                    jump=True,
                )
            )
            engaged_ticks += 1
            if (
                engaged_ticks > 72
                and controller.steps - start < budget
                and not controller.exhausted
            ):
                # The candidate was likely foliage or unreachable terrain.
                # Reset pitch and rotate before acquiring another target.
                controller.step(Action(ActionType.CAMERA, yaw=35.0))
                engaged_ticks = 0
        memory.update_state(controller.observe())
        current = log_count(memory.inventory)
        success = current >= quantity
        return SkillResult(
            success,
            (
                f"collected at least {quantity} logs"
                if success
                else f"wood collection budget ended with {current}/{quantity} logs"
            ),
            controller.steps - start,
            metadata={
                "search_turns": search_turns,
                "alignment_turns": alignment_turns,
                "target_acquisitions": target_acquisitions,
                "approach_attacks": approach_attacks,
                "contact_attacks": contact_attacks,
                "hazard_turns": hazard_turns,
                "logs": current,
                "target_logs": quantity,
            },
        )


class MineBlockSkill:
    name = "mine_block"
    description = "Break the block currently under the crosshair for a bounded duration."

    def execute(
        self,
        controller: MinecraftController,
        memory: AgentMemory,
        arguments: dict[str, object],
    ) -> SkillResult:
        ticks = bounded_int(arguments.get("ticks"), default=20, minimum=1, maximum=320)
        target_item = str(arguments.get("item", "")).strip()
        approach = bool(arguments.get("approach", False))
        pickup_steps = bounded_int(
            arguments.get("pickup_steps"), default=0, minimum=0, maximum=32
        )
        settle_steps = bounded_int(
            arguments.get("settle_steps"), default=0, minimum=0, maximum=16
        )
        start = controller.steps
        before = dict(controller.observe().inventory or {})
        while controller.steps - start < ticks and not controller.exhausted:
            observation = controller.observe()
            if target_item and int(
                (observation.inventory or {}).get(target_item, 0) or 0
            ) > int(before.get(target_item, 0) or 0):
                break
            controller.step(
                Action(
                    ActionType.ATTACK,
                    dx=1 if approach else 0,
                    jump=approach,
                )
            )
        for _ in range(pickup_steps):
            observation = controller.observe()
            if target_item and int(
                (observation.inventory or {}).get(target_item, 0) or 0
            ) > int(before.get(target_item, 0) or 0):
                break
            if controller.exhausted:
                break
            controller.step(Action(ActionType.MOVE, dx=1, jump=True))
        for _ in range(settle_steps):
            observation = controller.observe()
            if target_item and int(
                (observation.inventory or {}).get(target_item, 0) or 0
            ) > int(before.get(target_item, 0) or 0):
                break
            if controller.exhausted:
                break
            controller.step(Action(ActionType.WAIT))
        memory.update_state(controller.observe())
        changed = (
            int(memory.inventory.get(target_item, 0) or 0)
            > int(before.get(target_item, 0) or 0)
            if target_item
            else memory.inventory != before
        )
        return SkillResult(
            changed,
            "inventory changed after mining" if changed else "no mined item observed",
            controller.steps - start,
            metadata={
                "inventory_before": before,
                "inventory_after": dict(memory.inventory),
            },
        )


__all__ = [
    "CollectWoodSkill",
    "MineBlockSkill",
    "_lava_ahead",
    "_trunk_under_crosshair",
    "_trunk_horizontal_offset",
    "_tree_horizontal_offset",
    "log_count",
]

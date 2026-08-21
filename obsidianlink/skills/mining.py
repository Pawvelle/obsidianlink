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
)


def log_count(inventory: dict[str, int]) -> int:
    return sum(int(inventory.get(name, 0) or 0) for name in _LOG_NAMES)


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
        cycle = 0
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
                )
            if current_logs > previous_logs:
                previous_logs = current_logs
                controller.step(Action(ActionType.CAMERA, pitch=-12.0))
                continue
            offset = _tree_horizontal_offset(observation.frame)
            if offset is None:
                phase = (controller.steps - start) % 24
                if observation.frame is None and phase < 16:
                    controller.step(Action(ActionType.ATTACK))
                elif phase < 20:
                    controller.step(Action(ActionType.MOVE, dx=1, jump=True))
                else:
                    controller.step(Action(ActionType.CAMERA, yaw=30.0))
            else:
                if abs(offset) > 0.10:
                    controller.step(Action(ActionType.CAMERA, yaw=offset * 35.0))
                elif cycle % 5 in {0, 1, 2}:
                    controller.step(Action(ActionType.MOVE, dx=1, jump=True))
                else:
                    controller.step(Action(ActionType.ATTACK))
                cycle += 1
        memory.update_state(controller.observe())
        current = log_count(memory.inventory)
        return SkillResult(
            False,
            f"wood collection budget ended with {current}/{quantity} logs",
            controller.steps - start,
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
        ticks = bounded_int(arguments.get("ticks"), default=20, minimum=1, maximum=120)
        start = controller.steps
        before = dict(controller.observe().inventory or {})
        while controller.steps - start < ticks and not controller.exhausted:
            controller.step(Action(ActionType.ATTACK))
        memory.update_state(controller.observe())
        changed = memory.inventory != before
        return SkillResult(
            changed,
            "inventory changed after mining" if changed else "no mined item observed",
            controller.steps - start,
        )


__all__ = [
    "CollectWoodSkill",
    "MineBlockSkill",
    "_tree_horizontal_offset",
    "log_count",
]

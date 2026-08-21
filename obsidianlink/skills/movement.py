"""Reusable movement/exploration skills."""

from __future__ import annotations

from obsidianlink.agents.memory import AgentMemory
from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.skills.base import SkillResult, bounded_int


def _frame_changed(before: object, after: object) -> bool | None:
    if before is None or after is None:
        return None
    try:
        import numpy as np

        first = np.asarray(before)
        second = np.asarray(after)
        if first.shape != second.shape or first.size == 0:
            return None
        difference = np.mean(np.abs(first.astype(float) - second.astype(float)))
        return bool(float(difference) > 0.5)
    except (TypeError, ValueError, AttributeError):
        return None


class MoveForwardSkill:
    name = "move_forward"
    description = (
        "Move forward for a bounded number of Minecraft ticks, optionally "
        "jumping over terrain."
    )

    def execute(
        self,
        controller: MinecraftController,
        memory: AgentMemory,
        arguments: dict[str, object],
    ) -> SkillResult:
        ticks = bounded_int(arguments.get("ticks"), default=12, minimum=1, maximum=96)
        jump = bool(arguments.get("jump", True))
        start = controller.steps
        before = controller.observe().frame
        while controller.steps - start < ticks and not controller.exhausted:
            controller.step(
                Action(
                    ActionType.MOVE,
                    dx=1,
                    jump=jump and (controller.steps - start) % 8 == 0,
                )
            )
        observation = controller.observe()
        memory.update_state(observation)
        executed = controller.steps - start
        changed = _frame_changed(before, observation.frame)
        return SkillResult(
            executed == ticks,
            f"executed {executed}/{ticks} forward movement ticks",
            executed,
            metadata={"frame_changed": changed, "jump_enabled": jump},
        )


class ExploreAreaSkill:
    name = "explore_area"
    description = "Scan and move through the nearby area for a bounded number of ticks."

    def execute(
        self,
        controller: MinecraftController,
        memory: AgentMemory,
        arguments: dict[str, object],
    ) -> SkillResult:
        budget = bounded_int(arguments.get("steps"), default=24, minimum=1, maximum=96)
        start = controller.steps
        while controller.steps - start < budget and not controller.exhausted:
            phase = (controller.steps - start) % 8
            if phase in {0, 4}:
                controller.step(Action(ActionType.CAMERA, yaw=35.0))
            else:
                controller.step(Action(ActionType.MOVE, dx=1, jump=True))
        memory.update_state(controller.observe())
        return SkillResult(True, "explored nearby area", controller.steps - start)


__all__ = ["ExploreAreaSkill", "MoveForwardSkill", "_frame_changed"]

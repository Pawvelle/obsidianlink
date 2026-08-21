"""Reusable movement/exploration skills."""

from __future__ import annotations

from obsidianlink.agents.memory import AgentMemory
from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.skills.base import SkillResult, bounded_int


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


__all__ = ["ExploreAreaSkill"]

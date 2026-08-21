"""Minimal native-use building skill."""

from __future__ import annotations

from obsidianlink.agents.memory import AgentMemory
from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.skills.base import SkillResult, bounded_int


def select_hotbar_item(
    controller: MinecraftController, item: str
) -> tuple[bool, int]:
    start = controller.steps
    if controller.observe().selected_item == item:
        return True, 0
    for slot in range(1, 10):
        if controller.exhausted:
            break
        observation = controller.step(Action(ActionType.HOTBAR, target=str(slot)))
        if observation.selected_item == item:
            return True, controller.steps - start
    return False, controller.steps - start


class BuildStructureSkill:
    name = "build_structure"
    description = "Select an inventory block and place a small bounded number with native use."

    def execute(
        self,
        controller: MinecraftController,
        memory: AgentMemory,
        arguments: dict[str, object],
    ) -> SkillResult:
        item = str(arguments.get("item", "")).strip()
        count = bounded_int(arguments.get("count"), default=1, minimum=1, maximum=16)
        start = controller.steps
        if not item or int((controller.observe().inventory or {}).get(item, 0) or 0) < 1:
            return SkillResult(False, f"missing build item: {item or '<empty>'}", 0)
        selected, _ = select_hotbar_item(controller, item)
        if not selected:
            memory.update_state(controller.observe())
            return SkillResult(False, f"could not select {item}", controller.steps - start)
        for index in range(count):
            if controller.exhausted:
                break
            controller.step(Action(ActionType.USE, sneak=True))
            if index + 1 < count and not controller.exhausted:
                controller.step(Action(ActionType.CAMERA, yaw=12.0))
        memory.update_state(controller.observe())
        return SkillResult(True, f"placed up to {count} {item}", controller.steps - start)


__all__ = ["BuildStructureSkill", "select_hotbar_item"]

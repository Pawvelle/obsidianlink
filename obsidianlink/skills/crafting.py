"""Named-recipe crafting through MineDojo event-level craft/place commands."""

from __future__ import annotations

from obsidianlink.agents.memory import AgentMemory
from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.skills.base import SkillResult
from obsidianlink.skills.mining import log_count


def _qty(controller: MinecraftController, *names: str) -> int:
    inventory = dict(controller.observe().inventory or {})
    total = 0
    for name in names:
        total += int(inventory.get(name, 0) or 0)
    return total


class CraftItemSkill:
    name = "craft_item"
    description = (
        "Craft wooden_pickaxe from logs using MineDojo recipe commands: "
        "planks, sticks, crafting_table, place the table, then the pickaxe."
    )

    def execute(
        self,
        controller: MinecraftController,
        memory: AgentMemory,
        arguments: dict[str, object],
    ) -> SkillResult:
        item = str(arguments.get("item", "wooden_pickaxe")).strip().lower()
        start = controller.steps
        if item != "wooden_pickaxe":
            return SkillResult(False, f"unsupported prototype recipe: {item}", 0)
        if _qty(controller, "wooden_pickaxe") >= 1:
            memory.update_state(controller.observe())
            return SkillResult(True, "wooden pickaxe already present", 0)
        if log_count(dict(controller.observe().inventory or {})) < 2:
            return SkillResult(False, "need at least 2 logs before crafting", 0)

        for recipe in ("planks", "oak_planks", "stick", "crafting_table"):
            if controller.exhausted:
                break
            controller.step(Action(ActionType.CRAFT, target=recipe))

        if _qty(controller, "crafting_table") < 1:
            memory.update_state(controller.observe())
            return SkillResult(
                False, "crafting table recipe was not observed", controller.steps - start
            )

        controller.step(Action(ActionType.PLACE, target="crafting_table", sneak=True))
        if _qty(controller, "crafting_table") >= 1:
            memory.update_state(controller.observe())
            return SkillResult(
                False, "crafting table placement was not observed", controller.steps - start
            )

        controller.step(Action(ActionType.CRAFT, target="table:wooden_pickaxe"))
        memory.update_state(controller.observe())
        success = _qty(controller, "wooden_pickaxe") >= 1
        return SkillResult(
            success,
            "crafted wooden pickaxe" if success else "wooden pickaxe recipe was not observed",
            controller.steps - start,
        )


__all__ = ["CraftItemSkill"]

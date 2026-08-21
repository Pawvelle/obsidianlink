"""Inventory-grounded crafting through Minecraft's real GUI controls.

MineRL 1.0.2's MCP-Reborn bridge crashes on structured ``craft none``
commands. This skill therefore uses the fixed 640x360 inventory and crafting
table GUIs. Mouse/key details remain below the skill boundary and are never
part of the LLM planner vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass

from obsidianlink.agents.memory import AgentMemory
from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.skills.base import SkillResult
from obsidianlink.skills.mining import log_count

_PLANK_NAMES = (
    "acacia_planks",
    "birch_planks",
    "dark_oak_planks",
    "jungle_planks",
    "oak_planks",
    "spruce_planks",
)
_CURSOR_CENTER = (320, 180)
_MOUSE_UNITS_PER_DEGREE = 2400.0 / 360.0


def _count(inventory: dict[str, int], names: tuple[str, ...]) -> int:
    return sum(int(inventory.get(name, 0) or 0) for name in names)


def _qty(controller: MinecraftController, item: str) -> int:
    return int((controller.observe().inventory or {}).get(item, 0) or 0)


def _find_log_slot(controller: MinecraftController) -> int | None:
    for slot in range(1, 10):
        observation = controller.step(Action(ActionType.HOTBAR, target=str(slot)))
        selected = observation.selected_item or ""
        if selected.endswith("_log"):
            return slot
    return None


def _hotbar_xy(slot: int) -> tuple[int, int]:
    return 240 + 18 * (slot - 1), 239


@dataclass
class _GuiDriver:
    controller: MinecraftController
    x: int = _CURSOR_CENTER[0]
    y: int = _CURSOR_CENTER[1]

    def move_to(self, x: int, y: int) -> None:
        self.controller.step(
            Action(
                ActionType.CAMERA,
                yaw=(x - self.x) / _MOUSE_UNITS_PER_DEGREE,
                pitch=(y - self.y) / _MOUSE_UNITS_PER_DEGREE,
            )
        )
        self.x, self.y = x, y

    def left_click(self, point: tuple[int, int]) -> None:
        self.move_to(*point)
        self.controller.step(Action(ActionType.ATTACK))

    def right_click(self, point: tuple[int, int]) -> None:
        self.move_to(*point)
        self.controller.step(Action(ActionType.USE))


def _personal_crafting(
    controller: MinecraftController,
    *,
    log_slot: int,
    plank_slot: int,
    table_slot: int,
    stick_slot: int,
) -> None:
    controller.step(Action(ActionType.INVENTORY))
    gui = _GuiDriver(controller)
    log = _hotbar_xy(log_slot)
    planks = _hotbar_xy(plank_slot)
    table = _hotbar_xy(table_slot)
    sticks = _hotbar_xy(stick_slot)
    craft_a = (330, 116)
    craft_b = (348, 116)
    craft_c = (330, 134)
    craft_d = (348, 134)
    result = (386, 125)

    for _ in range(3):
        gui.left_click(log)
        gui.right_click(craft_a)
        gui.left_click(log)
        gui.left_click(result)
        gui.left_click(planks)

    gui.left_click(planks)
    for cell in (craft_a, craft_b, craft_c, craft_d):
        gui.right_click(cell)
    gui.left_click(planks)
    gui.left_click(result)
    gui.left_click(table)

    gui.left_click(planks)
    gui.right_click(craft_a)
    gui.right_click(craft_c)
    gui.left_click(planks)
    gui.left_click(result)
    gui.left_click(sticks)
    controller.step(Action(ActionType.INVENTORY))


def _table_crafting(
    controller: MinecraftController,
    *,
    plank_slot: int,
    stick_slot: int,
    pickaxe_slot: int,
) -> None:
    gui = _GuiDriver(controller)
    planks = _hotbar_xy(plank_slot)
    sticks = _hotbar_xy(stick_slot)
    pickaxe = _hotbar_xy(pickaxe_slot)
    top_row = ((262, 114), (280, 114), (298, 114))
    handle = ((280, 132), (280, 150))
    result = (356, 132)

    gui.left_click(planks)
    for cell in top_row:
        gui.right_click(cell)
    gui.left_click(planks)
    gui.left_click(sticks)
    for cell in handle:
        gui.right_click(cell)
    gui.left_click(sticks)
    gui.left_click(result)
    gui.left_click(pickaxe)
    controller.step(Action(ActionType.INVENTORY))


class CraftItemSkill:
    name = "craft_item"
    description = (
        "Craft an item in the real inventory GUI; wooden_pickaxe includes planks, "
        "sticks, crafting-table placement, and the 3x3 recipe."
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
        if log_count(dict(controller.observe().inventory or {})) < 3:
            return SkillResult(False, "need at least 3 logs before crafting", 0)

        log_slot = _find_log_slot(controller)
        if log_slot is None:
            memory.update_state(controller.observe())
            return SkillResult(False, "logs were not selectable in hotbar", controller.steps - start)
        free_slots = [slot for slot in range(1, 10) if slot not in {1, log_slot}]
        if len(free_slots) < 4:
            return SkillResult(False, "not enough known hotbar slots", controller.steps - start)
        plank_slot, table_slot, stick_slot, pickaxe_slot = free_slots[:4]

        _personal_crafting(
            controller,
            log_slot=log_slot,
            plank_slot=plank_slot,
            table_slot=table_slot,
            stick_slot=stick_slot,
        )
        inventory = dict(controller.observe().inventory or {})
        if (
            _count(inventory, _PLANK_NAMES) < 3
            or int(inventory.get("stick", 0) or 0) < 2
            or int(inventory.get("crafting_table", 0) or 0) < 1
        ):
            memory.update_state(controller.observe())
            return SkillResult(
                False, "2x2 GUI recipes were not observed", controller.steps - start
            )

        controller.step(Action(ActionType.HOTBAR, target=str(table_slot)))
        if controller.observe().selected_item != "crafting_table":
            memory.update_state(controller.observe())
            return SkillResult(
                False, "crafted table was not in expected hotbar slot", controller.steps - start
            )
        controller.step(Action(ActionType.CAMERA, pitch=80.0))
        before_table = _qty(controller, "crafting_table")
        controller.step(Action(ActionType.USE, sneak=True))
        controller.step(Action(ActionType.WAIT))
        if _qty(controller, "crafting_table") >= before_table:
            memory.update_state(controller.observe())
            return SkillResult(
                False, "crafting table placement was not observed", controller.steps - start
            )

        controller.step(Action(ActionType.USE))
        _table_crafting(
            controller,
            plank_slot=plank_slot,
            stick_slot=stick_slot,
            pickaxe_slot=pickaxe_slot,
        )
        memory.update_state(controller.observe())
        success = _qty(controller, "wooden_pickaxe") >= 1
        return SkillResult(
            success,
            "crafted wooden pickaxe" if success else "3x3 GUI recipe was not observed",
            controller.steps - start,
        )


__all__ = ["CraftItemSkill"]

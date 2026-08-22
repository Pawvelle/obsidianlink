"""Bounded primitive capabilities exposed to the GeneralAgent planner.

Each skill performs one kind of Minecraft interaction. Task workflows belong
to the planner, not to this module.
"""

from __future__ import annotations

from obsidianlink.agents.memory import AgentMemory
from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.skills.base import SkillResult, bounded_int


def _finish(
    controller: MinecraftController,
    memory: AgentMemory,
    start: int,
    message: str,
    *,
    success: bool = True,
    metadata: dict[str, object] | None = None,
) -> SkillResult:
    memory.update_state(controller.observe())
    return SkillResult(
        success,
        message,
        controller.steps - start,
        metadata=dict(metadata or {}),
    )


class MoveSkill:
    name = "move"
    description = (
        "Primitive movement. arguments: direction=forward|back|left|right, "
        "ticks=1..64, jump=true|false."
    )

    def execute(self, controller, memory, arguments):
        direction = str(arguments.get("direction", "forward")).strip().lower()
        vectors = {
            "forward": (1, 0),
            "back": (-1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }
        if direction not in vectors:
            return SkillResult(False, f"invalid movement direction: {direction}", 0)
        ticks = bounded_int(arguments.get("ticks"), default=1, minimum=1, maximum=64)
        jump = bool(arguments.get("jump", False))
        start = controller.steps
        dx, dz = vectors[direction]
        while controller.steps - start < ticks and not controller.exhausted:
            controller.step(Action(ActionType.MOVE, dx=dx, dz=dz, jump=jump))
        executed = controller.steps - start
        return _finish(
            controller,
            memory,
            start,
            f"moved {direction} for {executed}/{ticks} ticks",
            success=executed == ticks,
            metadata={"direction": direction, "requested_ticks": ticks, "jump": jump},
        )


class LookSkill:
    name = "look"
    description = "Primitive camera turn. arguments: yaw=-180..180, pitch=-90..90 degrees."

    def execute(self, controller, memory, arguments):
        try:
            yaw = max(-180.0, min(180.0, float(arguments.get("yaw", 0.0))))
            pitch = max(-90.0, min(90.0, float(arguments.get("pitch", 0.0))))
        except (TypeError, ValueError):
            return SkillResult(False, "yaw and pitch must be numeric", 0)
        start = controller.steps
        if controller.exhausted:
            return _finish(controller, memory, start, "controller step budget exhausted", success=False)
        controller.step(Action(ActionType.CAMERA, yaw=yaw, pitch=pitch))
        return _finish(
            controller, memory, start, "camera adjusted", metadata={"yaw": yaw, "pitch": pitch}
        )


class AttackSkill:
    name = "attack"
    description = "Primitive mining/attack interaction under the crosshair. arguments: ticks=1..320."

    def execute(self, controller, memory, arguments):
        ticks = bounded_int(arguments.get("ticks"), default=1, minimum=1, maximum=320)
        start = controller.steps
        before = dict(controller.observe().inventory or {})
        while controller.steps - start < ticks and not controller.exhausted:
            controller.step(Action(ActionType.ATTACK))
        return _finish(
            controller,
            memory,
            start,
            f"attacked for {controller.steps - start}/{ticks} ticks",
            success=controller.steps - start == ticks,
            metadata={
                "inventory_before": before,
                "inventory_after": dict(controller.observe().inventory or {}),
            },
        )


class InteractSkill:
    name = "interact"
    description = "Primitive use/interact action on the crosshair target. arguments: sneak=true|false."

    def execute(self, controller, memory, arguments):
        start = controller.steps
        if controller.exhausted:
            return _finish(controller, memory, start, "controller step budget exhausted", success=False)
        controller.step(Action(ActionType.USE, sneak=bool(arguments.get("sneak", False))))
        return _finish(controller, memory, start, "used selected item once")


class EquipItemSkill:
    name = "equip_item"
    description = (
        "Equip one inventory item by name. arguments: item=<inventory item>, "
        "for example wooden_sword or crafting_table."
    )

    def execute(self, controller, memory, arguments):
        item = str(arguments.get("item") or arguments.get("target") or "").strip()
        start = controller.steps
        if not item:
            return SkillResult(False, "equip_item requires item", 0)
        if controller.exhausted:
            return _finish(controller, memory, start, "controller step budget exhausted", success=False)
        controller.step(Action(ActionType.EQUIP, target=item))
        selected = controller.observe().selected_item
        return _finish(
            controller,
            memory,
            start,
            f"equipped {item}",
            success=selected == item,
            metadata={"item": item, "selected_item": selected},
        )


class InspectInventorySkill:
    name = "inspect_inventory"
    description = "Read the current agent-visible inventory and selected item; no world action."

    def execute(self, controller, memory, arguments):
        start = controller.steps
        observation = controller.observe()
        memory.update_state(observation)
        return SkillResult(
            True,
            "inventory inspected",
            0,
            metadata={
                "inventory": dict(observation.inventory or {}),
                "selected_item": observation.selected_item,
            },
        )


class PlaceBlockSkill:
    name = "place_block"
    description = (
        "Place one named inventory block at the crosshair. arguments: "
        "item=<block name>, sneak=true|false. If item is omitted, uses the equipped item."
    )

    def execute(self, controller, memory, arguments):
        start = controller.steps
        item = str(arguments.get("item") or arguments.get("target") or "").strip()
        selected = controller.observe().selected_item
        target = item or selected
        if not target:
            return _finish(controller, memory, start, "no item to place", success=False)
        if controller.exhausted:
            return _finish(controller, memory, start, "controller step budget exhausted", success=False)
        controller.step(
            Action(
                ActionType.PLACE,
                target=target,
                sneak=bool(arguments.get("sneak", True)),
            )
        )
        return _finish(
            controller,
            memory,
            start,
            f"attempted one placement with {target}",
            metadata={"item": target},
        )


class CraftSkill:
    name = "craft"
    description = (
        "Craft one named item through MineDojo's recipe command. arguments: "
        "item=<recipe output>, table=true when a placed crafting table is required."
    )

    def execute(self, controller, memory, arguments):
        item = str(arguments.get("item") or arguments.get("target") or "").strip()
        start = controller.steps
        if not item:
            return SkillResult(False, "craft requires item", 0)
        if controller.exhausted:
            return _finish(controller, memory, start, "controller step budget exhausted", success=False)
        target = f"table:{item}" if arguments.get("table") else item
        before = dict(controller.observe().inventory or {})
        controller.step(Action(ActionType.CRAFT, target=target))
        after = dict(controller.observe().inventory or {})
        gained = int(after.get(item, 0) or 0) > int(before.get(item, 0) or 0)
        return _finish(
            controller,
            memory,
            start,
            f"crafted {item}" if gained else f"crafted {item} but inventory unchanged",
            success=gained,
            metadata={"item": item, "inventory_before": before, "inventory_after": after},
        )


class SmeltSkill:
    name = "smelt"
    description = (
        "Smelt one named item at a placed furnace. arguments: item=<input or output>, "
        "for example iron_ingot."
    )

    def execute(self, controller, memory, arguments):
        item = str(arguments.get("item") or arguments.get("target") or "").strip()
        start = controller.steps
        if not item:
            return SkillResult(False, "smelt requires item", 0)
        if controller.exhausted:
            return _finish(controller, memory, start, "controller step budget exhausted", success=False)
        controller.step(Action(ActionType.SMELT, target=item))
        return _finish(
            controller,
            memory,
            start,
            f"attempted smelt {item}",
            metadata={"item": item},
        )


class WaitSkill:
    name = "wait"
    description = "Wait without interacting. arguments: ticks=1..64."

    def execute(self, controller, memory, arguments):
        ticks = bounded_int(arguments.get("ticks"), default=1, minimum=1, maximum=64)
        start = controller.steps
        while controller.steps - start < ticks and not controller.exhausted:
            controller.step(Action(ActionType.WAIT))
        executed = controller.steps - start
        return _finish(
            controller, memory, start, f"waited {executed}/{ticks} ticks", success=executed == ticks
        )


__all__ = [
    "AttackSkill",
    "CraftSkill",
    "EquipItemSkill",
    "InspectInventorySkill",
    "InteractSkill",
    "LookSkill",
    "MoveSkill",
    "PlaceBlockSkill",
    "SmeltSkill",
    "WaitSkill",
]

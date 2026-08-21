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


class SelectHotbarSkill:
    name = "select_hotbar"
    description = "Select one hotbar slot. arguments: slot=1..9."

    def execute(self, controller, memory, arguments):
        slot = bounded_int(arguments.get("slot"), default=1, minimum=1, maximum=9)
        start = controller.steps
        if controller.exhausted:
            return _finish(controller, memory, start, "controller step budget exhausted", success=False)
        controller.step(Action(ActionType.HOTBAR, target=str(slot)))
        return _finish(
            controller,
            memory,
            start,
            f"selected hotbar slot {slot}",
            metadata={"slot": slot, "selected_item": controller.observe().selected_item},
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
        "Place/use the currently selected block once against the crosshair target. "
        "arguments: sneak=true|false. Select the hotbar item separately."
    )

    def execute(self, controller, memory, arguments):
        start = controller.steps
        selected = controller.observe().selected_item
        if not selected:
            return _finish(controller, memory, start, "no selected item to place", success=False)
        if controller.exhausted:
            return _finish(controller, memory, start, "controller step budget exhausted", success=False)
        controller.step(Action(ActionType.USE, sneak=bool(arguments.get("sneak", True))))
        return _finish(
            controller,
            memory,
            start,
            f"attempted one placement with {selected}",
            metadata={"selected_item": selected},
        )


class CraftingActionSkill:
    name = "crafting_action"
    description = (
        "One primitive inventory/crafting GUI action. arguments: "
        "operation=toggle|left_click|right_click; x=0..639 and y=0..359 for clicks."
    )

    def __init__(self) -> None:
        self._cursor = (320, 180)

    def execute(self, controller, memory, arguments):
        operation = str(arguments.get("operation", "toggle")).strip().lower()
        start = controller.steps
        if controller.exhausted:
            return _finish(controller, memory, start, "controller step budget exhausted", success=False)
        if operation == "toggle":
            controller.step(Action(ActionType.INVENTORY))
            self._cursor = (320, 180)
        elif operation in {"left_click", "right_click"}:
            x = bounded_int(arguments.get("x"), default=320, minimum=0, maximum=639)
            y = bounded_int(arguments.get("y"), default=180, minimum=0, maximum=359)
            old_x, old_y = self._cursor
            controller.step(
                Action(
                    ActionType.CAMERA,
                    yaw=(x - old_x) / (2400.0 / 360.0),
                    pitch=(y - old_y) / (2400.0 / 360.0),
                )
            )
            if controller.exhausted:
                return _finish(
                    controller,
                    memory,
                    start,
                    "controller budget ended before crafting click",
                    success=False,
                    metadata={"operation": operation, "cursor": (x, y)},
                )
            click = ActionType.ATTACK if operation == "left_click" else ActionType.USE
            controller.step(Action(click))
            self._cursor = (x, y)
        else:
            return SkillResult(False, f"invalid crafting operation: {operation}", 0)
        return _finish(
            controller,
            memory,
            start,
            f"crafting GUI action executed: {operation}",
            metadata={"operation": operation, "cursor": self._cursor},
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
    "CraftingActionSkill",
    "InspectInventorySkill",
    "InteractSkill",
    "LookSkill",
    "MoveSkill",
    "PlaceBlockSkill",
    "SelectHotbarSkill",
    "WaitSkill",
]

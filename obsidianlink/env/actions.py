"""Bounded actions. The adapter maps each type onto MineDojo event-level keys.

Active verbs on ``MineDojoEnvironment``:

* MOVE / CAMERA / ATTACK / USE / WAIT — locomotion and crosshair interaction
* EQUIP / PLACE / CRAFT / SMELT / DROP — item-name commands, not hotbar slots

``HOTBAR`` and ``INVENTORY`` remain on the enum for archived MineRL code.
``MineDojoEnvironment`` rejects them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    MOVE = "move"
    CAMERA = "camera"
    ATTACK = "attack"
    USE = "use"
    PLACE = "place"
    EQUIP = "equip"
    CRAFT = "craft"
    SMELT = "smelt"
    DROP = "drop"
    WAIT = "wait"
    HOTBAR = "hotbar"
    INVENTORY = "inventory"


@dataclass(frozen=True)
class Action:
    type: ActionType

    # MOVE. dx>0 forward, dx<0 back, dz>0 right, dz<0 left.
    dx: int = 0
    dz: int = 0

    # CAMERA delta, degrees. MineDojo consumes ``[pitch, yaw]``.
    yaw: float = 0.0
    pitch: float = 0.0

    # EQUIP / PLACE / CRAFT / SMELT payload: a concrete item name.
    # CRAFT may use ``table:<item>`` to force the crafting-table recipe.
    target: str = ""

    # USE / PLACE modifier. Maps to MineDojo ``sneak``.
    sneak: bool = False

    # MOVE modifier for natural terrain.
    jump: bool = False


__all__ = ["Action", "ActionType"]

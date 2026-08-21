"""Bounded actions. The adapter decides how each type maps to MineRL.

Do not assume every verb is live-reliable. Current evidence:

* MOVE / CAMERA / ATTACK / WAIT — exercised on live MineRL (Phase 1).
* USE — present in the adapter; reliability is task-dependent.
* HOTBAR — L1 item select. MineRL 1.0.2 / MCP-Reborn cannot use
  EquipAction (``equip none`` crashes ``constructKeyboardState``).
* PLACE / EQUIP — declared for completeness. Malmo ``PlaceBlock`` has
  crashed the server on this stack. Do not send EquipAction on L1.
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
    HOTBAR = "hotbar"
    INVENTORY = "inventory"
    WAIT = "wait"


@dataclass(frozen=True)
class Action:
    type: ActionType

    # MOVE. dx>0 forward, dx<0 back, dz>0 right, dz<0 left.
    dx: int = 0
    dz: int = 0

    # CAMERA delta, degrees. MineRL consumes ``[pitch, yaw]``.
    yaw: float = 0.0
    pitch: float = 0.0

    # USE / PLACE / EQUIP / HOTBAR payload.
    # HOTBAR target is slot ``"1"``–``"9"`` (or ``"hotbar.N"``).
    target: str = ""

    # Optional modifier. Needed to place against a block without
    # falling into fluids. Maps to MineRL ``sneak``, not a new verb.
    sneak: bool = False

    # Controller-only movement modifier for natural terrain traversal.
    jump: bool = False


__all__ = ["Action", "ActionType"]

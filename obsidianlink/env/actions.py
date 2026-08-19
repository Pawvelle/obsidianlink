"""Bounded actions. The adapter decides how each type maps to MineRL.

Do not assume every verb is live-reliable. Current evidence:

* MOVE / CAMERA / ATTACK / WAIT — exercised on live MineRL (Phase 1).
* USE — present in the adapter; reliability is task-dependent.
* PLACE / EQUIP — declared so a later L1 can use them; Malmo
  ``PlaceBlock`` has crashed the server on this stack. Do not treat
  them as verified until a live task produces world-effect evidence.
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

    # USE / PLACE / EQUIP payload. Free-form item or block name.
    target: str = ""


__all__ = ["Action", "ActionType"]

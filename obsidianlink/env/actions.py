"""Structured actions for the Minecraft agent loop.

Phase 1 introduces a small payload for the bounded action set
(MOVE / CAMERA / ATTACK / USE / PLACE). All payload fields default to
"no effect" so ``Action(type=ActionType.WAIT)`` keeps working and the
older in-process stubs continue to type-check.

The payload is intentionally flat: no nested dicts, no per-type
subclasses. The MineRL adapter (and any future adapter) is the only
place that decides how each ``ActionType`` is mapped onto a backend
action space.
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
    WAIT = "wait"


@dataclass(frozen=True)
class Action:
    type: ActionType

    # Movement payload (MOVE). Sign convention:
    #   dx > 0  -> forward
    #   dx < 0  -> back
    #   dz > 0  -> right
    #   dz < 0  -> left
    dx: int = 0
    dz: int = 0

    # Look payload (CAMERA), in degrees, applied as a delta this step.
    yaw: float = 0.0
    pitch: float = 0.0

    # Targeting / placement payload (ATTACK, USE, PLACE). ``target`` is
    # a free-form block / item name (e.g. "dirt"); ``slot`` is the
    # hotbar slot index when the action implies a slot selection.
    target: str = ""
    slot: int = 0


__all__ = ["Action", "ActionType"]

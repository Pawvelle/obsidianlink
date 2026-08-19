"""Structured actions for the Minecraft agent loop.

Phase 1 introduces a small payload for the bounded action set
(MOVE / CAMERA / ATTACK / USE / PLACE). All payload fields default to
"no effect" so ``Action(type=ActionType.WAIT)`` keeps working and the
older in-process stubs continue to type-check.

Phase 3 adds :attr:`ActionType.EQUIP` for the L1 hotbar switch:
L1 ships with two hotbar slots (obsidian at slot 0,
flint_and_steel at slot 1); the agent emits
``{"action": "equip", "target": "flint_and_steel"}`` (or
``"obsidian"``) to switch the active item before the next PLACE / USE.

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
    EQUIP = "equip"
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

    # Targeting / placement payload (ATTACK, USE, PLACE, EQUIP).
    # ``target`` is a free-form block / item name (e.g. "dirt",
    # "flint_and_steel"). ``slot`` is a 1-based hotbar slot
    # index (1..9) and is reserved for future use; the L1 path
    # uses ``target`` with EQUIP to switch hotbar items.
    target: str = ""
    slot: int = 0


__all__ = ["Action", "ActionType"]

"""Structured actions. No Minecraft execution lives here."""

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

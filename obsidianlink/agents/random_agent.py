"""Random legal actions. Interface smoke only — not a task solver."""

from __future__ import annotations

import random
from typing import Sequence

from obsidianlink.agents.base_agent import BaseAgent
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Observation

LEGAL_TYPES: tuple[ActionType, ...] = (
    ActionType.MOVE,
    ActionType.CAMERA,
    ActionType.ATTACK,
    ActionType.USE,
    ActionType.EQUIP,
    ActionType.PLACE,
    ActionType.CRAFT,
    ActionType.WAIT,
)
_ARCHIVED_TYPES = frozenset({ActionType.HOTBAR, ActionType.INVENTORY})


class RandomAgent(BaseAgent):
    """Uniform sample over the MineDojo event-level action surface."""

    def __init__(
        self,
        *,
        rng: random.Random | None = None,
        types: Sequence[ActionType] = LEGAL_TYPES,
    ) -> None:
        self._rng = rng if rng is not None else random.Random()
        self._types = tuple(types)
        if any(kind in _ARCHIVED_TYPES for kind in self._types):
            raise ValueError("RandomAgent must not emit archived HOTBAR/INVENTORY")

    def act(self, observation: Observation) -> Action:
        kind = self._rng.choice(self._types)
        items = [
            name
            for name, qty in dict(observation.inventory or {}).items()
            if int(qty or 0) > 0
        ]
        if kind is ActionType.MOVE:
            return Action(
                type=ActionType.MOVE,
                dx=self._rng.choice((-1, 0, 1)),
                dz=self._rng.choice((-1, 0, 1)),
            )
        if kind is ActionType.CAMERA:
            return Action(
                type=ActionType.CAMERA,
                yaw=float(self._rng.choice((-15, 0, 15))),
                pitch=float(self._rng.choice((-15, 0, 15))),
            )
        if kind is ActionType.EQUIP:
            if not items:
                return Action(type=ActionType.WAIT)
            return Action(type=ActionType.EQUIP, target=self._rng.choice(items))
        if kind is ActionType.PLACE:
            if not items:
                return Action(type=ActionType.WAIT)
            return Action(
                type=ActionType.PLACE,
                target=self._rng.choice(items),
                sneak=self._rng.choice((False, True)),
            )
        if kind is ActionType.CRAFT:
            return Action(type=ActionType.CRAFT, target="stick")
        if kind is ActionType.USE:
            return Action(type=ActionType.USE, sneak=self._rng.choice((False, True)))
        if kind is ActionType.ATTACK:
            return Action(type=ActionType.ATTACK)
        return Action(type=ActionType.WAIT)


__all__ = ["LEGAL_TYPES", "RandomAgent"]

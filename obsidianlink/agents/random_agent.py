"""Random legal actions. Interface smoke only — not a task solver."""

from __future__ import annotations

import random
from typing import Sequence

from obsidianlink.agents.base_agent import BaseAgent
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Observation

# L1-legal verbs. EQUIP / PLACE are declared on ActionType but must not be
# sampled: EquipAction crashes this MineRL stack; PlaceBlock has crashed
# the Malmo server.
LEGAL_TYPES: tuple[ActionType, ...] = (
    ActionType.MOVE,
    ActionType.CAMERA,
    ActionType.ATTACK,
    ActionType.USE,
    ActionType.HOTBAR,
    ActionType.WAIT,
)
HOTBAR_SLOTS: tuple[str, ...] = tuple(str(i) for i in range(1, 10))


class RandomAgent(BaseAgent):
    """Uniform sample over the legal action surface."""

    def __init__(
        self,
        *,
        rng: random.Random | None = None,
        types: Sequence[ActionType] = LEGAL_TYPES,
    ) -> None:
        self._rng = rng if rng is not None else random.Random()
        self._types = tuple(types)
        if ActionType.EQUIP in self._types or ActionType.PLACE in self._types:
            raise ValueError("RandomAgent must not emit EQUIP or PLACE")

    def act(self, observation: Observation) -> Action:
        del observation
        kind = self._rng.choice(self._types)
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
        if kind is ActionType.HOTBAR:
            return Action(
                type=ActionType.HOTBAR,
                target=self._rng.choice(HOTBAR_SLOTS),
            )
        if kind is ActionType.USE:
            return Action(type=ActionType.USE, sneak=self._rng.choice((False, True)))
        if kind is ActionType.ATTACK:
            return Action(type=ActionType.ATTACK)
        return Action(type=ActionType.WAIT)


__all__ = ["HOTBAR_SLOTS", "LEGAL_TYPES", "RandomAgent"]

"""Inventory-rule baseline for the L1 obsidian casting task.

Not an LLM. Not a planner. Distinct from ``obsidianlink.agents.reactive``
(the vision/Wiki model agent). This class only reads ``Observation``
(frame / inventory / selected_item) and never hidden_state.

Rule sketch:

* no lava_bucket → select empty bucket, look down, walk, USE
* has lava_bucket → look down, sneak-USE to place
* lava placed and still has water_bucket → select water, yaw nudge, sneak-USE
* after watering → WAIT (obsidian settle); then stop
"""

from __future__ import annotations

from obsidianlink.agents.base_agent import BaseAgent
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Observation

HOTBAR_WATER = "1"
HOTBAR_BUCKET = "2"


def _qty(observation: Observation, name: str) -> int:
    inv = observation.inventory or {}
    try:
        return int(inv.get(name, 0) or 0)
    except (TypeError, ValueError):
        return 0


class ReactiveAgent(BaseAgent):
    """Simple L1 baseline. Success is not guaranteed; the loop must stay safe."""

    def __init__(self) -> None:
        self.finished = False
        self._phase = "scoop"
        self._ticks = 0
        self._saw_lava_bucket = False
        self._poured_lava = False
        self._used_water = False
        self._wait_after_water = 0

    def reset(self) -> None:
        self.finished = False
        self._phase = "scoop"
        self._ticks = 0
        self._saw_lava_bucket = False
        self._poured_lava = False
        self._used_water = False
        self._wait_after_water = 0

    def act(self, observation: Observation) -> Action:
        lava = _qty(observation, "lava_bucket")
        water = _qty(observation, "water_bucket")
        selected = observation.selected_item

        if lava >= 1:
            self._saw_lava_bucket = True
        if self._saw_lava_bucket and lava == 0:
            self._poured_lava = True
        if self._poured_lava and water == 0:
            self._used_water = True

        if self.finished:
            return Action(type=ActionType.WAIT)

        if self._used_water:
            self._wait_after_water += 1
            if self._wait_after_water >= 8:
                self.finished = True
            return Action(type=ActionType.WAIT)

        if lava >= 1:
            return self._place_lava(selected)

        if self._poured_lava and water >= 1:
            return self._place_water(selected)

        return self._scoop_lava(selected)

    def _scoop_lava(self, selected: str | None) -> Action:
        self._ticks += 1
        if selected != "bucket":
            return Action(type=ActionType.HOTBAR, target=HOTBAR_BUCKET)
        # Look down toward the pool, walk forward, then a single USE.
        if self._ticks % 6 == 1:
            return Action(type=ActionType.CAMERA, pitch=15.0)
        if self._ticks % 6 in {2, 3, 4}:
            return Action(type=ActionType.MOVE, dx=1)
        return Action(type=ActionType.USE)

    def _place_lava(self, selected: str | None) -> Action:
        self._ticks += 1
        if selected != "lava_bucket":
            return Action(type=ActionType.HOTBAR, target=HOTBAR_BUCKET)
        if self._ticks % 4 == 1:
            return Action(type=ActionType.MOVE, dx=-1)
        if self._ticks % 4 == 2:
            return Action(type=ActionType.CAMERA, pitch=20.0)
        return Action(type=ActionType.USE, sneak=True)

    def _place_water(self, selected: str | None) -> Action:
        self._ticks += 1
        if selected != "water_bucket":
            return Action(type=ActionType.HOTBAR, target=HOTBAR_WATER)
        if self._ticks % 4 == 1:
            return Action(type=ActionType.CAMERA, yaw=12.0)
        return Action(type=ActionType.USE, sneak=True)


__all__ = ["HOTBAR_BUCKET", "HOTBAR_WATER", "ReactiveAgent"]

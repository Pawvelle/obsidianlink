"""Small Portal-task reference agents.

These controllers deliberately use only the public ``Observation`` API.
They are not a planner and do not inspect evaluator truth.  The Oracle is a
deterministic mechanics reference used to establish whether the benchmark can
be completed; evaluator truth remains the sole authority for success.
"""

from __future__ import annotations

from enum import Enum

from obsidianlink.agents.base_agent import BaseAgent
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Observation


class PortalState(str, Enum):
    FIND_RESOURCE = "FIND_RESOURCE"
    COLLECT = "COLLECT"
    BUILD = "BUILD"
    ACTIVATE = "ACTIVATE"
    COMPLETE = "COMPLETE"


def _quantity(observation: Observation, item: str) -> int:
    try:
        return int((observation.inventory or {}).get(item, 0) or 0)
    except (TypeError, ValueError):
        return 0


class RuleBasedPortalAgent(BaseAgent):
    """Inventory-driven FSM baseline for the formal Portal task.

    Its navigation is intentionally simple.  It provides a reproducible
    non-LLM baseline, not a claim that a heuristic can solve L1 reliably.
    """

    def __init__(self) -> None:
        self.state = PortalState.FIND_RESOURCE
        self.finished = False
        self._ticks = 0
        self._casts = 0

    def reset(self) -> None:
        self.state = PortalState.FIND_RESOURCE
        self.finished = False
        self._ticks = 0
        self._casts = 0

    def act(self, observation: Observation) -> Action:
        self._ticks += 1
        lava = _quantity(observation, "lava_bucket")
        water = _quantity(observation, "water_bucket")
        selected = observation.selected_item

        if self.state is PortalState.FIND_RESOURCE:
            self.state = PortalState.COLLECT
            return Action(type=ActionType.HOTBAR, target="2")
        if self.state is PortalState.COLLECT:
            if lava:
                self.state = PortalState.BUILD
                return Action(type=ActionType.CAMERA, pitch=35.0)
            if selected != "bucket":
                return Action(type=ActionType.HOTBAR, target="2")
            phase = self._ticks % 6
            if phase in (1, 2, 3):
                return Action(type=ActionType.MOVE, dx=1)
            if phase == 4:
                return Action(type=ActionType.CAMERA, pitch=35.0)
            return Action(type=ActionType.USE)
        if self.state is PortalState.BUILD:
            if lava:
                if selected != "lava_bucket":
                    return Action(type=ActionType.HOTBAR, target="2")
                return Action(type=ActionType.USE, sneak=True)
            if water:
                if selected != "water_bucket":
                    return Action(type=ActionType.HOTBAR, target="1")
                self._casts += 1
                return Action(type=ActionType.USE, sneak=True)
            # A real completion is never inferred from inventory.  The FSM
            # merely reaches its activation attempt after its fixed budget.
            if self._casts >= 10:
                self.state = PortalState.ACTIVATE
            else:
                self.state = PortalState.COLLECT
            return Action(type=ActionType.WAIT)
        if self.state is PortalState.ACTIVATE:
            if selected != "flint_and_steel":
                return Action(type=ActionType.HOTBAR, target="5")
            self.state = PortalState.COMPLETE
            self.finished = True
            return Action(type=ActionType.USE)
        return Action(type=ActionType.WAIT)


class OraclePortalAgent(BaseAgent):
    """Deterministic legal-action reference controller for Portal L1.

    The sequence repeats the verified primitive ``bucket -> lava -> water``
    ten times, then attempts ignition and portal entry.  It intentionally
    reads no pose, reward, biome, grid, or other hidden state.  On this
    MineRL stack, L1's known fluid-server timeout may still make a live run
    fail; that is reported by the benchmark rather than hidden by the agent.
    """

    def __init__(self) -> None:
        self._program: list[Action] = []
        self._index = 0
        self.finished = False
        self.reset()

    def reset(self) -> None:
        self._index = 0
        self.finished = False
        program: list[Action] = []
        for _ in range(10):
            program.extend(
                [
                    Action(ActionType.HOTBAR, target="2"),
                    Action(ActionType.CAMERA, pitch=35.0),
                    Action(ActionType.MOVE, dx=1),
                    Action(ActionType.MOVE, dx=1),
                    Action(ActionType.USE),
                    Action(ActionType.MOVE, dx=-1),
                    Action(ActionType.MOVE, dx=-1),
                    Action(ActionType.USE, sneak=True),
                    Action(ActionType.HOTBAR, target="1"),
                    Action(ActionType.CAMERA, yaw=12.0, pitch=10.0),
                    Action(ActionType.USE, sneak=True),
                    Action(ActionType.WAIT),
                    Action(ActionType.WAIT),
                ]
            )
        program.extend(
            [Action(ActionType.HOTBAR, target="5"), Action(ActionType.USE)]
            + [Action(ActionType.MOVE, dx=1) for _ in range(40)]
        )
        self._program = program

    def act(self, observation: Observation) -> Action:
        del observation
        if self._index >= len(self._program):
            self.finished = True
            return Action(type=ActionType.WAIT)
        action = self._program[self._index]
        self._index += 1
        return action


__all__ = ["OraclePortalAgent", "PortalState", "RuleBasedPortalAgent"]

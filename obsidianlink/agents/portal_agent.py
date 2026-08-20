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
    """Deterministic, calibrated mechanics reference for Portal L1.

    The program is the shortest *live-verified* bucket-casting trace from the
    fixed L1 spawn: empty bucket -> lava -> water -> newly generated obsidian
    (65 actions).  It is deliberately finite: pretending that repeated water
    clicks build a ten-block frame would create flowing-water load and yield a
    false Oracle.  Full-frame/ignition capability is only claimed after a
    similarly verified, low-fluid construction trace exists.

    It reads no pose, reward, biome, grid, or other hidden state.
    """

    def __init__(self) -> None:
        self._program: list[Action] = []
        self._index = 0
        self.finished = False
        self.reset()

    def reset(self) -> None:
        self._index = 0
        self.finished = False
        self._program = _gate_one_program()

    def act(self, observation: Observation) -> Action:
        del observation
        if self._index >= len(self._program):
            self.finished = True
            return Action(type=ActionType.WAIT)
        action = self._program[self._index]
        self._index += 1
        return action


def _gate_one_program() -> list[Action]:
    """Return the 65-action trace recorded by the successful live gate-1 run.

    Keeping the trace as compact runs avoids speculative navigation and avoids
    all repeated fluid placement.  Each bucket interaction is exactly one
    ``USE``; additional USE ticks can undo a filled/placed bucket on this
    MineRL/MCP-Reborn stack.
    """
    out = [Action(ActionType.HOTBAR, target="2"), Action(ActionType.WAIT)]
    out += [Action(ActionType.MOVE, dx=1) for _ in range(19)]
    out += [Action(ActionType.CAMERA, pitch=20.0), Action(ActionType.USE)]
    out += [Action(ActionType.MOVE, dx=-1) for _ in range(8)]
    out += [Action(ActionType.CAMERA, yaw=45.0), Action(ActionType.CAMERA, yaw=45.0)]
    out += [Action(ActionType.MOVE, dx=1) for _ in range(6)]
    out += [Action(ActionType.CAMERA, pitch=13.0), Action(ActionType.USE, sneak=True)]
    out += [Action(ActionType.WAIT) for _ in range(8)]
    out += [Action(ActionType.HOTBAR, target="1"), Action(ActionType.WAIT)]
    out += [Action(ActionType.CAMERA, yaw=12.0), Action(ActionType.USE, sneak=True)]
    out += [Action(ActionType.CAMERA, pitch=-30.0), Action(ActionType.CAMERA, pitch=-10.0)]
    out += [Action(ActionType.WAIT) for _ in range(2)]
    out += [Action(ActionType.CAMERA, pitch=30.0), Action(ActionType.CAMERA, pitch=10.0)]
    out += [Action(ActionType.WAIT) for _ in range(6)]
    assert len(out) == 65
    return out


__all__ = ["OraclePortalAgent", "PortalState", "RuleBasedPortalAgent"]

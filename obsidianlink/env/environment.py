"""Environment interface and agent-visible Observation.

``Observation`` is the *only* world state an Agent may see. Evaluator-only
truth (reward, biome, scene labels, success flags) must never appear here.
Coordinates and facing are agent-visible for navigation experiments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any

from obsidianlink.env.actions import Action

# Frozen contract: add a field only if the Agent is allowed to see it.
_AGENT_VISIBLE_FIELDS = frozenset(
    {"frame", "inventory", "selected_item", "x", "y", "z", "yaw", "pitch"}
)


@dataclass(frozen=True)
class Observation:
    """Agent-visible world state.

    ``frame`` is the RGB POV. ``inventory`` is ``{item: count}``.
    ``selected_item`` is the equipped main-hand item when the backend
    exposes it; otherwise a best-effort inventory hint.
    ``x`` / ``y`` / ``z`` / ``yaw`` / ``pitch`` are world pose when the
    backend exposes ``location_stats``.
    """

    frame: Any = None
    inventory: dict[str, int] | None = None
    selected_item: str | None = None
    x: float | None = None
    y: float | None = None
    z: float | None = None
    yaw: float | None = None
    pitch: float | None = None

    def pose(self) -> dict[str, float]:
        """Known coordinate fields only; omitted keys were not reported."""
        out: dict[str, float] = {}
        for name in ("x", "y", "z", "yaw", "pitch"):
            value = getattr(self, name)
            if value is not None:
                out[name] = float(value)
        return out

    def agent_view(self) -> dict[str, Any]:
        """Planner-safe summary. Never includes RGB pixels or evaluator truth."""
        return {
            "inventory": dict(self.inventory or {}),
            "selected_item": self.selected_item,
            "position": self.pose(),
            "has_visual_frame": self.frame is not None,
        }


def observation_field_names() -> frozenset[str]:
    return frozenset(f.name for f in fields(Observation))


class Environment(ABC):
    @abstractmethod
    def reset(self) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def observe(self) -> Observation:
        """Latest agent-visible observation, without stepping."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Action) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


assert observation_field_names() == _AGENT_VISIBLE_FIELDS

__all__ = ["Environment", "Observation", "observation_field_names"]

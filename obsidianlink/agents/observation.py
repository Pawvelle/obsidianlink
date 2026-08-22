"""Planner-facing grounded observation.

The Environment ``Observation`` contract is
``frame`` / ``inventory`` / ``selected_item`` / pose. Goal, last action,
and success feedback are agent state, not evaluator truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from obsidianlink.agents.memory import AgentMemory
from obsidianlink.env.environment import Observation

_TOOL_MARKERS = (
    "pickaxe",
    "axe",
    "sword",
    "shovel",
    "hoe",
    "shears",
    "shield",
    "bow",
    "bucket",
    "flint_and_steel",
)


@dataclass(frozen=True)
class GroundedObservation:
    """Compact decision context built for the Planner."""

    goal: str
    subgoal: str
    next_objective: str
    position: dict[str, Any]
    inventory: dict[str, int]
    equipment: dict[str, Any]
    health: dict[str, Any]
    nearby_blocks: list[dict[str, Any]]
    visible_resources: list[str]
    nearby_entities: list[dict[str, Any]]
    last_action: dict[str, Any] | None
    action_result: dict[str, Any] | None
    feedback: dict[str, Any]
    selected_item: str | None = None
    has_visual_frame: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def as_prompt(self) -> dict[str, Any]:
        """JSON-safe Planner view. No RGB pixels or evaluator truth."""
        return {
            "goal": self.goal,
            "subgoal": self.subgoal,
            "next_objective": self.next_objective,
            "position": dict(self.position),
            "inventory": dict(self.inventory),
            "equipment": dict(self.equipment),
            "health": dict(self.health),
            "nearby_blocks": list(self.nearby_blocks),
            "visible_resources": list(self.visible_resources),
            "nearby_entities": list(self.nearby_entities),
            "selected_item": self.selected_item,
            "last_action": dict(self.last_action) if self.last_action else None,
            "action_result": dict(self.action_result) if self.action_result else None,
            "feedback": dict(self.feedback),
            "has_visual_frame": self.has_visual_frame,
        }


def build_grounded_observation(
    memory: AgentMemory,
    observation: Observation | None = None,
    *,
    local_view: dict[str, Any] | None = None,
) -> GroundedObservation:
    """Compose Environment Observation + Memory + optional local world hints."""
    obs = observation or memory.last_observation or Observation()
    view = dict(local_view or getattr(memory, "local_view", {}) or {})
    inventory = dict(obs.inventory or memory.inventory or {})
    last_step = memory.completed_steps[-1] if memory.completed_steps else None
    last_action = None
    action_result = None
    if last_step is not None:
        last_action = {
            "skill": last_step.skill,
            "arguments": dict(last_step.arguments),
            "success": last_step.success,
        }
        action_result = {
            "success": last_step.success,
            "message": last_step.message,
            "environment_steps": last_step.environment_steps,
            "inventory_delta": dict(memory.inventory_delta),
        }
    reflection = memory.last_reflection
    feedback = {
        "success": last_step.success if last_step is not None else None,
        "advanced_goal": reflection.advanced_goal if reflection is not None else None,
        "reason": reflection.reason if reflection is not None else None,
        "matched_expected": reflection.matched if reflection is not None else None,
    }
    return GroundedObservation(
        goal=memory.goal,
        subgoal=memory.current_subgoal or "",
        next_objective=memory.next_objective,
        position=_position_view(obs, view),
        inventory=inventory,
        equipment=_equipment_view(obs, inventory, view),
        health=_health_view(view),
        nearby_blocks=_list_of_dicts(view.get("nearby_blocks")),
        visible_resources=_string_list(view.get("visible_resources")),
        nearby_entities=_list_of_dicts(view.get("nearby_entities")),
        last_action=last_action,
        action_result=action_result,
        feedback=feedback,
        selected_item=obs.selected_item,
        has_visual_frame=obs.frame is not None,
    )


def _position_view(
    observation: Observation, local_view: dict[str, Any]
) -> dict[str, Any]:
    pose = observation.pose()
    raw = local_view.get("position")
    extras = dict(raw) if isinstance(raw, dict) else {}
    extras.update(pose)
    extras.pop("xpos", None)
    extras.pop("ypos", None)
    extras.pop("zpos", None)
    if extras:
        return extras
    return {"status": "unknown"}


def _health_view(local_view: dict[str, Any]) -> dict[str, Any]:
    raw = local_view.get("health")
    if isinstance(raw, dict) and raw:
        return dict(raw)
    if raw is not None and not isinstance(raw, dict):
        return {"value": raw}
    return {"status": "unknown"}


def _equipment_view(
    observation: Observation,
    inventory: dict[str, int],
    local_view: dict[str, Any],
) -> dict[str, Any]:
    tools = {
        name: int(count)
        for name, count in inventory.items()
        if _looks_like_tool(name) and int(count or 0) > 0
    }
    raw = local_view.get("equipment")
    extra = dict(raw) if isinstance(raw, dict) else {}
    return {
        "mainhand": extra.get("mainhand") or observation.selected_item,
        "tools": extra.get("tools") or tools,
    }


def _looks_like_tool(name: str) -> bool:
    key = str(name).casefold()
    return any(marker in key for marker in _TOOL_MARKERS)


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            out.append(dict(item))
        elif item:
            out.append({"name": str(item)})
    return out


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


__all__ = ["GroundedObservation", "build_grounded_observation"]

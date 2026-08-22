"""Reject primitive actions that cannot advance the current subgoal.

This is not a planner and not a workflow skill. It only blocks obvious
mismatches such as mining dirt while the active subgoal is collect wood.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from obsidianlink.agents.memory import AgentMemory
from obsidianlink.agents.planner import PlannerDecision
from obsidianlink.env.environment import Observation

_WOOD_TERMS = frozenset(
    {
        "wood",
        "log",
        "logs",
        "oak",
        "birch",
        "spruce",
        "jungle",
        "acacia",
        "tree",
        "trees",
        "plank",
        "planks",
        "stick",
        "sticks",
        "sapling",
    }
)
_DIRT_ITEMS = frozenset(
    {"dirt", "grass", "grass_block", "coarse_dirt", "podzol", "mycelium"}
)
_FILLER_ITEMS = _DIRT_ITEMS | frozenset({"sand", "red_sand", "gravel", "clay"})
_IRON_TERMS = frozenset({"iron", "sword", "ingot", "ore"})
_IRON_PATH_ITEMS = frozenset(
    {
        "oak_log",
        "birch_log",
        "spruce_log",
        "jungle_log",
        "acacia_log",
        "dark_oak_log",
        "oak_planks",
        "birch_planks",
        "spruce_planks",
        "jungle_planks",
        "acacia_planks",
        "dark_oak_planks",
        "stick",
        "sticks",
        "crafting_table",
        "cobblestone",
        "stone",
        "coal",
        "charcoal",
        "furnace",
        "iron_ore",
        "iron_ingot",
        "iron_sword",
        "wooden_pickaxe",
        "stone_pickaxe",
        "iron_pickaxe",
        "wooden_axe",
        "stone_axe",
    }
)
_NOOP_SKILLS = frozenset({"wait", "inspect_inventory"})


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    reason: str
    code: str = "ok"

    def as_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "reason": self.reason, "code": self.code}


def validate_skill_decision(
    decision: PlannerDecision,
    memory: AgentMemory,
    observation: Observation | None = None,
) -> ValidationResult:
    """Return whether one primitive skill may run for the current subgoal."""
    if decision.type != "skill":
        return ValidationResult(True, "non-skill decisions are not validated")

    missing = _missing_prerequisite(decision, memory, observation)
    if missing is not None:
        return missing

    mentioned = _mentioned_items(decision)
    objective = _objective_text(memory, decision)
    if mentioned and _is_unrelated_to_objective(mentioned, objective):
        items = ", ".join(sorted(mentioned))
        return ValidationResult(
            False,
            (
                f"{decision.name} targeting {items} cannot advance "
                f"current objective {objective!r}"
            ),
            "unrelated_to_subgoal",
        )

    noop = _repeated_noop(decision, memory)
    if noop is not None:
        return noop
    return ValidationResult(True, "action is compatible with the current subgoal")


def _missing_prerequisite(
    decision: PlannerDecision,
    memory: AgentMemory,
    observation: Observation | None,
) -> ValidationResult | None:
    selected = None
    if observation is not None:
        selected = observation.selected_item
    if selected is None:
        selected = memory.selected_item
    if decision.name == "place_block" and not selected:
        return ValidationResult(
            False,
            "place_block is missing a selected item",
            "missing_prerequisite",
        )
    inventory = dict((observation.inventory if observation is not None else None) or memory.inventory)
    if decision.name == "crafting_action" and not inventory:
        subgoal = (decision.subgoal or memory.current_subgoal or "").casefold()
        if any(term in subgoal for term in _WOOD_TERMS | _IRON_TERMS | {"collect", "mine", "find"}):
            return ValidationResult(
                False,
                "crafting_action has no inventory materials for the current gather subgoal",
                "missing_prerequisite",
            )
    return None


def _repeated_noop(
    decision: PlannerDecision, memory: AgentMemory
) -> ValidationResult | None:
    if decision.name not in _NOOP_SKILLS:
        return None
    recent = memory.completed_steps[-2:]
    if len(recent) < 2:
        return None
    if all(step.skill == decision.name for step in recent):
        return ValidationResult(
            False,
            f"repeated {decision.name} does not advance the current subgoal",
            "no_progress",
        )
    return None


def _mentioned_items(decision: PlannerDecision) -> set[str]:
    items: set[str] = set()
    for key in ("target", "block", "item", "resource", "block_type"):
        value = decision.arguments.get(key)
        if value:
            items.add(_normalize_item(value))
    expected = decision.expected if isinstance(decision.expected, dict) else {}
    for key in ("inventory_min", "inventory_delta"):
        mapping = expected.get(key)
        if isinstance(mapping, dict):
            items.update(_normalize_item(name) for name in mapping)
    blob = " ".join(
        (
            decision.name,
            decision.reason,
            " ".join(str(v) for v in decision.arguments.values()),
        )
    ).casefold()
    for item in _FILLER_ITEMS:
        if item in blob:
            items.add(item)
    return {item for item in items if item}


def _objective_text(memory: AgentMemory, decision: PlannerDecision) -> str:
    parts = [
        memory.goal,
        memory.current_subgoal or "",
        decision.subgoal,
    ]
    return " ".join(part for part in parts if part).strip()


def _is_unrelated_to_objective(mentioned: set[str], objective: str) -> bool:
    text = objective.casefold()
    if not text:
        return False
    mentioned = {item for item in mentioned if item}
    if not mentioned:
        return False
    collecting_wood = any(term in text for term in _WOOD_TERMS)
    if collecting_wood and mentioned & _FILLER_ITEMS and not mentioned & _wood_items():
        return True
    if _looks_like_iron_task(text):
        if mentioned <= _FILLER_ITEMS:
            return True
        if mentioned and mentioned.isdisjoint(_IRON_PATH_ITEMS | _wood_items()) and mentioned <= _FILLER_ITEMS:
            return True
    return False


def _looks_like_iron_task(text: str) -> bool:
    return "iron" in text and ("sword" in text or "ingot" in text or "ore" in text)


def _wood_items() -> frozenset[str]:
    return frozenset(
        {
            "oak_log",
            "birch_log",
            "spruce_log",
            "jungle_log",
            "acacia_log",
            "dark_oak_log",
            "oak_planks",
            "oak_wood",
            "wood",
            "log",
            "logs",
            "planks",
            "stick",
            "sticks",
        }
    )


def _normalize_item(value: Any) -> str:
    return str(value).strip().casefold().split(":", 1)[-1]


__all__ = ["ValidationResult", "validate_skill_decision"]

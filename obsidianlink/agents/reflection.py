"""Lightweight result analysis after a primitive skill.

This is not a reflection framework, extra model call, or recovery policy.
It writes expected-vs-observed plus goal-progress notes into memory so
the next planner decision can change approach.
"""

from __future__ import annotations

from typing import Any

from obsidianlink.agents.memory import AgentMemory, ReflectionRecord
from obsidianlink.agents.planner import PlannerDecision
from obsidianlink.env.environment import Observation

_POSITIONING_SKILLS = frozenset(
    {"move", "look", "equip_item", "wait", "inspect_inventory"}
)
_FILLER_ITEMS = frozenset(
    {
        "dirt",
        "grass",
        "grass_block",
        "coarse_dirt",
        "sand",
        "red_sand",
        "gravel",
        "clay",
        "wheat_seeds",
        "seeds",
    }
)
_WOOD_TERMS = frozenset(
    {"wood", "log", "logs", "oak", "tree", "plank", "planks", "stick", "sticks"}
)
_WOOD_ITEMS = frozenset(
    {
        "log",
        "oak_log",
        "birch_log",
        "spruce_log",
        "jungle_log",
        "acacia_log",
        "dark_oak_log",
        "planks",
        "oak_planks",
        "birch_planks",
        "spruce_planks",
        "jungle_planks",
        "acacia_planks",
        "dark_oak_planks",
        "stick",
        "oak_wood",
        "wood",
        "log",
    }
)
_IRON_PATH_ITEMS = _WOOD_ITEMS | frozenset(
    {
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


def reflect_skill_outcome(
    memory: AgentMemory,
    decision: PlannerDecision,
    observation: Observation,
    *,
    skill_success: bool,
    skill_message: str,
) -> ReflectionRecord:
    observed = {
        "inventory": dict(observation.inventory or {}),
        "selected_item": observation.selected_item,
        "inventory_delta": dict(memory.inventory_delta),
        "skill_success": skill_success,
    }
    expected = dict(decision.expected or {})
    mismatches = _compare(expected, observed) if expected else []
    if not expected and not skill_success:
        mismatches.append(skill_message or "skill reported failure")
    matched = not mismatches
    advanced, progress_note = _goal_progress(memory, decision, observed)
    if matched:
        reason = skill_message or "observation matched the expected result"
    else:
        reason = "; ".join(mismatches)
    if progress_note and progress_note not in reason:
        reason = f"{reason}; {progress_note}" if reason else progress_note
    record = ReflectionRecord(
        skill=decision.name,
        subgoal=decision.subgoal or memory.current_subgoal or "",
        matched=matched,
        reason=reason,
        expected=expected,
        observed={
            "inventory": observed["inventory"],
            "selected_item": observed["selected_item"],
            "inventory_delta": observed["inventory_delta"],
        },
        advanced_goal=advanced,
        progress_note=progress_note,
    )
    memory.record_reflection(record)
    # A SkillResult failure is persisted by AgentMemory.record_step.  Only add
    # a reflection failure when the primitive itself completed but its claimed
    # observable outcome did not materialize, or the gain did not help the goal.
    if skill_success and (not matched or advanced is False):
        memory.record_failure(
            source="reflection",
            message=reason,
            arguments=dict(decision.arguments),
        )
    return record


def _goal_progress(
    memory: AgentMemory,
    decision: PlannerDecision,
    observed: dict[str, Any],
) -> tuple[bool | None, str]:
    skill = decision.name
    if skill in _POSITIONING_SKILLS:
        return None, ""
    delta = {
        str(name).casefold(): int(count)
        for name, count in dict(observed.get("inventory_delta") or {}).items()
        if int(count or 0) > 0
    }
    objective = " ".join(
        part
        for part in (memory.goal, memory.current_subgoal or "", decision.subgoal)
        if part
    ).casefold()
    if not delta:
        return False, "action produced no useful inventory change toward the current goal"
    gained = set(delta)
    if gained <= _FILLER_ITEMS and _objective_cares_about_progress(objective):
        items = ", ".join(sorted(gained))
        return False, f"obtained {items}, which does not advance {objective or 'the goal'}"
    if _looks_like_wood_task(objective) and gained <= _FILLER_ITEMS:
        items = ", ".join(sorted(gained))
        return False, f"obtained {items} while collecting wood; avoid unnecessary {items} mining"
    if _looks_like_iron_task(objective) and gained.isdisjoint(_IRON_PATH_ITEMS):
        items = ", ".join(sorted(gained))
        return False, f"obtained {items}, which does not advance iron-sword progress"
    return True, "inventory change is consistent with the current objective"


def _objective_cares_about_progress(objective: str) -> bool:
    return bool(objective) and (
        _looks_like_wood_task(objective) or _looks_like_iron_task(objective)
    )


def _looks_like_wood_task(text: str) -> bool:
    return any(term in text for term in _WOOD_TERMS)


def _looks_like_iron_task(text: str) -> bool:
    return "iron" in text and ("sword" in text or "ingot" in text or "ore" in text)


def _compare(expected: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    inventory = dict(observed.get("inventory") or {})
    delta = dict(observed.get("inventory_delta") or {})
    for item, count in _int_map(expected.get("inventory_min")).items():
        have = int(inventory.get(item, 0) or 0)
        if have < count:
            mismatches.append(f"expected at least {count} {item}, observed {have}")
    for item, count in _int_map(expected.get("inventory_delta")).items():
        have = int(delta.get(item, 0) or 0)
        if have < count:
            mismatches.append(f"expected inventory_delta {item}{count:+d}, observed {have:+d}")
    wanted = expected.get("selected_item")
    if wanted is not None and observed.get("selected_item") != wanted:
        mismatches.append(
            f"expected selected_item {wanted!r}, observed {observed.get('selected_item')!r}"
        )
    return mismatches


def _int_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for name, count in value.items():
        try:
            parsed = int(count)
        except (TypeError, ValueError):
            continue
        out[str(name)] = parsed
    return out


__all__ = ["reflect_skill_outcome"]

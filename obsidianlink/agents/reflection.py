"""Lightweight expected-vs-observed check after a primitive skill.

This is not a reflection framework, extra model call, or recovery policy.
It writes a short comparison into memory so the next planner decision can
change approach.
"""

from __future__ import annotations

from typing import Any

from obsidianlink.agents.memory import AgentMemory, ReflectionRecord
from obsidianlink.agents.planner import PlannerDecision
from obsidianlink.env.environment import Observation


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
    if matched:
        reason = skill_message or "observation matched the expected result"
    else:
        reason = "; ".join(mismatches)
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
    )
    memory.record_reflection(record)
    if not matched and skill_success:
        memory.record_failure(
            source="reflection",
            message=reason,
            arguments=dict(decision.arguments),
        )
    return record


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

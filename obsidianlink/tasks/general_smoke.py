"""Agent-local live smoke task definitions (not benchmark tasks)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from obsidianlink.agents.memory import AgentMemory
from obsidianlink.agents.planner import PlannerDecision
from obsidianlink.env.environment import Observation
from obsidianlink.skills.mining import log_count


@dataclass(frozen=True)
class GeneralAgentSmokeTask:
    task_id: str
    goal: str
    max_environment_steps: int
    max_skill_steps: int


COLLECT_ONE_LOG_SMOKE = GeneralAgentSmokeTask(
    task_id="general_live_collect_one_log",
    goal="Collect 1 log",
    max_environment_steps=600,
    max_skill_steps=520,
)

MINE_ONE_OBSIDIAN_SMOKE = GeneralAgentSmokeTask(
    task_id="general_live_mine_one_obsidian",
    goal="Mine 1 obsidian block",
    max_environment_steps=280,
    max_skill_steps=240,
)


def collect_wood_quantity(task: str) -> int:
    """Read a small English/Chinese wood-collection goal."""
    normalized = task.strip().lower()
    collection_word = any(
        word in normalized
        for word in ("collect", "gather", "get", "收集", "获取")
    )
    wood_word = any(
        word in normalized for word in ("wood", "log", "原木", "木头")
    )
    if not collection_word or not wood_word:
        raise ValueError("live smoke currently supports only wood collection tasks")
    match = re.search(r"\d+", normalized)
    return max(1, min(16, int(match.group(0)) if match else 1))


class CollectWoodSmokePlanner:
    """Deterministic Phase-1 planner for isolating real environment execution."""

    def __init__(self, *, max_skill_steps: int = 520) -> None:
        if max_skill_steps < 8:
            raise ValueError("max_skill_steps must be >= 8")
        self.max_skill_steps = int(max_skill_steps)
        self.planning_calls = 0

    def plan(
        self,
        memory: AgentMemory,
        observation: Observation,
        skill_descriptions: dict[str, str],
    ) -> PlannerDecision:
        self.planning_calls += 1
        if "collect_wood" not in skill_descriptions:
            raise ValueError("collect_wood skill is unavailable")
        target = collect_wood_quantity(memory.goal)
        current = log_count(dict(observation.inventory or {}))
        if current >= target:
            return PlannerDecision(
                "finish",
                reason=f"inventory contains {current}/{target} logs",
            )
        return PlannerDecision(
            "skill",
            name="collect_wood",
            arguments={"quantity": target, "max_steps": self.max_skill_steps},
            reason=f"inventory contains {current}/{target} logs",
        )


class MineObsidianSmokePlanner:
    """Deterministic planner for the controlled real-block smoke task."""

    def __init__(self, *, max_skill_steps: int = 240) -> None:
        if max_skill_steps < 8:
            raise ValueError("max_skill_steps must be >= 8")
        self.max_skill_steps = int(max_skill_steps)
        self.planning_calls = 0

    def plan(
        self,
        memory: AgentMemory,
        observation: Observation,
        skill_descriptions: dict[str, str],
    ) -> PlannerDecision:
        self.planning_calls += 1
        normalized = memory.goal.strip().lower()
        if "obsidian" not in normalized or not any(
            verb in normalized for verb in ("mine", "break", "挖", "破坏")
        ):
            raise ValueError("controlled smoke supports only obsidian mining")
        if "mine_block" not in skill_descriptions:
            raise ValueError("mine_block skill is unavailable")
        current = int((observation.inventory or {}).get("obsidian", 0) or 0)
        if current >= 1:
            return PlannerDecision("finish", reason="obsidian verified in inventory")
        return PlannerDecision(
            "skill",
            name="mine_block",
            arguments={
                "ticks": self.max_skill_steps,
                "item": "obsidian",
                "approach": False,
                "pickup_steps": 12,
                "settle_steps": 6,
            },
            reason="mine the fixed block under the crosshair",
        )


def collect_wood_goal_verified(
    task: str,
    _memory: AgentMemory,
    observation: Observation,
) -> bool:
    target = collect_wood_quantity(task)
    return log_count(dict(observation.inventory or {})) >= target


def obsidian_goal_verified(
    _task: str,
    _memory: AgentMemory,
    observation: Observation,
) -> bool:
    return int((observation.inventory or {}).get("obsidian", 0) or 0) >= 1


__all__ = [
    "COLLECT_ONE_LOG_SMOKE",
    "CollectWoodSmokePlanner",
    "GeneralAgentSmokeTask",
    "MINE_ONE_OBSIDIAN_SMOKE",
    "MineObsidianSmokePlanner",
    "collect_wood_goal_verified",
    "collect_wood_quantity",
    "obsidian_goal_verified",
]

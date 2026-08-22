from obsidianlink.agents.memory import AgentMemory, StepRecord
from obsidianlink.agents.planner import PlannerDecision
from obsidianlink.agents.validator import validate_skill_decision
from obsidianlink.env.environment import Observation


def test_validator_rejects_dirt_mining_during_collect_wood() -> None:
    memory = AgentMemory()
    memory.reset("make iron sword")
    memory.begin_subgoal("collect wood")
    decision = PlannerDecision(
        "skill",
        name="attack",
        arguments={"target": "dirt", "ticks": 16},
        subgoal="collect wood",
        expected={"inventory_min": {"dirt": 1}},
        reason="mine dirt",
    )

    verdict = validate_skill_decision(
        decision, memory, Observation(inventory={})
    )

    assert verdict.accepted is False
    assert verdict.code == "unrelated_to_subgoal"
    assert "cannot advance" in verdict.reason


def test_validator_allows_attack_when_collecting_wood_without_dirt_target() -> None:
    memory = AgentMemory()
    memory.reset("make iron sword")
    memory.begin_subgoal("collect wood")
    decision = PlannerDecision(
        "skill",
        name="attack",
        arguments={"ticks": 16},
        subgoal="collect wood",
        expected={"inventory_min": {"oak_log": 1}},
    )

    verdict = validate_skill_decision(decision, memory, Observation(inventory={}))

    assert verdict.accepted is True


def test_validator_rejects_place_without_selected_item() -> None:
    memory = AgentMemory()
    memory.reset("place a crafting table")
    memory.begin_subgoal("place table")
    decision = PlannerDecision("skill", name="place_block", subgoal="place table")

    verdict = validate_skill_decision(decision, memory, Observation(inventory={}))

    assert verdict.accepted is False
    assert verdict.code == "missing_prerequisite"


def test_validator_rejects_repeated_wait() -> None:
    memory = AgentMemory()
    memory.reset("collect wood")
    memory.begin_subgoal("find a tree")
    memory.record_step(StepRecord("wait", {"ticks": 4}, True, "waited", 4))
    memory.record_step(StepRecord("wait", {"ticks": 4}, True, "waited", 4))

    verdict = validate_skill_decision(
        PlannerDecision("skill", name="wait", arguments={"ticks": 8}),
        memory,
        Observation(inventory={}),
    )

    assert verdict.accepted is False
    assert verdict.code == "no_progress"

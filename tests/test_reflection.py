from obsidianlink.agents.memory import AgentMemory
from obsidianlink.agents.planner import PlannerDecision
from obsidianlink.agents.reflection import reflect_skill_outcome
from obsidianlink.env.environment import Observation


def test_reflection_records_expected_inventory_mismatch() -> None:
    memory = AgentMemory()
    memory.reset("Mine 1 cobblestone")
    memory.update_state(Observation(inventory={}), baseline={})
    memory.begin_subgoal("mine stone")
    decision = PlannerDecision(
        "skill",
        name="attack",
        arguments={"ticks": 8},
        subgoal="mine stone",
        expected={"inventory_min": {"cobblestone": 1}},
    )

    record = reflect_skill_outcome(
        memory,
        decision,
        Observation(inventory={}),
        skill_success=True,
        skill_message="attacked for 8/8 ticks",
    )

    assert record.matched is False
    assert memory.last_reflection is record
    assert "expected at least 1 cobblestone" in record.reason
    assert memory.failed_attempts[-1].source == "reflection"


def test_reflection_matches_when_observation_meets_expectation() -> None:
    memory = AgentMemory()
    memory.reset("Mine 1 cobblestone")
    memory.update_state(Observation(inventory={"cobblestone": 1}), baseline={})
    decision = PlannerDecision(
        "skill",
        name="attack",
        expected={"inventory_min": {"cobblestone": 1}},
    )

    record = reflect_skill_outcome(
        memory,
        decision,
        Observation(inventory={"cobblestone": 1}),
        skill_success=True,
        skill_message="attacked",
    )

    assert record.matched is True
    assert record.advanced_goal is True
    assert memory.failed_attempts == []


def test_reflection_marks_dirt_gain_as_not_advancing_iron_sword() -> None:
    memory = AgentMemory()
    memory.reset("Find iron and craft iron sword")
    memory.update_state(Observation(inventory={"dirt": 1}), baseline={})
    memory.begin_subgoal("collect wood")
    decision = PlannerDecision(
        "skill",
        name="attack",
        subgoal="collect wood",
        expected={"inventory_min": {"dirt": 1}},
    )

    record = reflect_skill_outcome(
        memory,
        decision,
        Observation(inventory={"dirt": 1}),
        skill_success=True,
        skill_message="obtained dirt",
    )

    assert record.matched is True
    assert record.advanced_goal is False
    assert "does not advance" in record.progress_note
    assert memory.failed_attempts[-1].source == "reflection"

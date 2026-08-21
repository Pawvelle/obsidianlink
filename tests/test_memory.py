from obsidianlink.agents.memory import AgentMemory, StepRecord
from obsidianlink.env.environment import Observation


def test_memory_prompt_state_is_decision_state_not_an_action_log() -> None:
    memory = AgentMemory()
    memory.reset("Mine 1 cobblestone")
    memory.update_state(Observation(inventory={"dirt": 1}, selected_item="dirt"))
    memory.begin_subgoal("look for stone")
    memory.remember_knowledge("cobblestone", "Mine stone with a pickaxe.")
    memory.record_step(
        StepRecord(
            skill="attack",
            arguments={"ticks": 8},
            success=False,
            message="no block broke",
            environment_steps=8,
            metadata={"inventory_before": {"dirt": 1}, "inventory_after": {"dirt": 1}},
        )
    )
    memory.begin_subgoal("break stone under crosshair")

    state = memory.prompt_state()

    assert state["task"] == "Mine 1 cobblestone"
    assert state["task_status"] == "in_progress"
    assert state["current_subgoal"] == "break stone under crosshair"
    assert "look for stone" not in state["completed_subgoals"]
    assert state["wiki_knowledge"]["cobblestone"] == "Mine stone with a pickaxe."
    assert state["recent_failures"][0]["source"] == "attack"
    assert state["recent_failures"][0]["message"] == "no block broke"
    assert state["environment"]["inventory"] == {"dirt": 1}
    assert state["environment"]["selected_item"] == "dirt"
    assert "inventory_before" not in str(state)
    assert "metadata" not in state["recent_skills"][0]


def test_memory_tracks_completed_subgoals_and_inventory_delta() -> None:
    memory = AgentMemory()
    memory.reset("Collect cobblestone")
    memory.update_state(Observation(inventory={}))
    memory.begin_subgoal("mine stone")
    memory.update_state(Observation(inventory={"cobblestone": 2}), baseline={})
    memory.begin_subgoal("check inventory")

    assert memory.completed_subgoals == ["mine stone"]
    assert memory.current_subgoal == "check inventory"
    assert memory.inventory_delta == {"cobblestone": 2}

    memory.mark_task_completed()
    assert memory.completed_subgoals == ["mine stone", "check inventory"]
    assert memory.current_subgoal is None
    assert memory.task_status == "completed"


def test_successful_skill_clears_last_error_but_keeps_failure_history() -> None:
    memory = AgentMemory()
    memory.reset("Collect cobblestone")
    memory.begin_subgoal("mine stone")
    memory.record_step(
        StepRecord("attack", {}, False, "missed", 1)
    )
    memory.record_step(
        StepRecord("attack", {"ticks": 20}, True, "mined cobblestone", 20)
    )

    assert memory.last_error is None
    assert [item.source for item in memory.failed_attempts] == ["attack"]
    assert memory.prompt_state()["recent_skills"][-1]["success"] is True

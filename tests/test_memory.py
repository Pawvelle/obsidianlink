from obsidianlink.agents.memory import AgentMemory, ReflectionRecord, StepRecord
from obsidianlink.agents.planner import PlannedSubgoal
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


def test_memory_tracks_pending_subgoals_knowledge_usage_and_reflection() -> None:
    memory = AgentMemory()
    memory.reset("Mine 1 cobblestone")
    memory.update_state(Observation(inventory={}))
    memory.apply_plan("learn cobblestone rule", ("approach stone", "mine stone"))
    memory.remember_knowledge("cobblestone", "Mine stone with a pickaxe.")
    memory.record_reflection(
        ReflectionRecord(
            skill="attack",
            subgoal="mine stone",
            matched=False,
            reason="expected at least 1 cobblestone, observed 0",
        )
    )

    state = memory.prompt_state()
    assert state["subgoal_progress"]["current"] == "learn cobblestone rule"
    assert state["subgoal_progress"]["pending"] == ["approach stone", "mine stone"]
    assert state["knowledge_usage"]["retrieved"]["cobblestone"] == "Mine stone with a pickaxe."
    assert state["knowledge_usage"]["recent"][0]["query"] == "cobblestone"
    assert state["last_reflection"]["matched"] is False
    assert "expected at least 1 cobblestone" in state["last_reflection"]["reason"]


def test_long_term_memory_survives_working_memory_reset() -> None:
    memory = AgentMemory()
    memory.reset("First task")
    memory.remember_knowledge(
        "wooden pickaxe recipe",
        "Wooden Pickaxe: crafted from planks and sticks",
        knowledge_type="recipe",
        subject="Wooden Pickaxe",
        attributes={"ingredients": ("planks", "sticks")},
    )
    memory.remember_location(
        "oak grove",
        position=(10.0, 64.0, -4.0),
        resources={"oak_log": 6},
    )
    memory.record_failure(source="attack", message="target was out of reach")

    memory.reset("Second task")
    state = memory.prompt_state()

    assert memory.find_knowledge("  WOODEN PICKAXE RECIPE ") is not None
    assert state["semantic_memory"][0]["knowledge_type"] == "recipe"
    assert state["episodic_memory"][0]["outcome"] == "target was out of reach"
    assert state["spatial_memory"][0]["resources"] == {"oak_log": 6}
    assert state["working_memory"]["task"] == "Second task"
    assert state["working_memory"]["inventory"] == {}


def test_memory_merges_hierarchical_plan_and_tracks_attempts() -> None:
    memory = AgentMemory()
    memory.reset("Craft a tool")
    memory.apply_plan(
        "obtain materials",
        plan=(
            PlannedSubgoal("materials", "obtain materials", "in_progress"),
            PlannedSubgoal("craft", "craft tool", "pending", depends_on=("materials",)),
        ),
        active_subgoal_id="materials",
        revision_reason="initial decomposition",
    )
    memory.record_step(StepRecord("attack", {"ticks": 4}, True, "got logs", 4))

    state = memory.prompt_state()
    assert memory.active_subgoal_id == "materials"
    assert memory.subgoal_states["materials"].attempts == 1
    assert memory.pending_subgoals == ["craft tool"]
    assert state["working_memory"]["plan_revision"] == 1
    assert state["working_memory"]["recent_plan_revisions"][0]["statuses"] == {
        "materials": "in_progress",
        "craft": "pending",
    }
    assert state["subgoal_progress"]["nodes"][1]["depends_on"] == ("materials",)

    memory.apply_plan(
        "craft tool",
        plan=(
            PlannedSubgoal("materials", "obtain materials", "completed"),
            PlannedSubgoal("craft", "craft tool", "in_progress", depends_on=("materials",)),
        ),
        active_subgoal_id="craft",
    )
    memory.apply_plan(
        "obtain materials",
        plan=(PlannedSubgoal("materials", "obtain materials", "pending"),),
        active_subgoal_id="materials",
    )
    assert memory.subgoal_states["materials"].status == "completed"


def test_memory_retrieval_ranks_relevant_semantic_episodic_and_spatial_records() -> None:
    memory = AgentMemory()
    memory.reset("Mine iron")
    memory.remember_knowledge(
        "iron ore item",
        "Iron ore requires a stone pickaxe or better.",
        knowledge_type="item",
        subject="Iron Ore",
    )
    memory.remember_knowledge(
        "bread recipe",
        "Bread is crafted from wheat.",
        knowledge_type="recipe",
        subject="Bread",
    )
    memory.remember_location(
        "nearby iron vein",
        position=(12.0, 20.0, -3.0),
        resources={"iron_ore": 5},
        notes="iron vein below the stone ridge",
    )
    memory.begin_subgoal("mine iron ore", subgoal_id="mine")
    memory.record_failure(
        source="attack",
        message="iron ore did not break with wooden pickaxe",
        arguments={"ticks": 20},
    )

    result = memory.retrieve("mine iron ore", limit=4)

    assert {item.memory_type for item in result.items} == {
        "semantic",
        "episodic",
        "spatial",
    }
    assert all("bread" not in item.summary.casefold() for item in result.items)
    assert result.items[0].score >= result.items[-1].score
    assert memory.prompt_state()["retrieved_memory"]["query"] == "mine iron ore"
    assert memory.subgoal_states["mine"].status == "failed"
    assert memory.subgoal_states["mine"].failures == 1


def test_memory_exposes_next_objective_and_decision_retrieval() -> None:
    memory = AgentMemory()
    memory.reset("craft iron sword")
    memory.apply_plan(
        "collect wood",
        ("craft basic tools", "find iron"),
    )
    memory.remember_knowledge("oak logs", "Punch a tree to get logs.")
    memory.record_failure(source="attack", message="mined dirt instead of wood")

    retrieval = memory.retrieve_for_decision(limit=4)
    state = memory.prompt_state()

    assert memory.next_objective == "craft basic tools"
    assert state["working_memory"]["current_goal"] == "craft iron sword"
    assert "collect wood" in retrieval.query
    assert any(item.memory_type == "episodic" for item in retrieval.items)


def test_memory_retrieval_can_limit_memory_types() -> None:
    memory = AgentMemory()
    memory.reset("Find oak logs")
    memory.remember_knowledge("oak logs", "Oak logs come from oak trees.")
    memory.remember_location("oak grove", resources={"oak_log": 4})

    result = memory.retrieve("oak logs", memory_types=("spatial",), limit=2)

    assert [item.memory_type for item in result.items] == ["spatial"]

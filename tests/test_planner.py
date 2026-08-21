from obsidianlink.agents.memory import AgentMemory, StepRecord
from obsidianlink.agents.planner import _build_planner_prompt, parse_planner_decision
from obsidianlink.env.environment import Observation


def test_planner_parser_accepts_optional_subgoal() -> None:
    decision = parse_planner_decision(
        '{"type":"skill","subgoal":"break the block ahead","name":"attack","arguments":{"ticks":20}}',
        frozenset({"attack"}),
    )
    assert decision.subgoal == "break the block ahead"
    assert decision.name == "attack"
    assert decision.arguments == {"ticks": 20}


def test_planner_prompt_exposes_observation_memory_and_subgoal_loop() -> None:
    memory = AgentMemory()
    memory.reset("Mine 1 cobblestone")
    memory.update_state(Observation(inventory={"dirt": 1}, selected_item="dirt"))
    memory.begin_subgoal("find stone")
    memory.remember_knowledge("cobblestone", "Mine stone with a pickaxe.")
    memory.record_step(
        StepRecord(
            skill="attack",
            arguments={"ticks": 8},
            success=False,
            message="no block broke",
            environment_steps=8,
            metadata={"inventory_before": {"secret": 99}},
        )
    )
    observation = Observation(
        frame=object(),
        inventory={"dirt": 1},
        selected_item="dirt",
    )

    prompt = _build_planner_prompt(
        memory,
        {"attack": "Primitive mining/attack interaction."},
        observation=observation,
    )

    assert "remaining subgoals → current subgoal" in prompt
    assert "subgoal_progress" in prompt
    assert "last_reflection" in prompt
    assert "knowledge_usage" in prompt
    assert "recent_failures" in prompt
    assert "wiki_knowledge" in prompt
    assert '"has_visual_frame": true' in prompt
    assert '"dirt": 1' in prompt
    assert "secret" not in prompt
    assert "no block broke" in prompt
    assert "Mine stone with a pickaxe." in prompt


def test_planner_parser_accepts_pending_subgoals_and_expected_outcome() -> None:
    decision = parse_planner_decision(
        '{"type":"skill","subgoal":"mine stone","pending_subgoals":["verify inventory"],'
        '"name":"attack","arguments":{"ticks":8},"expected":{"inventory_min":{"cobblestone":1}}}',
        frozenset({"attack"}),
    )
    assert decision.subgoal == "mine stone"
    assert decision.pending_subgoals == ("verify inventory",)
    assert decision.expected == {"inventory_min": {"cobblestone": 1}}


def test_planner_parser_accepts_hierarchical_plan_update() -> None:
    decision = parse_planner_decision(
        '{"type":"skill","name":"attack","arguments":{"ticks":8},'
        '"active_subgoal_id":"mine","plan_revision_reason":"stone is now reachable",'
        '"plan":[{"id":"find","description":"find stone","status":"completed"},'
        '{"id":"mine","description":"mine stone","status":"in_progress",'
        '"depends_on":["find"]}]}',
        frozenset({"attack"}),
    )
    assert decision.active_subgoal_id == "mine"
    assert decision.plan[0].status == "completed"
    assert decision.plan[1].depends_on == ("find",)
    assert decision.plan_revision_reason == "stone is now reachable"


def test_planner_parser_accepts_bounded_memory_retrieval() -> None:
    decision = parse_planner_decision(
        '{"type":"memory","query":"failed attempts mining iron",'
        '"memory_types":["episodic","semantic","unknown","episodic"],'
        '"retrieval_limit":99,"reason":"recover from failure"}',
        frozenset(),
    )

    assert decision.type == "memory"
    assert decision.memory_types == ("episodic", "semantic")
    assert decision.retrieval_limit == 12

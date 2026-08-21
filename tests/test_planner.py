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

    assert "task → current subgoal" in prompt
    assert "completed_subgoals" in prompt
    assert "recent_failures" in prompt
    assert "wiki_knowledge" in prompt
    assert '"has_visual_frame": true' in prompt
    assert '"dirt": 1' in prompt
    assert "secret" not in prompt
    assert "no block broke" in prompt
    assert "Mine stone with a pickaxe." in prompt

from obsidianlink.agents.memory import AgentMemory, ReflectionRecord, StepRecord
from obsidianlink.agents.observation import build_grounded_observation
from obsidianlink.env.environment import Observation
from obsidianlink.env.fake import FakeMinecraftEnv


def test_grounded_observation_includes_goal_last_action_and_feedback() -> None:
    memory = AgentMemory()
    memory.reset("craft iron sword")
    memory.begin_subgoal("collect wood")
    memory.update_state(Observation(inventory={"dirt": 1}, selected_item="dirt"))
    memory.record_step(
        StepRecord(
            skill="attack",
            arguments={"target": "dirt", "ticks": 8},
            success=True,
            message="attacked dirt",
            environment_steps=8,
        )
    )
    memory.record_reflection(
        ReflectionRecord(
            skill="attack",
            subgoal="collect wood",
            matched=True,
            reason="got dirt",
            advanced_goal=False,
            progress_note="dirt does not advance collect wood",
        )
    )
    env = FakeMinecraftEnv(target="oak_log", distance=2, remaining=3)
    env.reset()

    grounded = build_grounded_observation(
        memory,
        Observation(inventory={"dirt": 1}, selected_item="dirt"),
        local_view=env.local_view(),
    )
    view = grounded.as_prompt()

    assert view["goal"] == "craft iron sword"
    assert view["subgoal"] == "collect wood"
    assert view["next_objective"]
    assert view["inventory"] == {"dirt": 1}
    assert view["equipment"]["mainhand"] == "dirt"
    assert view["nearby_blocks"][0]["name"] == "oak_log"
    assert view["visible_resources"] == ["oak_log"]
    assert view["last_action"]["skill"] == "attack"
    assert view["action_result"]["message"] == "attacked dirt"
    assert view["feedback"]["advanced_goal"] is False
    assert view["position"]["relative"] == "near"
    assert "xpos" not in view
    assert "frame" not in view
    assert view["health"]["status"] == "unknown"


def test_grounded_observation_does_not_copy_evaluator_pose() -> None:
    memory = AgentMemory()
    memory.reset("look around")
    view = build_grounded_observation(
        memory,
        Observation(inventory={}),
        local_view={"position": {"xpos": 12.0, "relative": "near"}},
    ).as_prompt()

    assert "xpos" not in view["position"]
    assert view["position"]["relative"] == "near"

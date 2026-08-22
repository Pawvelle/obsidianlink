from __future__ import annotations

from obsidianlink.agents.planner import PlannerDecision
from obsidianlink.env.fake import FakeMinecraftEnv
from obsidianlink.voyager import (
    CurriculumObjective,
    InventoryCurriculum,
    InventoryCritic,
    MineDojoVoyager,
    VoyagerTask,
)


class _MineOneBlockPlanner:
    def __init__(self) -> None:
        self.retrieval = None

    def plan(self, memory, _observation, _skills):
        self.retrieval = memory.last_retrieval
        return PlannerDecision(
            "skill",
            name="attack",
            arguments={"ticks": 1},
            subgoal="mine the block under the crosshair",
            expected={"inventory_min": {"cobblestone": 1}},
        )


def _has_cobblestone(_task, _memory, observation) -> bool:
    return int((observation.inventory or {}).get("cobblestone", 0)) >= 1


def _has_two_cobblestone(_task, _memory, observation) -> bool:
    return int((observation.inventory or {}).get("cobblestone", 0)) >= 2


def test_minedojo_voyager_runs_through_primitive_agent_loop() -> None:
    environment = FakeMinecraftEnv(target="stone", mine_ticks=1, remaining=2)
    voyager = MineDojoVoyager(
        environment,
        planner_factory=_MineOneBlockPlanner,
        max_steps=8,
        max_planning_cycles=2,
    )

    episode = voyager.run_task(
        VoyagerTask("mine_one_cobblestone", "Collect 1 cobblestone", _has_cobblestone)
    )

    assert episode.result.success is True
    assert episode.result.inventory["cobblestone"] == 1
    assert [step.skill for step in episode.result.completed_steps] == ["attack"]
    assert episode.action_counts == {"attack": 1}
    assert voyager.skill_memory.as_dict()["attack"]


def test_minedojo_voyager_retrieves_observed_primitive_skills_for_next_task() -> None:
    environment = FakeMinecraftEnv(target="stone", mine_ticks=1, remaining=2)
    voyager = MineDojoVoyager(
        environment,
        planner_factory=_MineOneBlockPlanner,
        max_steps=8,
        max_planning_cycles=2,
    )
    task = VoyagerTask("mine_one_cobblestone", "Collect 1 cobblestone", _has_cobblestone)

    episodes = voyager.learn([task, VoyagerTask("mine_again", "Mine cobblestone", _has_cobblestone)])

    assert len(episodes) == 2
    assert episodes[1].retrieved_skills == ("attack",)
    voyager.close()


def test_retrieved_skill_is_injected_into_planner_memory() -> None:
    planners: list[_MineOneBlockPlanner] = []

    def factory() -> _MineOneBlockPlanner:
        planner = _MineOneBlockPlanner()
        planners.append(planner)
        return planner

    environment = FakeMinecraftEnv(target="stone", mine_ticks=1, remaining=2)
    voyager = MineDojoVoyager(environment, factory, max_steps=8, max_planning_cycles=2)
    task = VoyagerTask("mine_one", "Collect 1 cobblestone", _has_cobblestone)
    voyager.run_task(task)
    voyager.run_task(VoyagerTask("mine_two", "Collect 2 cobblestone", _has_two_cobblestone))

    retrieval = planners[1].retrieval
    assert retrieval is not None
    assert retrieval.items[0].metadata["knowledge_type"] == "verified_skill"


def test_inventory_curriculum_advances_only_after_verified_result() -> None:
    curriculum = InventoryCurriculum(
        (
            CurriculumObjective("first", "Collect 1 cobblestone", {"cobblestone": 1}),
            CurriculumObjective("second", "Collect 2 cobblestone", {"cobblestone": 2}),
        )
    )
    environment = FakeMinecraftEnv(target="stone", mine_ticks=1, remaining=4)
    voyager = MineDojoVoyager(
        environment,
        planner_factory=_MineOneBlockPlanner,
        max_steps=8,
        max_planning_cycles=2,
    )

    episodes = voyager.learn_curriculum(curriculum, max_tasks=2)

    assert [episode.task_id for episode in episodes] == ["first", "second"]
    assert all(episode.result.success for episode in episodes)
    assert episodes[-1].result.inventory["cobblestone"] == 2


def test_voyager_checkpoint_restores_portable_skill_memory(tmp_path) -> None:
    environment = FakeMinecraftEnv(target="stone", mine_ticks=1, remaining=2)
    voyager = MineDojoVoyager(
        environment, planner_factory=_MineOneBlockPlanner, max_steps=8, max_planning_cycles=2
    )
    voyager.run_task(VoyagerTask("mine", "Collect 1 cobblestone", _has_cobblestone))
    checkpoint = voyager.checkpoint(tmp_path / "voyager.json")

    restored = MineDojoVoyager.load_skill_memory(checkpoint)

    assert restored.retrieve("collect cobblestone") == ("attack",)


def test_inventory_critic_reports_only_visible_missing_items() -> None:
    critic = InventoryCritic({"oak_log": 2})
    observation = FakeMinecraftEnv(inventory={"oak_log": 1}).reset()

    outcome = critic.evaluate(observation)

    assert outcome.success is False
    assert outcome.missing == {"oak_log": 1}

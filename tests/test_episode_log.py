from obsidianlink.agents.episode_log import EpisodeLogger
from obsidianlink.agents.general_agent import GeneralAgent
from obsidianlink.agents.planner import PlannerDecision
from obsidianlink.controller.minecraft_controller import MinecraftController
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.skills.base import SkillLibrary, SkillResult
from tests.test_general_agent import SequencePlanner


class EmptyEnv(Environment):
    def reset(self) -> Observation:
        return Observation(inventory={})

    def observe(self) -> Observation:
        return Observation(inventory={})

    def step(self, action: Action) -> Observation:
        assert action.type is ActionType.WAIT
        return Observation(inventory={})

    def close(self) -> None:
        return None


class WaitSkill:
    name = "wait"
    description = "Wait."

    def execute(self, controller, memory, arguments):
        start = controller.steps
        controller.step(Action(ActionType.WAIT))
        return SkillResult(True, "waited", controller.steps - start)


def test_episode_logger_writes_events_and_summary(tmp_path) -> None:
    logger = EpisodeLogger(tmp_path / "episode_test")
    logger.record("task", {"task": "demo"})
    logger.write_summary({"success": False})

    lines = (tmp_path / "episode_test" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert '"event": "task"' in lines[0]
    assert (tmp_path / "episode_test" / "summary.json").is_file()


def test_general_agent_writes_episode_trace(tmp_path) -> None:
    logger = EpisodeLogger(tmp_path / "episode_agent")
    agent = GeneralAgent(
        SequencePlanner(
            [
                PlannerDecision("skill", name="wait", arguments={"ticks": 1}),
                PlannerDecision("finish"),
            ]
        ),
        MinecraftController(EmptyEnv(), max_steps=8),
        skills=SkillLibrary([WaitSkill()]),
        episode_logger=logger,
    )

    result = agent.run("Stand still briefly")

    assert result.success is True
    text = logger.events_path.read_text(encoding="utf-8")
    assert '"event": "task"' in text
    assert '"event": "planner_output"' in text
    assert '"event": "validation"' in text
    assert '"event": "skill_execution"' in text
    assert '"event": "result"' in text
    assert logger.summary_path.is_file()

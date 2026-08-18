"""Offline smoke tests. No MineRL, Minecraft, Gradle, or network."""

import obsidianlink
from obsidianlink.agents.reactive import ReactiveAgent
from obsidianlink.benchmark.evaluator import Evaluator
from obsidianlink.benchmark.result import Result
from obsidianlink.benchmark.runner import BenchmarkRunner
from obsidianlink.benchmark.task import Task
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation


class _StubModel:
    def complete(self, prompt: str) -> str:
        del prompt
        return "wait"


class _StubAgent:
    def act(self, observation: Observation) -> Action:
        del observation
        return Action(type=ActionType.WAIT)


class _StubEnvironment(Environment):
    def reset(self) -> Observation:
        return Observation()

    def step(self, action: Action) -> Observation:
        del action
        return Observation()

    def close(self) -> None:
        return None


class _StubEvaluator(Evaluator):
    def evaluate(
        self,
        task: Task,
        *,
        steps: int,
        model_calls: int,
        invalid_actions: int,
        elapsed_time: float,
    ) -> Result:
        return Result(
            task_id=task.task_id,
            success=True,
            steps=steps,
            model_calls=model_calls,
            invalid_actions=invalid_actions,
            elapsed_time=elapsed_time,
        )


def test_import_obsidianlink() -> None:
    assert obsidianlink.__version__


def test_action_constructs() -> None:
    action = Action(type=ActionType.WAIT)
    assert action.type is ActionType.WAIT


def test_observation_constructs() -> None:
    observation = Observation(frame=None, inventory=None, selected_item=None)
    assert observation.frame is None


def test_task_constructs() -> None:
    task = Task(task_id="smoke", goal="construct a portal", max_steps=8)
    assert task.task_id == "smoke"
    assert task.max_steps == 8


def test_result_constructs() -> None:
    result = Result(
        task_id="smoke",
        success=False,
        steps=0,
        model_calls=0,
        invalid_actions=0,
        elapsed_time=0.0,
    )
    assert result.success is False


def test_stub_runner_flow() -> None:
    result = BenchmarkRunner().run(
        task=Task(task_id="smoke", goal="offline loop", max_steps=2),
        env=_StubEnvironment(),
        agent=_StubAgent(),
        evaluator=_StubEvaluator(),
    )
    assert result.task_id == "smoke"
    assert result.success is True
    assert result.steps == 2
    assert result.invalid_actions == 0
    assert result.elapsed_time >= 0.0


def test_reactive_agent_uses_model_client() -> None:
    agent = ReactiveAgent(_StubModel())
    action = agent.act(Observation())
    assert action.type is ActionType.WAIT
    assert agent.model_calls == 1

from obsidianlink.agents.heuristic import HeuristicModelClient
from obsidianlink.agents.reactive import ReactiveAgent
from obsidianlink.benchmark.evaluator import Evaluator
from obsidianlink.benchmark.result import (
    AGENT_FAILURE,
    ENVIRONMENT_FAILURE,
    EVALUATOR_FAILURE,
    Result,
)
from obsidianlink.benchmark.runner import BenchmarkRunner, clamp_to_allowed
from obsidianlink.benchmark.task import Task
from obsidianlink.env.actions import Action, ActionType
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.tasks.diagnostic import D1_LAVA_POSITIVE, D1LavaEvaluator, d1_prompt


class _StubEnv(Environment):
    def __init__(self, *, hidden: dict | None = None, fail: bool = False) -> None:
        self.hidden_state = hidden or {"target_truths": {"lava": True}, "ypos": 101.0}
        self.closed = False
        self._obs = Observation(frame=b"rgb", inventory={}, selected_item=None)
        self.fail = fail
        self.fail_step = False
        self.steps = 0
        self.last_action: Action | None = None

    def reset(self) -> Observation:
        if self.fail:
            raise RuntimeError("boom")
        return self._obs

    def observe(self) -> Observation:
        return self._obs

    def step(self, action: Action) -> Observation:
        if self.fail_step:
            raise RuntimeError("step-boom")
        self.last_action = action
        self.steps += 1
        return self._obs

    def close(self) -> None:
        self.closed = True


class _VisionYes:
    def complete(self, prompt: str) -> str:
        raise AssertionError("text path")

    def complete_with_vision(self, prompt: str, *, frame: object) -> str:
        del prompt, frame
        return '{"action": "wait", "visible": true}'


def test_runner_closes_env_and_forwards_metrics() -> None:
    env = _StubEnv()
    agent = ReactiveAgent(model=_VisionYes(), prompt_builder=d1_prompt)
    result = BenchmarkRunner().run(
        task=D1_LAVA_POSITIVE,
        env=env,
        agent=agent,
        evaluator=D1LavaEvaluator(),
    )
    assert env.closed is True
    assert env.steps == 1
    assert result.success is True
    assert result.steps == 1
    assert result.model_calls == 1
    assert result.invalid_actions == 0
    assert result.evidence["used_vision"] is True
    assert result.evidence["vision_calls"] == 1


def test_runner_records_environment_failure_and_still_closes() -> None:
    env = _StubEnv(fail=True)
    agent = ReactiveAgent(model=HeuristicModelClient())
    result = BenchmarkRunner().run(
        task=D1_LAVA_POSITIVE,
        env=env,
        agent=agent,
        evaluator=D1LavaEvaluator(),
    )
    assert env.closed is True
    assert result.success is False
    assert result.evidence["failure_class"] == ENVIRONMENT_FAILURE
    assert result.evidence["reason"] == "environment_exception"


def test_runner_records_agent_failure_without_raising() -> None:
    class _BoomAgent:
        model_calls = 0
        invalid_actions = 0

        def act(self, observation: Observation) -> Action:
            del observation
            raise RuntimeError("model-boom")

    env = _StubEnv()
    result = BenchmarkRunner().run(
        task=D1_LAVA_POSITIVE,
        env=env,
        agent=_BoomAgent(),
        evaluator=D1LavaEvaluator(),
    )
    assert env.closed is True
    assert result.success is False
    assert result.evidence["failure_class"] == AGENT_FAILURE
    assert result.evidence["reason"] == "agent_exception"
    assert env.steps == 0


def test_runner_records_evaluator_failure_without_raising() -> None:
    class _BoomEvaluator(Evaluator):
        def evaluate(self, task: Task, **kwargs: object) -> Result:
            del task, kwargs
            raise RuntimeError("eval-boom")

    env = _StubEnv()
    agent = ReactiveAgent(model=_VisionYes(), prompt_builder=d1_prompt)
    result = BenchmarkRunner().run(
        task=D1_LAVA_POSITIVE,
        env=env,
        agent=agent,
        evaluator=_BoomEvaluator(),
    )
    assert env.closed is True
    assert result.success is False
    assert result.evidence["failure_class"] == EVALUATOR_FAILURE
    assert result.evidence["reason"] == "evaluator_exception"


def test_runner_clamps_disallowed_action_to_wait() -> None:
    class _MoveVision:
        def complete(self, prompt: str) -> str:
            raise AssertionError("text path")

        def complete_with_vision(self, prompt: str, *, frame: object) -> str:
            del prompt, frame
            return '{"action": "move", "dx": 1, "visible": true}'

    env = _StubEnv()
    agent = ReactiveAgent(model=_MoveVision(), prompt_builder=d1_prompt)
    result = BenchmarkRunner().run(
        task=D1_LAVA_POSITIVE,
        env=env,
        agent=agent,
        evaluator=D1LavaEvaluator(),
    )
    assert env.last_action is not None
    assert env.last_action.type is ActionType.WAIT
    assert result.invalid_actions == 1
    assert result.evidence["disallowed_action"] == "move"
    assert result.success is True


def test_runner_records_step_environment_failure() -> None:
    env = _StubEnv()
    env.fail_step = True
    agent = ReactiveAgent(model=_VisionYes(), prompt_builder=d1_prompt)
    result = BenchmarkRunner().run(
        task=D1_LAVA_POSITIVE,
        env=env,
        agent=agent,
        evaluator=D1LavaEvaluator(),
    )
    assert env.closed is True
    assert result.success is False
    assert result.evidence["failure_class"] == ENVIRONMENT_FAILURE
    assert result.evidence["reason"] == "environment_exception"


def test_clamp_empty_allowed_is_unrestricted() -> None:
    action = Action(type=ActionType.MOVE, dx=1)
    out, disallowed = clamp_to_allowed(action, ())
    assert disallowed is None
    assert out.type is ActionType.MOVE

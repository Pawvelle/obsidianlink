from obsidianlink.agents.heuristic import HeuristicModelClient
from obsidianlink.agents.reactive import ReactiveAgent
from obsidianlink.benchmark.result import ENVIRONMENT_FAILURE
from obsidianlink.benchmark.runner import BenchmarkRunner
from obsidianlink.env.actions import Action
from obsidianlink.env.environment import Environment, Observation
from obsidianlink.tasks.diagnostic import D1_LAVA_POSITIVE, D1LavaEvaluator, d1_prompt


class _StubEnv(Environment):
    def __init__(self, *, hidden: dict | None = None, fail: bool = False) -> None:
        self.hidden_state = hidden or {"target_truths": {"lava": True}, "ypos": 101.0}
        self.closed = False
        self._obs = Observation(frame=b"rgb", inventory={}, selected_item=None)
        self.fail = fail
        self.steps = 0

    def reset(self) -> Observation:
        if self.fail:
            raise RuntimeError("boom")
        return self._obs

    def observe(self) -> Observation:
        return self._obs

    def step(self, action: Action) -> Observation:
        del action
        self.steps += 1
        return self._obs

    def close(self) -> None:
        self.closed = True


class _VisionYes:
    def complete(self, prompt: str) -> str:
        raise AssertionError("text path")

    def complete_with_vision(self, prompt: str, *, frame: object) -> str:
        del prompt, frame
        return '{"visible": true}'


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

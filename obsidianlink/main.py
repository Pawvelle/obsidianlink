"""Offline smoke entry point. Does not start Minecraft."""

from obsidianlink.agents.reactive import ReactiveAgent
from obsidianlink.benchmark.evaluator import Evaluator
from obsidianlink.benchmark.result import Result
from obsidianlink.benchmark.runner import BenchmarkRunner
from obsidianlink.benchmark.task import Task
from obsidianlink.env.actions import Action
from obsidianlink.env.environment import Environment, Observation


class _StubModel:
    def complete(self, prompt: str) -> str:
        del prompt
        return "wait"


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


def main() -> None:
    print("ObsidianLink")
    print("Current status: clean restart / minimal research infrastructure.")
    print("Phase 0 / Clean Restart: COMPLETE")
    print("Phase 1 / Minimal Minecraft Agent Loop: NOT STARTED")

    result = BenchmarkRunner().run(
        task=Task(task_id="smoke", goal="offline skeleton check", max_steps=1),
        env=_StubEnvironment(),
        agent=ReactiveAgent(_StubModel()),
        evaluator=_StubEvaluator(),
    )
    print(
        f"smoke: success={result.success} steps={result.steps} "
        f"model_calls={result.model_calls}"
    )


if __name__ == "__main__":
    main()

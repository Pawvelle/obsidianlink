"""Minimal benchmark loop. Does not start Minecraft."""

import time

from obsidianlink.agents.base import Agent
from obsidianlink.benchmark.evaluator import Evaluator
from obsidianlink.benchmark.result import Result
from obsidianlink.benchmark.task import Task
from obsidianlink.env.environment import Environment


class BenchmarkRunner:
    def run(
        self,
        task: Task,
        env: Environment,
        agent: Agent,
        evaluator: Evaluator,
    ) -> Result:
        started = time.perf_counter()
        observation = env.reset()
        steps = 0
        try:
            for _ in range(task.max_steps):
                action = agent.act(observation)
                observation = env.step(action)
                steps += 1
        finally:
            env.close()

        model_calls = getattr(agent, "model_calls", 0)
        return evaluator.evaluate(
            task,
            steps=steps,
            model_calls=model_calls,
            invalid_actions=0,
            elapsed_time=time.perf_counter() - started,
        )

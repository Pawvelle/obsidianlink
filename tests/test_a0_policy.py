from __future__ import annotations

import unittest
import time

import numpy as np

from obsidianlink.agents import (
    AsyncA0PolicyWorker,
    DirectA0Policy,
    WorkflowA0Policy,
    prompt_text,
)
from obsidianlink.core.types import Observation
from tests.helpers import sample_task


def _observation() -> Observation:
    return Observation(
        episode_id="episode",
        agent_id="agent_1",
        step_id=3,
        timestamp=1.0,
        frame=np.zeros((2, 2, 3), dtype=np.uint8),
        visible_inventory={"obsidian": 14},
        workflow_stage="build_frame",
    )


class A0PolicyTests(unittest.TestCase):
    def test_workflow_policy_uses_only_agent_visible_context(self) -> None:
        policy = WorkflowA0Policy(
            lambda prompt: '{"action_type":"look","parameters":{"yaw":10}}'
        )
        decision = policy.decide(sample_task(), _observation())
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.action.action_type, "look")
        self.assertEqual(decision.prompt["workflow"], "route_a_a0")
        self.assertNotIn("evaluation_state", decision.prompt)
        self.assertNotIn("evaluator", decision.prompt)

    def test_direct_policy_fails_closed_on_invalid_model_output(self) -> None:
        policy = DirectA0Policy(lambda prompt: '{"action_type":"shell"}')
        decision = policy.decide(sample_task(), _observation())
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.action.action_type, "wait")
        self.assertNotIn("workflow", decision.prompt)

    def test_async_worker_does_not_block_submission_or_polling(self) -> None:
        def slow_responder(prompt):
            time.sleep(0.05)
            return '{"action_type":"wait"}'

        worker = AsyncA0PolicyWorker(WorkflowA0Policy(slow_responder))
        worker.start()
        try:
            self.assertTrue(worker.submit(sample_task(), _observation()))
            self.assertIsNone(worker.poll(episode_id="episode", agent_id="agent_1"))
            deadline = time.monotonic() + 1.0
            result = None
            while result is None and time.monotonic() < deadline:
                result = worker.poll(episode_id="episode", agent_id="agent_1")
                time.sleep(0.01)
        finally:
            worker.close()
        self.assertIsNotNone(result)
        self.assertEqual(result.decision.action.action_type, "wait")

    def test_async_worker_surfaces_responder_failure(self) -> None:
        def broken_responder(prompt):
            del prompt
            raise RuntimeError("model unavailable")

        worker = AsyncA0PolicyWorker(DirectA0Policy(broken_responder))
        worker.start()
        try:
            self.assertTrue(worker.submit(sample_task(), _observation()))
            deadline = time.monotonic() + 1.0
            while worker.failure is None and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            worker.close()
        self.assertIsInstance(worker.failure, RuntimeError)

    def test_qwen_prompt_text_excludes_the_raw_frame(self) -> None:
        policy = WorkflowA0Policy(lambda prompt: '{"action_type":"wait"}')
        decision = policy.decide(sample_task(), _observation())
        text = prompt_text(decision.prompt)
        self.assertIn("Task: Build and enter a Nether portal.", text)
        self.assertIn("Workflow: route_a_a0", text)
        self.assertNotIn("array(", text)


if __name__ == "__main__":
    unittest.main()

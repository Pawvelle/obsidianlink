from __future__ import annotations

import unittest

from obsidianlink.core.interfaces import EnvironmentBackend
from obsidianlink.core.types import MacroAction
from obsidianlink.env.fake import FakeEnvironmentBackend
from tests.helpers import sample_task


class FakeEnvironmentBackendTests(unittest.TestCase):
    def test_multi_agent_shape_and_identity(self) -> None:
        backend = FakeEnvironmentBackend()
        self.assertIsInstance(backend, EnvironmentBackend)
        backend.open()
        try:
            task = sample_task(("agent_1", "agent_2"))
            observations = backend.reset(task)
            self.assertEqual(set(observations), {"agent_1", "agent_2"})
            step = backend.step(
                {
                    "agent_1": MacroAction.wait(),
                    "agent_2": MacroAction.wait(),
                }
            )
            self.assertEqual(step.step_id, 1)
            self.assertEqual(step.observations["agent_2"].agent_id, "agent_2")
        finally:
            backend.close()

    def test_missing_agent_action_is_rejected(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(sample_task(("agent_1", "agent_2")))
            with self.assertRaisesRegex(ValueError, "every task agent"):
                backend.step({"agent_1": MacroAction.wait()})
        finally:
            backend.close()

    def test_reset_requires_open(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not open"):
            FakeEnvironmentBackend().reset(sample_task())


if __name__ == "__main__":
    unittest.main()

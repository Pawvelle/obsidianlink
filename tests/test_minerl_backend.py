from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from obsidianlink.core.interfaces import EnvironmentBackend
from obsidianlink.core.types import MacroAction
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
from obsidianlink.env.portal_spec import (
    PORTAL_GRID_BLOCKS,
    PORTAL_GRID_SIZE,
    PortalA0EnvSpec,
)
from tests.helpers import sample_task


class _ControlledMineRLEnv:
    def __init__(self) -> None:
        self.action_space = PortalA0EnvSpec().action_space
        self.seed_value: int | None = None
        self.closed = False
        self.steps = 0
        self.grid = np.zeros(PORTAL_GRID_SIZE, dtype=np.int32)

    def seed(self, value: int) -> None:
        self.seed_value = value

    def reset(self) -> dict[str, Any]:
        return self._observation()

    def step(self, action: dict[str, Any]):
        self.assert_action(action)
        self.steps += 1
        self.grid[0] = PORTAL_GRID_BLOCKS.index("obsidian")
        self.grid[1] = PORTAL_GRID_BLOCKS.index("nether_portal")
        observation = self._observation()
        observation["portal_dimension"] = np.asarray("minecraft:the_nether")
        return observation, 0.0, False, {"private": "not_forwarded"}

    def assert_action(self, action: dict[str, Any]) -> None:
        if not self.action_space.contains(action):
            raise AssertionError("invalid test action")

    def close(self) -> None:
        self.closed = True

    def _observation(self) -> dict[str, Any]:
        return {
            "pov": np.zeros((360, 640, 3), dtype=np.uint8),
            "inventory": {
                "obsidian": np.asarray(10, dtype=np.int64),
                "flint_and_steel": np.asarray(1, dtype=np.int64),
                "dirt": np.asarray(0, dtype=np.int64),
            },
            "portal_grid": self.grid.copy(),
            "portal_dimension": np.asarray("minecraft:overworld"),
            "location_stats": {"xpos": 0.5, "ypos": 4.0, "zpos": 0.5},
            "use_item": {
                "obsidian": np.asarray(self.steps, dtype=np.int64),
                "flint_and_steel": np.asarray(0, dtype=np.int64),
            },
        }


class MineRLEnvironmentBackendTests(unittest.TestCase):
    def test_backend_implements_contract_and_hides_evaluator_truth(self) -> None:
        environment = _ControlledMineRLEnv()
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: environment,
            reset_warmup_steps=0,
        )
        self.assertIsInstance(backend, EnvironmentBackend)
        backend.open()
        try:
            observations = backend.reset(sample_task())
            self.assertEqual(environment.seed_value, 7)
            observation = observations["agent_1"]
            self.assertEqual(observation.visible_inventory["obsidian"], 10)
            self.assertFalse(hasattr(observation, "portal_grid"))

            step = backend.step(
                {"agent_1": MacroAction("place_block", target="obsidian")}
            )
            self.assertEqual(step.info["environment_info_keys"], ["private"])
            state = backend.get_evaluation_state()
            self.assertTrue(state.portal_built_by_episode)
            self.assertTrue(state.portal_activated)
            self.assertFalse(state.valid_portal_frame)
            self.assertEqual(state.agents_in_nether, frozenset({"agent_1"}))
            self.assertEqual(state.evidence["obsidian_added"], 1)
            self.assertEqual(state.evidence["max_obsidian_added"], 1)
            self.assertTrue(state.evidence["portal_activated_latched"])
            self.assertEqual(len(state.evidence["portal_grid_changes"]), 2)
            self.assertEqual(state.evidence["use_item_stats"]["obsidian"], 1)
        finally:
            backend.close()
        self.assertTrue(environment.closed)

    def test_backend_rejects_multi_agent_task_in_phase_one(self) -> None:
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: _ControlledMineRLEnv(),
            reset_warmup_steps=0,
        )
        backend.open()
        try:
            with self.assertRaisesRegex(ValueError, "exactly agent_1"):
                backend.reset(sample_task(("agent_1", "agent_2")))
        finally:
            backend.close()

    def test_minerl_random_fallback_observation_is_rejected(self) -> None:
        class _FailingMineRLEnv(_ControlledMineRLEnv):
            def step(self, action: dict[str, Any]):
                self.assert_action(action)
                return self._observation(), 0.0, True, {"error": "socket closed"}

        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: _FailingMineRLEnv(),
            reset_warmup_steps=0,
        )
        backend.open()
        try:
            backend.reset(sample_task())
            with self.assertRaisesRegex(RuntimeError, "socket closed"):
                backend.step({"agent_1": MacroAction.wait()})
            self.assertEqual(backend.get_evaluation_state().step_id, 0)
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()

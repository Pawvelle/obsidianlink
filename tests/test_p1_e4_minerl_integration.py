from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from obsidianlink.core.types import BackendStep, Observation
from obsidianlink.env.integration.e4_adapter import MineRLE4CameraAdapter, camera_orientation_snapshot
from obsidianlink.env.integration.e4_config import E4_AGENT_ID, E4_COMPATIBILITY_INVENTORY, build_e4_compatibility_task
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
from obsidianlink.env.validation import E4_CAMERA_CASE, EnvironmentValidationRunner
from tests.helpers import sample_task
from tests.test_minerl_backend import _ControlledMineRLEnv


ROOT = Path(__file__).resolve().parents[1]
EPISODE = "e4-adapter-episode"


class _Backend:
    instances = []
    def __init__(self, **kwargs: Any):
        self._opened = False; self._env = None; self._owner_thread = None
        self.step_id = 0; self.yaw = 3.0; self.calls = []
        type(self).instances.append(self)
    def open(self): self._opened = True; self.calls.append("open")
    def reset(self, task):
        self._env = object(); self.calls.append("reset")
        return {E4_AGENT_ID: SimpleNamespace(episode_id=EPISODE, agent_id=E4_AGENT_ID, step_id=0, frame="drop", visible_inventory={"dirt": 1}, selected_item="dirt", messages=("drop",), workflow_stage="drop")}
    def get_camera_orientation_truth(self):
        return {"episode_id": EPISODE, "agent_id": E4_AGENT_ID, "step_id": self.step_id, "yaw": self.yaw, "pitch": 0.0}
    def step(self, actions):
        self.calls.append("step"); self.step_id = 1; self.yaw = 23.0
        obs = Observation(EPISODE, E4_AGENT_ID, 1, 0.0, frame="not-used")
        return BackendStep(EPISODE, 1, {E4_AGENT_ID: obs}, {E4_AGENT_ID: 0.0}, False, False, {"translation_accepted": True, "location_stats": "must-not-leak"})
    def close(self): self.calls.append("close"); self._opened = False; self._env = None; self._owner_thread = None


class _OrientationEnv(_ControlledMineRLEnv):
    def __init__(self):
        super().__init__(); self.yaw = 7.0; self.pitch = -2.0
    def step(self, action):
        camera = np.asarray(action.get("camera", (0.0, 0.0)))
        self.pitch += float(camera[0]); self.yaw += float(camera[1])
        self.steps += 1
        return self._observation(), 0.0, False, {"location_stats": {"yaw": np.float32(self.yaw), "pitch": np.float32(self.pitch)}, "secret": "not-public"}


class E4MineRLIntegrationTests(unittest.TestCase):
    def test_config_is_compatibility_only_and_unrelated_inventory_is_minimal(self):
        task = build_e4_compatibility_task(EPISODE)
        self.assertEqual(dict(task.initial_inventories[E4_AGENT_ID]), E4_COMPATIBILITY_INVENTORY)
        self.assertEqual(task.scenario_parameters["p1_validation_id"], "E4")
        self.assertTrue(task.scenario_parameters["compatibility_only"])

    def test_adapter_executes_one_action_and_does_not_leak_truth(self):
        result = EnvironmentValidationRunner().run(E4_CAMERA_CASE, MineRLE4CameraAdapter.lifecycle_factory(episode_id=EPISODE, backend_cls=_Backend), episode_id=EPISODE)
        self.assertTrue(result.success)
        backend = _Backend.instances[-1]
        self.assertEqual(backend.calls, ["open", "reset", "step", "close"])
        self.assertNotIn("location_stats", result.as_dict())

    def test_snapshot_rejects_missing_extra_and_malformed_truth(self):
        base = {"episode_id": EPISODE, "agent_id": E4_AGENT_ID, "step_id": 0, "yaw": 0.0, "pitch": 0.0}
        for value in ({k: v for k, v in base.items() if k != "yaw"}, {**base, "inventory": {}}, {**base, "yaw": True}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                camera_orientation_snapshot(value)

    def test_real_backend_truth_comes_from_minerl_info_not_action_intent(self):
        env = _OrientationEnv()
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=1)
        backend.open()
        try:
            backend.reset(sample_task())
            before = backend.get_camera_orientation_truth()
            self.assertEqual(float(before["yaw"]), 7.0)  # type: ignore[index]
            from obsidianlink.core.types import MacroAction
            step = backend.step({"agent_1": MacroAction("look", parameters={"pitch": 0.0, "yaw": 20.0})})
            after = backend.get_camera_orientation_truth()
            self.assertEqual(float(after["yaw"]), 27.0)  # type: ignore[index]
            self.assertNotIn("yaw", step.info)
            self.assertNotIn("pitch", step.info)
            self.assertFalse(hasattr(step.observations[E4_AGENT_ID], "yaw"))
        finally:
            backend.close()

    def test_import_is_lazy(self):
        for name in ("e4_adapter.py", "e4_config.py", "e4_run.py"):
            tree = ast.parse((ROOT / "obsidianlink/env/integration" / name).read_text())
            imports = "\n".join(ast.unparse(node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom)))
            self.assertNotIn("obsidianlink.env.minerl_backend", imports)
            self.assertNotIn("import minerl", imports)
        with patch.object(MineRLE4CameraAdapter, "_resolve_backend_cls") as resolver:
            adapter = MineRLE4CameraAdapter(episode_id=EPISODE)
            resolver.assert_not_called(); self.assertIsNone(adapter._backend)


if __name__ == "__main__": unittest.main()

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from obsidianlink.actions.minerl_translator import translate_macro_action
from obsidianlink.core.types import BackendStep, MacroAction, Observation
from obsidianlink.env.integration.e5_adapter import MineRLE5MovementAdapter, movement_orientation_snapshot, player_position_snapshot
from obsidianlink.env.integration.e5_config import E5_AGENT_ID, build_e5_compatibility_task
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
from obsidianlink.env.validation import E5_MOVEMENT_CASE, EnvironmentValidationRunner
from tests.helpers import sample_task
from tests.test_minerl_backend import _ControlledMineRLEnv


ROOT = Path(__file__).resolve().parents[1]
EPISODE = "e5-adapter-episode"


class _Backend:
    instances = []
    def __init__(self, **kwargs: Any):
        self._opened = False; self._env = None; self._owner_thread = None
        self.step_id = 0; self.z = 0.0; self.calls = []; type(self).instances.append(self)
    def open(self): self._opened = True; self.calls.append("open")
    def reset(self, task):
        self._env = object(); self.calls.append("reset")
        return {E5_AGENT_ID: SimpleNamespace(episode_id=EPISODE, agent_id=E5_AGENT_ID, step_id=0, frame="drop", visible_inventory={"dirt": 1}, selected_item="dirt", messages=("drop",), workflow_stage="drop")}
    def get_player_position_truth(self):
        return {"episode_id": EPISODE, "agent_id": E5_AGENT_ID, "step_id": self.step_id, "x": 0.0, "y": 4.0, "z": self.z}
    def get_camera_orientation_truth(self):
        return {"episode_id": EPISODE, "agent_id": E5_AGENT_ID, "step_id": self.step_id, "yaw": 0.0, "pitch": 0.0}
    def step(self, actions):
        self.calls.append("step"); self.step_id = 1; self.z = 0.1
        obs = Observation(EPISODE, E5_AGENT_ID, 1, 0.0, frame="not-used")
        return BackendStep(EPISODE, 1, {E5_AGENT_ID: obs}, {E5_AGENT_ID: 0.0}, False, False, {"translation_accepted": True, "location_stats": "must-not-leak"})
    def close(self): self.calls.append("close"); self._opened = False; self._env = None; self._owner_thread = None


class _PositionEnv(_ControlledMineRLEnv):
    def __init__(self): super().__init__(); self.z = 2.0
    def step(self, action):
        self.z += 0.1 if action.get("forward") else 0.0; self.steps += 1
        return self._observation(), 0.0, False, {"location_stats": {"xpos": np.float32(1.0), "ypos": np.float32(4.0), "zpos": np.float32(self.z), "yaw": np.float32(0.0), "pitch": np.float32(0.0)}, "secret": "not-public"}


class E5MineRLIntegrationTests(unittest.TestCase):
    def test_config_is_compatibility_only_flat_and_minimal(self):
        task = build_e5_compatibility_task(EPISODE)
        self.assertEqual(task.spawn_positions[E5_AGENT_ID], (0, 4, 0))
        self.assertEqual(task.scenario_parameters["p1_validation_id"], "E5")
        self.assertTrue(task.scenario_parameters["compatibility_only"])

    def test_adapter_executes_one_action_and_does_not_leak_truth(self):
        result = EnvironmentValidationRunner().run(E5_MOVEMENT_CASE, MineRLE5MovementAdapter.lifecycle_factory(episode_id=EPISODE, backend_cls=_Backend), episode_id=EPISODE)
        self.assertTrue(result.success); self.assertEqual(_Backend.instances[-1].calls, ["open", "reset", "step", "close"])
        self.assertNotIn("location_stats", result.as_dict())

    def test_snapshot_rejects_missing_extra_and_malformed_truth(self):
        position = {"episode_id": EPISODE, "agent_id": E5_AGENT_ID, "step_id": 0, "x": 0.0, "y": 4.0, "z": 0.0}
        orientation = {"episode_id": EPISODE, "agent_id": E5_AGENT_ID, "step_id": 0, "yaw": 0.0, "pitch": 0.0}
        for value in ({k: v for k, v in position.items() if k != "x"}, {**position, "inventory": {}}, {**position, "x": True}):
            with self.subTest(value=value), self.assertRaises(ValueError): player_position_snapshot(value)
        for value in ({k: v for k, v in orientation.items() if k != "yaw"}, {**orientation, "xpos": 0}, {**orientation, "yaw": True}):
            with self.subTest(value=value), self.assertRaises(ValueError): movement_orientation_snapshot(value)

    def test_real_backend_truth_comes_from_minerl_info_not_move_or_spawn(self):
        env = _PositionEnv(); backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=1)
        backend.open()
        try:
            backend.reset(sample_task()); before = backend.get_player_position_truth()
            self.assertAlmostEqual(float(before["z"]), 2.0)  # type: ignore[index]
            step = backend.step({"agent_1": MacroAction("move", parameters={"forward": 1.0, "strafe": 0.0, "sprint": False, "jump": False})})
            after = backend.get_player_position_truth()
            self.assertAlmostEqual(float(after["z"]), 2.1, places=5)  # type: ignore[index]
            for key in ("x", "y", "z", "xpos", "ypos", "zpos"): self.assertNotIn(key, step.info)
            self.assertFalse(hasattr(step.observations[E5_AGENT_ID], "x"))
        finally: backend.close()

    def test_protocol_translator_path_and_action_space_rejection(self):
        env = _ControlledMineRLEnv(); action = MacroAction("move", parameters={"forward": 1.0, "strafe": 0.0, "sprint": False, "jump": False})
        translated = translate_macro_action(action, env.action_space)
        self.assertTrue(translated.accepted); self.assertEqual(translated.action["forward"], 1)
        self.assertEqual(translated.action["back"], 0); self.assertEqual(translated.action["left"], 0); self.assertEqual(translated.action["right"], 0)
        with patch.object(env.action_space, "contains", return_value=False):
            self.assertFalse(translate_macro_action(action, env.action_space).accepted)

    def test_import_is_lazy(self):
        for name in ("e5_adapter.py", "e5_config.py", "e5_run.py"):
            tree = ast.parse((ROOT / "obsidianlink/env/integration" / name).read_text())
            imports = "\n".join(ast.unparse(node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom)))
            self.assertNotIn("obsidianlink.env.minerl_backend", imports); self.assertNotIn("import minerl", imports)
        with patch.object(MineRLE5MovementAdapter, "_resolve_backend_cls") as resolver:
            adapter = MineRLE5MovementAdapter(episode_id=EPISODE); resolver.assert_not_called(); self.assertIsNone(adapter._backend)


if __name__ == "__main__": unittest.main()

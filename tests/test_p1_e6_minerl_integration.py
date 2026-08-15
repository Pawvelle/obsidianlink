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
from obsidianlink.env.integration.e6_adapter import (
    MineRLE6PlacementAdapter,
    block_placement_snapshot,
)
from obsidianlink.env.integration.e6_config import (
    E6_AGENT_ID,
    E6_CALIBRATION_BLOCK,
    E6_TARGET_CELL,
    build_e6_compatibility_task,
)
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
from obsidianlink.env.portal_spec import (
    PORTAL_GRID_BLOCKS,
    PORTAL_GRID_MAX,
    PORTAL_GRID_MIN,
    PORTAL_GRID_SIZE,
)
from obsidianlink.env.validation import E6_PLACEMENT_CASE, EnvironmentValidationRunner
from tests.helpers import sample_task
from tests.test_minerl_backend import _ControlledMineRLEnv


ROOT = Path(__file__).resolve().parents[1]
EPISODE = "e6-adapter-episode"


def _flat_index(cell: tuple[int, int, int]) -> int:
    x_size = PORTAL_GRID_MAX[0] - PORTAL_GRID_MIN[0] + 1
    z_size = PORTAL_GRID_MAX[2] - PORTAL_GRID_MIN[2] + 1
    y_size = PORTAL_GRID_MAX[1] - PORTAL_GRID_MIN[1] + 1
    x = cell[0] - PORTAL_GRID_MIN[0]
    y = cell[1] - PORTAL_GRID_MIN[1]
    z = cell[2] - PORTAL_GRID_MIN[2]
    return x + x_size * z + x_size * z_size * y


class _Backend:
    instances = []

    def __init__(self, **kwargs: Any):
        self._opened = False
        self._env = None
        self._owner_thread = None
        self.step_id = 0
        self.block = "air"
        self.calls = []
        type(self).instances.append(self)

    def open(self):
        self._opened = True
        self.calls.append("open")

    def reset(self, task):
        self._env = object()
        self.calls.append("reset")
        return {
            E6_AGENT_ID: SimpleNamespace(
                episode_id=EPISODE,
                agent_id=E6_AGENT_ID,
                step_id=0,
                frame="drop",
                visible_inventory={"dirt": 1},
                selected_item="dirt",
                messages=("drop",),
                workflow_stage="drop",
            )
        }

    def get_block_placement_truth(self, cell):
        return {
            "episode_id": EPISODE,
            "agent_id": E6_AGENT_ID,
            "step_id": self.step_id,
            "x": cell[0],
            "y": cell[1],
            "z": cell[2],
            "block": self.block,
        }

    def step(self, actions):
        self.calls.append("step")
        self.step_id = 1
        self.block = "dirt"
        obs = Observation(EPISODE, E6_AGENT_ID, 1, 0.0, frame="not-used")
        return BackendStep(
            EPISODE,
            1,
            {E6_AGENT_ID: obs},
            {E6_AGENT_ID: 0.0},
            False,
            False,
            {"translation_accepted": True, "portal_grid": "must-not-leak"},
        )

    def close(self):
        self.calls.append("close")
        self._opened = False
        self._env = None
        self._owner_thread = None


class _PlacementEnv(_ControlledMineRLEnv):
    def __init__(self, *, after_block: str | None = "dirt"):
        super().__init__()
        self.after_block = after_block
        self.grid = np.zeros(PORTAL_GRID_SIZE, dtype=np.int32)
        self.grid[_flat_index(E6_TARGET_CELL)] = PORTAL_GRID_BLOCKS.index("air")

    def step(self, action):
        self.assert_action(action)
        self.steps += 1
        action_map = action if isinstance(action, dict) else {}
        if int(action_map.get("use", 0)) and int(action_map.get("hotbar.1", 0)):
            if self.after_block is not None:
                self.grid[_flat_index(E6_TARGET_CELL)] = PORTAL_GRID_BLOCKS.index(
                    self.after_block
                )
        return self._observation(), 0.0, False, {"secret": "not-public"}


class E6MineRLIntegrationTests(unittest.TestCase):
    def test_config_is_compatibility_only_dirt_and_minimal(self):
        task = build_e6_compatibility_task(EPISODE)
        self.assertEqual(task.spawn_positions[E6_AGENT_ID], (0, 4, 0))
        self.assertEqual(task.initial_inventories[E6_AGENT_ID], {"dirt": 1})
        self.assertEqual(task.scenario_parameters["p1_validation_id"], "E6")
        self.assertEqual(task.scenario_parameters["calibration_block"], "dirt")
        self.assertTrue(task.scenario_parameters["compatibility_only"])

    def test_adapter_executes_one_action_and_does_not_leak_truth(self):
        result = EnvironmentValidationRunner().run(
            E6_PLACEMENT_CASE,
            MineRLE6PlacementAdapter.lifecycle_factory(
                episode_id=EPISODE, backend_cls=_Backend
            ),
            episode_id=EPISODE,
        )
        self.assertTrue(result.success)
        self.assertEqual(_Backend.instances[-1].calls, ["open", "reset", "step", "close"])
        self.assertNotIn("portal_grid", result.as_dict())
        payload = result.as_dict()
        self.assertEqual(payload["before_block"], "air")
        self.assertEqual(payload["after_block"], "dirt")

    def test_adapter_rejects_second_and_wrong_action(self):
        adapter = MineRLE6PlacementAdapter(episode_id=EPISODE, backend_cls=_Backend)
        adapter.reset()
        action = MacroAction("place_block", target="dirt")
        adapter.execute_placement_action(action)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            adapter.execute_placement_action(action)
        adapter.close()
        adapter = MineRLE6PlacementAdapter(episode_id=EPISODE, backend_cls=_Backend)
        adapter.reset()
        with self.assertRaises(ValueError):
            adapter.execute_placement_action(MacroAction("move"))
        with self.assertRaises(ValueError):
            adapter.execute_placement_action(MacroAction("place_block", target="obsidian"))
        with self.assertRaises(ValueError):
            adapter.execute_placement_action(
                MacroAction("place_block", target="dirt", parameters={"jump": True})
            )
        adapter.close()

    def test_snapshot_rejects_missing_extra_malformed_and_wrong_cell(self):
        base = {
            "episode_id": EPISODE,
            "agent_id": E6_AGENT_ID,
            "step_id": 0,
            "x": 0,
            "y": 4,
            "z": 1,
            "block": "air",
        }
        for value in (
            {k: v for k, v in base.items() if k != "block"},
            {**base, "inventory": {}},
            {**base, "block": True},
            {**base, "block": "missing"},
            {**base, "z": 2},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                block_placement_snapshot(value)

    def test_real_backend_truth_comes_from_grid_not_action_target(self):
        env = _PlacementEnv(after_block=None)
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e6_compatibility_task(EPISODE))
            before = backend.get_block_placement_truth(E6_TARGET_CELL)
            self.assertEqual(before["block"], "air")
            step = backend.step(
                {E6_AGENT_ID: MacroAction("place_block", target=E6_CALIBRATION_BLOCK)}
            )
            after = backend.get_block_placement_truth(E6_TARGET_CELL)
            self.assertEqual(after["block"], "air")
            self.assertNotIn("block", step.info)
            self.assertNotIn("portal_grid", step.info)
            self.assertFalse(hasattr(step.observations[E6_AGENT_ID], "block"))
            self.assertFalse(hasattr(step.observations[E6_AGENT_ID], "portal_grid"))
        finally:
            backend.close()

    def test_real_backend_observes_dirt_from_grid_after_place(self):
        env = _PlacementEnv(after_block="dirt")
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e6_compatibility_task(EPISODE))
            step = backend.step(
                {E6_AGENT_ID: MacroAction("place_block", target="dirt")}
            )
            after = backend.get_block_placement_truth(E6_TARGET_CELL)
            self.assertEqual(after["block"], "dirt")
            self.assertTrue(step.info["translation_accepted"])
            self.assertEqual(backend._hotbar_mapping["dirt"], "hotbar.1")
        finally:
            backend.close()

    def test_legacy_a0_hotbar_is_unchanged_for_non_e6(self):
        env = _ControlledMineRLEnv()
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(sample_task())
            self.assertEqual(backend._hotbar_mapping["dirt"], "hotbar.3")
        finally:
            backend.close()

    def test_missing_getter_fails_closed(self):
        class MissingGetter(_Backend):
            pass

        MissingGetter.get_block_placement_truth = None  # type: ignore[method-assign]
        result = EnvironmentValidationRunner().run(
            E6_PLACEMENT_CASE,
            MineRLE6PlacementAdapter.lifecycle_factory(
                episode_id=EPISODE, backend_cls=MissingGetter
            ),
            episode_id=EPISODE,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "runtime_error")

    def test_protocol_translator_path_and_action_space_rejection(self):
        env = _ControlledMineRLEnv()
        action = MacroAction("place_block", target="dirt")
        translated = translate_macro_action(action, env.action_space)
        self.assertTrue(translated.accepted)
        self.assertEqual(translated.action["use"], 1)
        with patch.object(env.action_space, "contains", return_value=False):
            self.assertFalse(translate_macro_action(action, env.action_space).accepted)

    def test_import_is_lazy(self):
        for name in ("e6_adapter.py", "e6_config.py", "e6_run.py"):
            tree = ast.parse((ROOT / "obsidianlink/env/integration" / name).read_text())
            imports = "\n".join(
                ast.unparse(node)
                for node in tree.body
                if isinstance(node, (ast.Import, ast.ImportFrom))
            )
            self.assertNotIn("obsidianlink.env.minerl_backend", imports)
            self.assertNotIn("import minerl", imports)
        with patch.object(MineRLE6PlacementAdapter, "_resolve_backend_cls") as resolver:
            adapter = MineRLE6PlacementAdapter(episode_id=EPISODE)
            resolver.assert_not_called()
            self.assertIsNone(adapter._backend)


if __name__ == "__main__":
    unittest.main()

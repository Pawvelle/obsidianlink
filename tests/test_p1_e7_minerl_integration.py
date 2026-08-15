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
from obsidianlink.env.integration.e6_config import (
    E6_INITIAL_PITCH,
    E6_INITIAL_YAW,
    E6_SPAWN_WORLD,
    E6_TARGET_GRID_CELL,
    E6_TARGET_WORLD_CELL,
)
from obsidianlink.env.integration.e7_adapter import (
    MineRLE7BucketAdapter,
    bucket_fluid_snapshot,
)
from obsidianlink.env.integration.e7_config import (
    E7_AGENT_ID,
    E7_INITIAL_PITCH,
    E7_INITIAL_YAW,
    E7_SPAWN_WORLD,
    E7_TARGET_GRID_CELL,
    E7_TARGET_WORLD_CELL,
    E7_WATER_CALIBRATION,
    build_e7_compatibility_task,
)
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
from obsidianlink.env.portal_spec import (
    PORTAL_GRID_BLOCKS,
    PORTAL_GRID_MAX,
    PORTAL_GRID_MIN,
    PORTAL_GRID_SIZE,
)
from obsidianlink.env.validation import E7_BUCKET_CASE, EnvironmentValidationRunner
from tests.helpers import sample_task
from tests.test_minerl_backend import _ControlledMineRLEnv


ROOT = Path(__file__).resolve().parents[1]
EPISODE = "e7-adapter-episode"
KNOWN_SPAWN_WORLD = (0, 4, 0)
KNOWN_TARGET_WORLD = (0, 4, 1)
KNOWN_TARGET_GRID = (0, 0, 1)
MISTAKEN_WORLD_AS_GRID = (0, 4, 1)


def _flat_index(cell: tuple[int, int, int]) -> int:
    x_size = PORTAL_GRID_MAX[0] - PORTAL_GRID_MIN[0] + 1
    z_size = PORTAL_GRID_MAX[2] - PORTAL_GRID_MIN[2] + 1
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
        self.fluid = "none"
        self.inventory = {"water_bucket": 1}
        self.calls = []
        type(self).instances.append(self)

    def open(self):
        self._opened = True
        self.calls.append("open")

    def reset(self, task):
        self._env = object()
        self.calls.append("reset")
        return {
            E7_AGENT_ID: SimpleNamespace(
                episode_id=EPISODE,
                agent_id=E7_AGENT_ID,
                step_id=0,
                frame="drop",
                visible_inventory=dict(self.inventory),
                selected_item="water_bucket",
                messages=("drop",),
                workflow_stage="drop",
            )
        }

    def get_bucket_fluid_truth(self, cell):
        return {
            "episode_id": EPISODE,
            "agent_id": E7_AGENT_ID,
            "step_id": self.step_id,
            "world_x": cell[0],
            "world_y": cell[1],
            "world_z": cell[2],
            "grid_x": 0,
            "grid_y": 0,
            "grid_z": 1,
            "fluid": self.fluid,
            "fluid_present": self.fluid != "none",
        }

    def get_reset_audit(self):
        return {"reset_attempt_count": 1, "environment_launch_count": 1}

    def step(self, actions):
        self.calls.append("step")
        self.step_id = 1
        self.fluid = "water"
        self.inventory = {"bucket": 1}
        obs = Observation(
            EPISODE,
            E7_AGENT_ID,
            1,
            0.0,
            frame="not-used",
            visible_inventory=dict(self.inventory),
            selected_item="bucket",
        )
        return BackendStep(
            EPISODE,
            1,
            {E7_AGENT_ID: obs},
            {E7_AGENT_ID: 0.0},
            False,
            False,
            {"translation_accepted": True, "portal_grid": "must-not-leak"},
        )

    def close(self):
        self.calls.append("close")
        self._opened = False
        self._env = None
        self._owner_thread = None


class _BucketEnv(_ControlledMineRLEnv):
    def __init__(
        self,
        *,
        after_fluid: str | None = "water",
        filled_item: str = "water_bucket",
        grid_origin: tuple[int, int, int] | None = KNOWN_SPAWN_WORLD,
    ):
        super().__init__()
        self.after_fluid = after_fluid
        self.filled_item = filled_item
        self._grid_origin = grid_origin
        self.inventory = {filled_item: 1}
        self.grid = np.zeros(PORTAL_GRID_SIZE, dtype=np.int32)
        self.grid[_flat_index(KNOWN_TARGET_GRID)] = PORTAL_GRID_BLOCKS.index("air")
        self.grid[_flat_index(MISTAKEN_WORLD_AS_GRID)] = PORTAL_GRID_BLOCKS.index("grass")

    def _observation(self):
        observation = super()._observation()
        observation["inventory"] = {
            item: np.asarray(quantity, dtype=np.int64)
            for item, quantity in self.inventory.items()
        }
        if self._grid_origin is None:
            observation.pop("portal_grid_origin", None)
        else:
            observation["portal_grid_origin"] = np.asarray(
                self._grid_origin, dtype=np.int32
            )
        return observation

    def step(self, action):
        self.assert_action(action)
        self.steps += 1
        action_map = action if isinstance(action, dict) else {}
        if int(action_map.get("use", 0)) and int(action_map.get("hotbar.1", 0)):
            if self.after_fluid is not None:
                self.grid[_flat_index(KNOWN_TARGET_GRID)] = PORTAL_GRID_BLOCKS.index(
                    self.after_fluid
                )
                self.inventory = {"bucket": 1}
        return self._observation(), 0.0, False, {"secret": "not-public"}


class E7MineRLIntegrationTests(unittest.TestCase):
    def test_config_is_compatibility_only_and_reuses_e6_geometry(self):
        task = build_e7_compatibility_task(EPISODE, "water")
        self.assertEqual(task.spawn_positions[E7_AGENT_ID], (0, 4, 0))
        self.assertEqual(task.initial_inventories[E7_AGENT_ID], {"water_bucket": 1})
        self.assertEqual(task.scenario_parameters["p1_validation_id"], "E7")
        self.assertEqual(task.scenario_parameters["bucket_item"], "water_bucket")
        self.assertTrue(task.scenario_parameters["compatibility_only"])
        self.assertEqual(E7_SPAWN_WORLD, E6_SPAWN_WORLD)
        self.assertEqual(E7_TARGET_WORLD_CELL, E6_TARGET_WORLD_CELL)
        self.assertEqual(E7_TARGET_GRID_CELL, E6_TARGET_GRID_CELL)
        self.assertEqual((E7_INITIAL_YAW, E7_INITIAL_PITCH), (E6_INITIAL_YAW, E6_INITIAL_PITCH))
        lava = build_e7_compatibility_task(EPISODE, "lava")
        self.assertEqual(lava.initial_inventories[E7_AGENT_ID], {"lava_bucket": 1})
        self.assertNotIn("water_bucket", lava.initial_inventories[E7_AGENT_ID])

    def test_adapter_executes_one_action_and_does_not_leak_truth(self):
        result = EnvironmentValidationRunner().run(
            E7_BUCKET_CASE,
            MineRLE7BucketAdapter.lifecycle_factory(
                episode_id=EPISODE, backend_cls=_Backend
            ),
            episode_id=EPISODE,
        )
        self.assertTrue(result.success)
        self.assertEqual(_Backend.instances[-1].calls, ["open", "reset", "step", "close"])
        payload = result.as_dict()
        self.assertNotIn("portal_grid", payload)
        self.assertEqual(payload["before_fluid"], "none")
        self.assertEqual(payload["after_fluid"], "water")
        self.assertEqual(payload["before_inventory"], {"water_bucket": 1})
        self.assertEqual(payload["after_inventory"], {"bucket": 1})
        self.assertEqual(payload["before_selected_item"], "water_bucket")
        self.assertEqual(payload["after_selected_item"], "bucket")

    def test_adapter_rejects_second_and_wrong_action(self):
        adapter = MineRLE7BucketAdapter(episode_id=EPISODE, backend_cls=_Backend)
        adapter.reset()
        action = MacroAction("use_item", target="water_bucket")
        adapter.execute_bucket_action(action)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            adapter.execute_bucket_action(action)
        adapter.close()
        adapter = MineRLE7BucketAdapter(episode_id=EPISODE, backend_cls=_Backend)
        adapter.reset()
        with self.assertRaises(ValueError):
            adapter.execute_bucket_action(MacroAction("move"))
        with self.assertRaises(ValueError):
            adapter.execute_bucket_action(MacroAction("use_item", target="lava_bucket"))
        with self.assertRaises(ValueError):
            adapter.execute_bucket_action(
                MacroAction("use_item", target="water_bucket", parameters={"jump": True})
            )
        adapter.close()

    def test_snapshot_rejects_missing_extra_malformed_and_wrong_cell(self):
        base = {
            "episode_id": EPISODE,
            "agent_id": E7_AGENT_ID,
            "step_id": 0,
            "world_x": 0,
            "world_y": 4,
            "world_z": 1,
            "grid_x": 0,
            "grid_y": 0,
            "grid_z": 1,
            "fluid": "none",
            "fluid_present": False,
        }
        for value in (
            {k: v for k, v in base.items() if k != "fluid"},
            {**base, "inventory": {}},
            {**base, "fluid": True},
            {**base, "fluid": "missing"},
            {**base, "world_z": 2},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                bucket_fluid_snapshot(value)

    def test_real_backend_truth_comes_from_grid_not_action_target(self):
        env = _BucketEnv(after_fluid=None)
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e7_compatibility_task(EPISODE, "water"))
            before = backend.get_bucket_fluid_truth(KNOWN_TARGET_WORLD)
            self.assertEqual(before["fluid"], "none")
            step = backend.step(
                {E7_AGENT_ID: MacroAction("use_item", target=E7_WATER_CALIBRATION.bucket_item)}
            )
            after = backend.get_bucket_fluid_truth(KNOWN_TARGET_WORLD)
            self.assertEqual(after["fluid"], "none")
            self.assertNotIn("fluid", step.info)
            self.assertNotIn("portal_grid", step.info)
            self.assertFalse(hasattr(step.observations[E7_AGENT_ID], "fluid"))
            self.assertFalse(hasattr(step.observations[E7_AGENT_ID], "portal_grid"))
            self.assertEqual(backend._hotbar_mapping["water_bucket"], "hotbar.1")
        finally:
            backend.close()

    def test_real_backend_observes_water_and_inventory_from_grid(self):
        env = _BucketEnv(after_fluid="water")
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e7_compatibility_task(EPISODE, "water"))
            step = backend.step(
                {E7_AGENT_ID: MacroAction("use_item", target="water_bucket")}
            )
            after = backend.get_bucket_fluid_truth(KNOWN_TARGET_WORLD)
            self.assertEqual(after["fluid"], "water")
            self.assertEqual(after["target_world_cell"] if False else after["world_x"], 0)
            self.assertEqual(
                (after["world_x"], after["world_y"], after["world_z"]),
                KNOWN_TARGET_WORLD,
            )
            self.assertEqual(
                (after["grid_x"], after["grid_y"], after["grid_z"]),
                KNOWN_TARGET_GRID,
            )
            self.assertTrue(step.info["translation_accepted"])
            self.assertEqual(step.observations[E7_AGENT_ID].visible_inventory, {"bucket": 1})
        finally:
            backend.close()

    def test_flowing_water_classifies_as_water(self):
        env = _BucketEnv(after_fluid="flowing_water")
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e7_compatibility_task(EPISODE, "water"))
            backend.step({E7_AGENT_ID: MacroAction("use_item", target="water_bucket")})
            after = backend.get_bucket_fluid_truth(KNOWN_TARGET_WORLD)
            self.assertEqual(after["fluid"], "water")
        finally:
            backend.close()

    def test_lava_variant_and_task_derived_hotbar(self):
        env = _BucketEnv(after_fluid="lava", filled_item="lava_bucket")
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e7_compatibility_task(EPISODE, "lava"))
            self.assertEqual(backend._hotbar_mapping["lava_bucket"], "hotbar.1")
            self.assertNotIn("water_bucket", backend._hotbar_mapping)
            backend.step({E7_AGENT_ID: MacroAction("use_item", target="lava_bucket")})
            after = backend.get_bucket_fluid_truth(KNOWN_TARGET_WORLD)
            self.assertEqual(after["fluid"], "lava")
        finally:
            backend.close()

    def test_mismatched_grid_origin_fails_closed(self):
        env = _BucketEnv(grid_origin=(0, 64, 0))
        backend = MineRLEnvironmentBackend(env_factory=lambda task: env, reset_warmup_steps=0)
        backend.open()
        try:
            backend.reset(build_e7_compatibility_task(EPISODE, "water"))
            with self.assertRaisesRegex(ValueError, "grid anchor differs from spawn"):
                backend.get_bucket_fluid_truth(KNOWN_TARGET_WORLD)
        finally:
            backend.close()

    def test_legacy_a0_hotbar_is_unchanged_for_non_e7(self):
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

        MissingGetter.get_bucket_fluid_truth = None  # type: ignore[method-assign]
        result = EnvironmentValidationRunner().run(
            E7_BUCKET_CASE,
            MineRLE7BucketAdapter.lifecycle_factory(
                episode_id=EPISODE, backend_cls=MissingGetter
            ),
            episode_id=EPISODE,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "runtime_error")

    def test_protocol_translator_path_and_action_space_rejection(self):
        env = _ControlledMineRLEnv()
        mapping = {"water_bucket": "hotbar.1"}
        action = MacroAction("use_item", target="water_bucket")
        translated = translate_macro_action(action, env.action_space, hotbar_mapping=mapping)
        self.assertTrue(translated.accepted)
        self.assertEqual(translated.action["use"], 1)
        self.assertEqual(translated.action["hotbar.1"], 1)
        with patch.object(env.action_space, "contains", return_value=False):
            self.assertFalse(
                translate_macro_action(action, env.action_space, hotbar_mapping=mapping).accepted
            )

    def test_import_is_lazy(self):
        for name in ("e7_adapter.py", "e7_config.py", "e7_run.py"):
            tree = ast.parse((ROOT / "obsidianlink/env/integration" / name).read_text())
            imports = "\n".join(
                ast.unparse(node)
                for node in tree.body
                if isinstance(node, (ast.Import, ast.ImportFrom))
            )
            self.assertNotIn("obsidianlink.env.minerl_backend", imports)
            self.assertNotIn("import minerl", imports)
        with patch.object(MineRLE7BucketAdapter, "_resolve_backend_cls") as resolver:
            adapter = MineRLE7BucketAdapter(episode_id=EPISODE)
            resolver.assert_not_called()
            self.assertIsNone(adapter._backend)


if __name__ == "__main__":
    unittest.main()

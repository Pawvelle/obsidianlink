from __future__ import annotations

import ast
import contextlib
import importlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from obsidianlink.core.types import BackendStep, Observation
from obsidianlink.env.integration.e11_config import (
    E11_CONTROL_WORLD_CELLS,
    E11_FRAME_BLOCKS,
    E11_IGNITION_TARGET_CELL,
    E11_INTERIOR_CELLS,
    E11_PROBE_GRID_CELLS,
    E11_PROBE_WORLD_CELLS,
)
from obsidianlink.env.integration.e11_geometry import (
    AUTHORIZED_LIVE_E11_GEOMETRY_RUN_VALUE,
    E11_GEOMETRY_READY,
    EXECUTION_MODE_AUTHORIZED_LIVE_E11_GEOMETRY,
    inspect_e11_geometry,
    main,
    preflight_authorized_e11_geometry,
    reset_authorized_e11_geometry_process_guards_for_tests,
    run_authorized_e11_geometry_smoke,
)
from obsidianlink.env.integration.e11_run import E11AuthorizationError
from obsidianlink.env.validation.truth import ServerBlockTruth, ServerTruthSnapshot


EPISODE = "e11-geometry-offline"
ROOT = Path(__file__).resolve().parents[1]


def _ready_blocks() -> dict[tuple[int, int, int], str]:
    blocks = {cell: "obsidian" for cell in E11_FRAME_BLOCKS}
    blocks.update({cell: "air" for cell in E11_INTERIOR_CELLS})
    blocks.update({cell: "air" for cell in E11_CONTROL_WORLD_CELLS})
    return blocks


def _snapshot(mapping: dict[tuple[int, int, int], str] | None = None) -> ServerTruthSnapshot:
    blocks = _ready_blocks() if mapping is None else mapping
    truths = tuple(
        ServerBlockTruth(world, grid, blocks[world])
        for world, grid in zip(E11_PROBE_WORLD_CELLS, E11_PROBE_GRID_CELLS)
    )
    return ServerTruthSnapshot(
        episode_id=EPISODE,
        agent_id="agent_1",
        step_id=0,
        position_world=(0.5, 4.0, 0.5),
        dimension="minecraft:overworld",
        grid_anchor_world=(0, 4, 0),
        anchor_source="portal_grid_origin",
        block_truth=truths,
        truth_missing_count=0,
    )


class _Backend:
    instances = []
    before_blocks = _ready_blocks()
    fail_reset = False

    def __init__(self, **kwargs: Any):
        self._opened = False
        self._env = None
        self._owner_thread = None
        self.step_id = 0
        self.calls = []
        type(self).instances.append(self)

    def open(self):
        self.calls.append("open")
        self._opened = True

    def reset(self, task):
        self.calls.append("reset")
        self._env = object()
        if type(self).fail_reset:
            raise RuntimeError("MineRL reset failed after 1 attempts")
        return {"agent_1": SimpleNamespace(episode_id=EPISODE, agent_id="agent_1", step_id=0)}

    def get_reset_audit(self):
        return {"reset_attempt_count": 1, "environment_launch_count": 1}

    def get_server_truth_snapshot(self, cells):
        mapping = type(self).before_blocks
        records = []
        for cell, grid in zip(cells, E11_PROBE_GRID_CELLS):
            records.append(
                {
                    "block": mapping[cell],
                    "grid_cell": list(grid),
                    "world_cell": list(cell),
                }
            )
        return {
            "agent_id": "agent_1",
            "anchor_source": "portal_grid_origin",
            "block_truth": records,
            "dimension": "minecraft:overworld",
            "episode_id": EPISODE,
            "grid_anchor_world": [0, 4, 0],
            "position_world": [0.5, 4.0, 0.5],
            "step_id": 0,
            "truth_missing_count": 0,
        }

    def step(self, actions):
        self.calls.append("step")
        obs = Observation(EPISODE, "agent_1", 1, 0.0, frame="unused")
        return BackendStep(
            EPISODE, 1, {"agent_1": obs}, {"agent_1": 0.0}, False, False, {"translation_accepted": True}
        )

    def close(self):
        self.calls.append("close")
        self._opened = False
        self._env = None
        self._owner_thread = None


class E11GeometrySmokeTests(unittest.TestCase):
    def setUp(self):
        reset_authorized_e11_geometry_process_guards_for_tests()
        _Backend.instances = []
        _Backend.fail_reset = False
        _Backend.before_blocks = _ready_blocks()

    def tearDown(self):
        reset_authorized_e11_geometry_process_guards_for_tests()

    def test_inspect_accepts_complete_frame_and_rejects_portal_or_fire(self) -> None:
        ready = inspect_e11_geometry(_snapshot())
        self.assertTrue(ready["ready"])
        self.assertEqual(ready["outcome"], E11_GEOMETRY_READY)
        self.assertTrue(ready["frame_valid_before"])
        self.assertEqual(ready["observed_frame_block_count"], 14)
        self.assertEqual(ready["interior_air_count"], 6)
        self.assertEqual(ready["before_portal_block_count"], 0)
        self.assertEqual(ready["fire_block_count"], 0)
        self.assertEqual(ready["ignition_block"], "air")
        self.assertTrue(ready["control_cells_expected"])
        missing = inspect_e11_geometry(None)
        self.assertEqual(missing["outcome"], "truth_snapshot_missing")
        incomplete = _ready_blocks()
        incomplete[(-1, 3, 1)] = "air"
        broken = inspect_e11_geometry(_snapshot(incomplete))
        self.assertFalse(broken["ready"])
        self.assertEqual(broken["observed_frame_block_count"], 13)
        portal = _ready_blocks()
        portal[E11_IGNITION_TARGET_CELL] = "nether_portal"
        portal_state = inspect_e11_geometry(_snapshot(portal))
        self.assertFalse(portal_state["ready"])
        self.assertEqual(portal_state["before_portal_block_count"], 1)
        fire = _ready_blocks()
        fire[E11_IGNITION_TARGET_CELL] = "fire"
        fire_state = inspect_e11_geometry(_snapshot(fire))
        self.assertFalse(fire_state["ready"])
        self.assertEqual(fire_state["fire_block_count"], 1)
        self.assertEqual(fire_state["ignition_block"], "fire")
        control = _ready_blocks()
        control[(0, 8, 1)] = "dirt"
        control_state = inspect_e11_geometry(_snapshot(control))
        self.assertFalse(control_state["control_cells_expected"])

    def test_check_and_preflight_remain_offline(self) -> None:
        module = importlib.import_module("obsidianlink.env.integration.e11_geometry")
        stdout = io.StringIO()
        with patch.object(module, "_production_backend_cls") as production, contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["--check"]), 0)
            production.assert_not_called()
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["success_outcome"], E11_GEOMETRY_READY)
        self.assertEqual(payload["planned_tested_stimulus_count"], 0)
        self.assertFalse(payload["integration_verified"])
        self.assertFalse(payload["production_backend_constructed"])
        self.assertTrue(payload["runtime_applies_obsidian_draw_blocks"])
        self.assertFalse(payload["needs_e11_runtime_geometry_authorization"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "p1_e11_portal_activation"
            root.mkdir()
            with patch("obsidianlink.env.integration.e11_run.FORMAL_E11_RUNS_ROOT", root.resolve()), patch(
                "obsidianlink.env.integration.e11_run.NEEDS_E11_RUNTIME_GEOMETRY_AUTHORIZATION", False
            ), patch(
                "obsidianlink.env.integration.e11_geometry._production_backend_cls"
            ) as production:
                preflight = preflight_authorized_e11_geometry(
                    execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E11_GEOMETRY,
                    authorized_live_run=AUTHORIZED_LIVE_E11_GEOMETRY_RUN_VALUE,
                    output_dir=root / "preflight",
                )
                production.assert_not_called()
        self.assertEqual(preflight["planned_tested_stimulus_count"], 0)

    def test_stub_success_does_not_step_or_claim_activation(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        try:
            root = Path(temporary.name) / "p1_e11_portal_activation"
            root.mkdir()
            output = root / "e11-geometry-001"
            with patch("obsidianlink.env.integration.e11_run.FORMAL_E11_RUNS_ROOT", root.resolve()), patch(
                "obsidianlink.env.integration.e11_run.NEEDS_E11_RUNTIME_GEOMETRY_AUTHORIZATION", False
            ), patch(
                "obsidianlink.env.integration.e11_geometry._production_backend_cls",
                return_value=_Backend,
            ), patch(
                "obsidianlink.env.integration.e11_geometry._runtime_identity",
                return_value={"jar_sha256": "a" * 64},
            ):
                record = run_authorized_e11_geometry_smoke(
                    execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E11_GEOMETRY,
                    authorized_live_run=AUTHORIZED_LIVE_E11_GEOMETRY_RUN_VALUE,
                    output_dir=output,
                    episode_id=EPISODE,
                )
            self.assertTrue(record.success)
            self.assertEqual(record.outcome, E11_GEOMETRY_READY)
            self.assertFalse(record.integration_verified)
            self.assertEqual(record.tested_action_count, 0)
            self.assertEqual(_Backend.instances[0].calls, ["open", "reset", "close"])
            payload = json.loads((output / "result.json").read_text())
            self.assertEqual(payload["outcome"], E11_GEOMETRY_READY)
            self.assertNotEqual(payload["outcome"], "portal_activation_ok")
            self.assertEqual(payload["tested_action_count"], 0)
            self.assertEqual(payload["observed_frame_block_count"], 14)
            self.assertEqual(payload["interior_air_count"], 6)
            self.assertEqual(payload["before_portal_block_count"], 0)
            self.assertEqual(payload["fire_block_count"], 0)
            xml = (output / "mission.xml").read_text()
            from obsidianlink.env.portal_spec import parse_mission_draw_blocks

            draw = parse_mission_draw_blocks(xml)
            self.assertEqual(len(draw), 14)
            self.assertTrue(all(block == "obsidian" for _, _, _, block in draw))
            self.assertFalse(any(block in {"portal", "nether_portal", "fire"} for _, _, _, block in draw))
        finally:
            temporary.cleanup()

    def test_activation_tokens_are_rejected(self) -> None:
        with patch("obsidianlink.env.integration.e11_geometry._production_backend_cls") as production:
            with self.assertRaises(E11AuthorizationError):
                run_authorized_e11_geometry_smoke(
                    execution_mode="authorized_live_e11",
                    authorized_live_run="e11_portal_activation",
                    output_dir=Path("/unused"),
                )
            production.assert_not_called()

    def test_module_does_not_import_activation_evaluator(self) -> None:
        tree = ast.parse(
            (ROOT / "obsidianlink/env/integration/e11_geometry.py").read_text(encoding="utf-8")
        )
        imports = "\n".join(
            ast.unparse(node)
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        )
        self.assertNotIn("EnvironmentValidationRunner", imports)
        self.assertNotIn("E11_PORTAL_ACTIVATION_CASE", imports)
        self.assertNotIn("obsidianlink.env.minerl_backend", imports)
        self.assertNotIn("execute_activation_stimulus", imports)


if __name__ == "__main__":
    unittest.main()

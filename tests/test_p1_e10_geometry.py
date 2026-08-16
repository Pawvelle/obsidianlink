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
from obsidianlink.env.integration.e10_geometry import (
    AUTHORIZED_LIVE_E10_GEOMETRY_RUN_VALUE,
    E10_GEOMETRY_READY,
    EXECUTION_MODE_AUTHORIZED_LIVE_E10_GEOMETRY,
    inspect_e10_geometry,
    main,
    preflight_authorized_e10_geometry,
    reset_authorized_e10_geometry_process_guards_for_tests,
    run_authorized_e10_geometry_smoke,
)
from obsidianlink.env.integration.e10_run import E10AuthorizationError
from obsidianlink.env.validation.truth import (
    ServerBlockTruth,
    ServerFluidTruth,
    ServerTruthSnapshot,
    classify_server_fluid,
)


EPISODE = "e10-geometry-offline"
PROBES = ((0, 4, 2), (0, 4, 1), (0, 5, 1), (0, 5, 2))
GRIDS = ((0, 0, 2), (0, 0, 1), (0, 1, 1), (0, 1, 2))
ROOT = Path(__file__).resolve().parents[1]


def _block(world, grid, block):
    return ServerBlockTruth(world, grid, block)


def _fluid(world, grid, block):
    present, fluid_type, flow_state = classify_server_fluid(block)
    return ServerFluidTruth(world, grid, block, present, fluid_type, flow_state)


def _snapshot(blocks=("lava", "air", "air", "air")):
    return ServerTruthSnapshot(
        episode_id=EPISODE,
        agent_id="agent_1",
        step_id=0,
        position_world=(0.5, 4.0, 0.5),
        dimension="minecraft:overworld",
        grid_anchor_world=(0, 4, 0),
        anchor_source="portal_grid_origin",
        block_truth=tuple(_block(PROBES[i], GRIDS[i], blocks[i]) for i in range(4)),
        truth_missing_count=0,
        fluid_truth=tuple(_fluid(PROBES[i], GRIDS[i], blocks[i]) for i in range(4)),
    )


def _fluid_record(block: str, world, grid) -> dict[str, object]:
    if block in {"water", "lava"}:
        fluid_type, flow_state, present = block, "source", True
    else:
        fluid_type, flow_state, present = "none", "none", False
    return {
        "flow_state": flow_state,
        "fluid_present": present,
        "fluid_type": fluid_type,
        "grid_cell": list(grid),
        "observed_block": block,
        "world_cell": list(world),
    }


class _Backend:
    instances = []
    before_blocks = ("lava", "air", "air", "air")
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
        records = []
        fluids = []
        for index, cell in enumerate(cells):
            block = type(self).before_blocks[index]
            records.append(
                {
                    "block": block,
                    "grid_cell": list(GRIDS[index]),
                    "world_cell": list(cell),
                }
            )
            fluids.append(_fluid_record(block, cell, GRIDS[index]))
        return {
            "agent_id": "agent_1",
            "anchor_source": "portal_grid_origin",
            "block_truth": records,
            "dimension": "minecraft:overworld",
            "episode_id": EPISODE,
            "fluid_truth": fluids,
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


class E10GeometrySmokeTests(unittest.TestCase):
    def setUp(self):
        reset_authorized_e10_geometry_process_guards_for_tests()
        _Backend.instances = []
        _Backend.fail_reset = False
        _Backend.before_blocks = ("lava", "air", "air", "air")

    def tearDown(self):
        reset_authorized_e10_geometry_process_guards_for_tests()

    def test_inspect_accepts_lava_source_and_rejects_air_or_obsidian(self) -> None:
        ready = inspect_e10_geometry(_snapshot())
        self.assertTrue(ready["ready"])
        self.assertEqual(ready["outcome"], E10_GEOMETRY_READY)
        self.assertEqual(ready["target_block"], "lava")
        self.assertEqual(ready["target_fluid_type"], "lava")
        self.assertEqual(ready["target_flow_state"], "source")
        self.assertEqual(ready["water_block"], "air")
        missing = inspect_e10_geometry(None)
        self.assertEqual(missing["outcome"], "truth_snapshot_missing")
        air = inspect_e10_geometry(_snapshot(("air", "air", "air", "air")))
        self.assertFalse(air["ready"])
        self.assertEqual(air["outcome"], "geometry_not_ready")
        obsidian = inspect_e10_geometry(_snapshot(("obsidian", "air", "air", "air")))
        self.assertFalse(obsidian["ready"])
        flowing = inspect_e10_geometry(_snapshot(("flowing_lava", "air", "air", "air")))
        self.assertFalse(flowing["ready"])
        self.assertEqual(flowing["target_flow_state"], "flowing")

    def test_check_and_preflight_remain_offline(self) -> None:
        module = importlib.import_module("obsidianlink.env.integration.e10_geometry")
        stdout = io.StringIO()
        with patch.object(module, "_production_backend_cls") as production, contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["--check"]), 0)
            production.assert_not_called()
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["success_outcome"], E10_GEOMETRY_READY)
        self.assertEqual(payload["planned_tested_stimulus_count"], 0)
        self.assertFalse(payload["integration_verified"])
        self.assertTrue(payload["runtime_applies_drawing_decorator"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "p1_e10_obsidian_conversion"
            root.mkdir()
            with patch("obsidianlink.env.integration.e10_run.FORMAL_E10_RUNS_ROOT", root.resolve()), patch(
                "obsidianlink.env.integration.e10_geometry._production_backend_cls"
            ) as production:
                preflight = preflight_authorized_e10_geometry(
                    execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E10_GEOMETRY,
                    authorized_live_run=AUTHORIZED_LIVE_E10_GEOMETRY_RUN_VALUE,
                    output_dir=root / "preflight",
                )
                production.assert_not_called()
        self.assertEqual(preflight["planned_tested_stimulus_count"], 0)

    def test_stub_success_does_not_step_or_claim_conversion(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        try:
            root = Path(temporary.name) / "p1_e10_obsidian_conversion"
            root.mkdir()
            output = root / "e10-geometry-001"
            with patch("obsidianlink.env.integration.e10_run.FORMAL_E10_RUNS_ROOT", root.resolve()), patch(
                "obsidianlink.env.integration.e10_geometry._production_backend_cls",
                return_value=_Backend,
            ), patch(
                "obsidianlink.env.integration.e10_geometry._runtime_identity",
                return_value={"jar_sha256": "a" * 64},
            ):
                record = run_authorized_e10_geometry_smoke(
                    execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E10_GEOMETRY,
                    authorized_live_run=AUTHORIZED_LIVE_E10_GEOMETRY_RUN_VALUE,
                    output_dir=output,
                    episode_id=EPISODE,
                )
            self.assertTrue(record.success)
            self.assertEqual(record.outcome, E10_GEOMETRY_READY)
            self.assertFalse(record.integration_verified)
            self.assertEqual(record.tested_action_count, 0)
            self.assertEqual(_Backend.instances[0].calls, ["open", "reset", "close"])
            payload = json.loads((output / "result.json").read_text())
            self.assertEqual(payload["outcome"], E10_GEOMETRY_READY)
            self.assertNotEqual(payload["outcome"], "obsidian_conversion_ok")
            self.assertEqual(payload["tested_action_count"], 0)
            self.assertFalse(payload["integration_verified"])
            xml = (output / "mission.xml").read_text()
            from obsidianlink.env.portal_spec import parse_mission_draw_blocks

            self.assertEqual(parse_mission_draw_blocks(xml), ((0, 4, 2, "lava"),))
            self.assertIn("<DrawingDecorator>", xml)
        finally:
            temporary.cleanup()

    def test_wrong_tokens_and_conversion_tokens_are_rejected(self) -> None:
        with patch("obsidianlink.env.integration.e10_geometry._production_backend_cls") as production:
            with self.assertRaises(E10AuthorizationError):
                run_authorized_e10_geometry_smoke(
                    execution_mode="authorized_live_e10",
                    authorized_live_run="e10_obsidian_conversion",
                    output_dir=Path("/unused"),
                )
            production.assert_not_called()

    def test_module_does_not_import_conversion_evaluator(self) -> None:
        tree = ast.parse(
            (ROOT / "obsidianlink/env/integration/e10_geometry.py").read_text(encoding="utf-8")
        )
        imports = "\n".join(
            ast.unparse(node)
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        )
        self.assertNotIn("EnvironmentValidationRunner", imports)
        self.assertNotIn("E10_OBSIDIAN_CONVERSION_CASE", imports)
        self.assertNotIn("obsidianlink.env.minerl_backend", imports)


if __name__ == "__main__":
    unittest.main()

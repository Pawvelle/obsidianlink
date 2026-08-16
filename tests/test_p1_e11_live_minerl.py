from __future__ import annotations

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
    E11_INTERIOR_CELLS,
    E11_PROBE_GRID_CELLS,
    E11_PROBE_WORLD_CELLS,
)
from obsidianlink.env.integration.e11_run import (
    AUTHORIZED_LIVE_E11_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E11,
    E11AuthorizationError,
    E11MineRLRunRecord,
    main,
    preflight_authorized_e11,
    reset_authorized_e11_process_guards_for_tests,
    run_authorized_e11_minerl,
)


EPISODE = "e11-live-offline"


def _world_blocks(interior: str = "air") -> dict[tuple[int, int, int], str]:
    blocks = {cell: "obsidian" for cell in E11_FRAME_BLOCKS}
    blocks.update({cell: interior for cell in E11_INTERIOR_CELLS})
    blocks.update({cell: "air" for cell in E11_CONTROL_WORLD_CELLS})
    return blocks


class _Backend:
    instances = []
    after_blocks = _world_blocks("nether_portal")
    before_blocks = _world_blocks()
    missing_after = False
    dirty = False
    fail_step = False
    fail_reset = False

    def __init__(self, **kwargs: Any):
        self._opened = False
        self._env = None
        self._owner_thread = None
        self.step_id = 0
        self.calls = []
        self.kwargs = dict(kwargs)
        type(self).instances.append(self)

    def open(self):
        self.calls.append("open")
        self._opened = True

    def reset(self, task):
        self.calls.append("reset")
        self._env = object()
        if type(self).fail_reset:
            try:
                raise TypeError("a bytes-like object is required, not 'NoneType'")
            except TypeError as exc:
                raise RuntimeError("MineRL reset failed after 1 attempts") from exc
        return {"agent_1": SimpleNamespace(episode_id=EPISODE, agent_id="agent_1", step_id=0)}

    def get_reset_audit(self):
        return {"reset_attempt_count": 1, "environment_launch_count": 1}

    def get_server_truth_snapshot(self, cells):
        if self.step_id and type(self).missing_after:
            return None
        mapping = type(self).before_blocks if not self.step_id else type(self).after_blocks
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
            "step_id": self.step_id,
            "truth_missing_count": 0,
        }

    def step(self, actions):
        self.calls.append("step")
        if type(self).fail_step:
            raise RuntimeError("step boom")
        self.step_id += 1
        obs = Observation(EPISODE, "agent_1", self.step_id, 0.0, frame="unused")
        return BackendStep(
            EPISODE, self.step_id, {"agent_1": obs}, {"agent_1": 0.0}, False, False, {"translation_accepted": True}
        )

    def close(self):
        self.calls.append("close")
        if not type(self).dirty:
            self._opened = False
            self._env = None
            self._owner_thread = None


def backend_cls(**settings):
    class Configured(_Backend):
        instances = []

    Configured.__name__ = "RecordingMineRLBackend"
    for key, value in settings.items():
        setattr(Configured, key, value)
    return Configured


class E11LiveGateTests(unittest.TestCase):
    def setUp(self):
        reset_authorized_e11_process_guards_for_tests()

    def tearDown(self):
        reset_authorized_e11_process_guards_for_tests()

    def run_stub(self, **kwargs):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name) / "p1_e11_portal_activation"
        root.mkdir()
        output = root / "e11-live-001"
        cls = backend_cls(**kwargs)
        with patch("obsidianlink.env.integration.e11_run.FORMAL_E11_RUNS_ROOT", root.resolve()), patch(
            "obsidianlink.env.integration.e11_run.NEEDS_E11_RUNTIME_GEOMETRY_AUTHORIZATION", False
        ), patch(
            "obsidianlink.env.integration.e11_run._production_backend_cls", return_value=cls
        ):
            record = run_authorized_e11_minerl(
                execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E11,
                authorized_live_run=AUTHORIZED_LIVE_E11_RUN_VALUE,
                output_dir=output,
                episode_id=EPISODE,
            )
        return record, output, cls, temporary

    def test_import_and_check_do_not_resolve_backend(self):
        module = importlib.import_module("obsidianlink.env.integration.e11_run")
        stdout = io.StringIO()
        with patch.object(module, "_production_backend_cls") as production, contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["--check"]), 0)
            production.assert_not_called()
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["production_backend_constructed"])
        self.assertFalse(payload["integration_verified"])
        self.assertFalse(payload["real_execution_performed"])
        self.assertEqual(payload["check_id"], "E11")
        self.assertEqual(payload["name"], "portal_activation")
        self.assertEqual(payload["probe_count"], len(E11_PROBE_WORLD_CELLS))
        self.assertTrue(payload["obsidian_frame_preplaced"])
        self.assertFalse(payload["portal_preplaced"])
        self.assertFalse(payload["fire_preplaced"])
        self.assertTrue(payload["needs_e11_runtime_geometry_authorization"])
        self.assertFalse(payload["runtime_applies_obsidian_draw_blocks"])
        self.assertEqual(payload["stimulus"]["target"], "flint_and_steel")
        self.assertEqual(payload["verification_level"], "unit_verified")

    def test_missing_wrong_mode_and_token_refuse_before_backend(self):
        attempts = (
            (None, AUTHORIZED_LIVE_E11_RUN_VALUE),
            ("offline", AUTHORIZED_LIVE_E11_RUN_VALUE),
            (EXECUTION_MODE_AUTHORIZED_LIVE_E11, None),
            (EXECUTION_MODE_AUTHORIZED_LIVE_E11, "E11"),
            (EXECUTION_MODE_AUTHORIZED_LIVE_E11, "e11_obsidian_conversion"),
        )
        with patch("obsidianlink.env.integration.e11_run._production_backend_cls") as production:
            for mode, token in attempts:
                with self.assertRaises(E11AuthorizationError):
                    run_authorized_e11_minerl(
                        execution_mode=mode, authorized_live_run=token, output_dir=Path("/unused")
                    )  # type: ignore[arg-type]
            production.assert_not_called()

    def test_live_without_runtime_authorization_does_not_construct_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "p1_e11_portal_activation"
            root.mkdir()
            with patch("obsidianlink.env.integration.e11_run.FORMAL_E11_RUNS_ROOT", root.resolve()), patch(
                "obsidianlink.env.integration.e11_run._production_backend_cls"
            ) as production:
                with self.assertRaisesRegex(E11AuthorizationError, "NEEDS_E11_RUNTIME_GEOMETRY_AUTHORIZATION"):
                    run_authorized_e11_minerl(
                        execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E11,
                        authorized_live_run=AUTHORIZED_LIVE_E11_RUN_VALUE,
                        output_dir=root / "e11-live-blocked",
                    )
                production.assert_not_called()

    def test_exact_preflight_remains_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "p1_e11_portal_activation"
            root.mkdir()
            with patch("obsidianlink.env.integration.e11_run.FORMAL_E11_RUNS_ROOT", root.resolve()), patch(
                "obsidianlink.env.integration.e11_run._production_backend_cls"
            ) as production:
                payload = preflight_authorized_e11(
                    execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E11,
                    authorized_live_run=AUTHORIZED_LIVE_E11_RUN_VALUE,
                    output_dir=root / "preflight",
                )
                production.assert_not_called()
        self.assertEqual(payload["probe_world_cells"], [list(cell) for cell in E11_PROBE_WORLD_CELLS])
        self.assertEqual(payload["probe_grid_cells"], [list(cell) for cell in E11_PROBE_GRID_CELLS])
        self.assertFalse(payload["integration_verified"])
        self.assertTrue(payload["needs_e11_runtime_geometry_authorization"])

    def test_stub_success_writes_narrow_evidence_and_one_step(self):
        record, output, cls, temporary = self.run_stub()
        try:
            self.assertIsInstance(record, E11MineRLRunRecord)
            self.assertTrue(record.success)
            self.assertFalse(record.real_execution_performed)
            payload = json.loads((output / "result.json").read_text())
            self.assertEqual(payload["outcome"], "portal_activation_ok")
            self.assertEqual(payload["tested_action_count"], 1)
            self.assertFalse(payload["integration_verified"])
            self.assertEqual(cls.instances[0].calls, ["open", "reset", "step", "close"])
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["stimulus_item"], "flint_and_steel")
            self.assertTrue(config["prebuilt_frame_is_calibration_fixture"])
            forbidden = {"portal_grid", "inventory", "rgb", "messages", "workflow_stage"}

            def _assert_no_forbidden(value: object) -> None:
                if isinstance(value, dict):
                    self.assertTrue(forbidden.isdisjoint(value))
                    for item in value.values():
                        _assert_no_forbidden(item)
                elif isinstance(value, list):
                    for item in value:
                        _assert_no_forbidden(item)

            _assert_no_forbidden(payload)
        finally:
            temporary.cleanup()

    def test_failures_cleanup_and_only_one_attempt(self):
        fire = _world_blocks("air")
        fire[(0, 4, 1)] = "fire"
        incomplete = _world_blocks("nether_portal")
        incomplete[(1, 6, 1)] = "air"
        for kwargs, outcome in (
            ({"after_blocks": _world_blocks("air")}, "ignition_effect_not_observed"),
            ({"after_blocks": fire}, "portal_activation_not_observed"),
            ({"after_blocks": incomplete}, "portal_pattern_incomplete"),
            ({"before_blocks": {**_world_blocks(), (0, 4, 1): "nether_portal"}}, "invalid_initial_state"),
            ({"missing_after": True}, "truth_snapshot_missing"),
            ({"fail_reset": True}, "reset_failed"),
            ({"fail_step": True}, "action_failed"),
            ({"dirty": True}, "cleanup_failed"),
        ):
            reset_authorized_e11_process_guards_for_tests()
            record, _, _, temporary = self.run_stub(**kwargs)
            try:
                self.assertFalse(record.success)
                self.assertEqual(record.outcome, outcome)
            finally:
                temporary.cleanup()


if __name__ == "__main__":
    unittest.main()

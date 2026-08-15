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
from obsidianlink.env.integration.e9_run import (
    AUTHORIZED_LIVE_E9_LAVA_RUN_VALUE,
    AUTHORIZED_LIVE_E9_WATER_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E9_LAVA,
    EXECUTION_MODE_AUTHORIZED_LIVE_E9_WATER,
    E9AuthorizationError,
    E9MineRLRunRecord,
    main,
    preflight_authorized_e9,
    reset_authorized_e9_process_guards_for_tests,
    run_authorized_e9_minerl,
)


EPISODE = "e9-live-offline"
PROBES = ((0, 4, 1), (0, 5, 1), (0, 5, 0))
GRIDS = ((0, 0, 1), (0, 1, 1), (0, 1, 0))


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
    after_blocks = ("water", "air", "air")
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
        blocks = ("air", "air", "air") if not self.step_id else type(self).after_blocks
        records = []
        fluids = []
        for index, cell in enumerate(cells):
            records.append(
                {
                    "block": blocks[index],
                    "grid_cell": list(GRIDS[index]),
                    "world_cell": list(cell),
                }
            )
            fluids.append(_fluid_record(blocks[index], cell, GRIDS[index]))
        return {
            "agent_id": "agent_1",
            "anchor_source": "portal_grid_origin",
            "block_truth": records,
            "dimension": "minecraft:overworld",
            "episode_id": EPISODE,
            "fluid_truth": fluids,
            "grid_anchor_world": [0, 4, 0],
            "position_world": [0.5, 4.0, 0.5],
            "step_id": self.step_id,
            "truth_missing_count": 0,
        }

    def step(self, actions):
        self.calls.append("step")
        if type(self).fail_step:
            raise RuntimeError("step boom")
        self.step_id = 1
        obs = Observation(EPISODE, "agent_1", 1, 0.0, frame="unused")
        return BackendStep(
            EPISODE, 1, {"agent_1": obs}, {"agent_1": 0.0}, False, False, {"translation_accepted": True}
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


class E9LiveGateTests(unittest.TestCase):
    def setUp(self):
        reset_authorized_e9_process_guards_for_tests()

    def tearDown(self):
        reset_authorized_e9_process_guards_for_tests()

    def run_stub(self, variant="water", **kwargs):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name) / "p1_e9_fluid_truth"
        root.mkdir()
        (root / variant).mkdir()
        output = root / variant / "run-1"
        cls = backend_cls(**kwargs)
        mode = (
            EXECUTION_MODE_AUTHORIZED_LIVE_E9_WATER
            if variant == "water"
            else EXECUTION_MODE_AUTHORIZED_LIVE_E9_LAVA
        )
        token = (
            AUTHORIZED_LIVE_E9_WATER_RUN_VALUE
            if variant == "water"
            else AUTHORIZED_LIVE_E9_LAVA_RUN_VALUE
        )
        with patch("obsidianlink.env.integration.e9_run.FORMAL_E9_RUNS_ROOT", root.resolve()), patch(
            "obsidianlink.env.integration.e9_run._production_backend_cls", return_value=cls
        ):
            record = run_authorized_e9_minerl(
                execution_mode=mode,
                authorized_live_run=token,
                output_dir=output,
                episode_id=EPISODE,
            )
        return record, output, cls, temporary

    def test_import_and_check_do_not_resolve_backend(self):
        module = importlib.import_module("obsidianlink.env.integration.e9_run")
        stdout = io.StringIO()
        with patch.object(module, "_production_backend_cls") as production, contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["--check"]), 0)
            production.assert_not_called()
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["production_backend_constructed"])
        self.assertFalse(payload["integration_verified"])
        self.assertFalse(payload["real_execution_performed"])
        self.assertEqual(payload["check_id"], "E9")
        self.assertEqual(payload["name"], "water_lava_fluid_truth")
        self.assertEqual(payload["probe_count"], 3)
        self.assertEqual(payload["truth_missing_required"], 0)
        self.assertEqual(payload["variants"]["water"]["stimulus_target"], "water_bucket")
        self.assertEqual(payload["variants"]["lava"]["expected_flow_state"], "source")

    def test_missing_wrong_mode_and_token_refuse_before_backend(self):
        attempts = (
            (None, AUTHORIZED_LIVE_E9_WATER_RUN_VALUE),
            ("offline", AUTHORIZED_LIVE_E9_WATER_RUN_VALUE),
            (EXECUTION_MODE_AUTHORIZED_LIVE_E9_WATER, None),
            (EXECUTION_MODE_AUTHORIZED_LIVE_E9_WATER, "E9"),
            (EXECUTION_MODE_AUTHORIZED_LIVE_E9_WATER, AUTHORIZED_LIVE_E9_LAVA_RUN_VALUE),
        )
        with patch("obsidianlink.env.integration.e9_run._production_backend_cls") as production:
            for mode, token in attempts:
                with self.assertRaises(E9AuthorizationError):
                    run_authorized_e9_minerl(
                        execution_mode=mode, authorized_live_run=token, output_dir=Path("/unused")
                    )  # type: ignore[arg-type]
            production.assert_not_called()

    def test_exact_preflight_remains_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "p1_e9_fluid_truth"
            root.mkdir()
            (root / "water").mkdir()
            with patch("obsidianlink.env.integration.e9_run.FORMAL_E9_RUNS_ROOT", root.resolve()), patch(
                "obsidianlink.env.integration.e9_run._production_backend_cls"
            ) as production:
                payload = preflight_authorized_e9(
                    execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E9_WATER,
                    authorized_live_run=AUTHORIZED_LIVE_E9_WATER_RUN_VALUE,
                    output_dir=root / "water" / "preflight",
                )
                production.assert_not_called()
        self.assertEqual(payload["probe_world_cells"], [[0, 4, 1], [0, 5, 1], [0, 5, 0]])
        self.assertEqual(payload["probe_grid_cells"], [[0, 0, 1], [0, 1, 1], [0, 1, 0]])

    def test_stub_success_writes_narrow_evidence_and_one_step(self):
        record, output, cls, temporary = self.run_stub()
        try:
            self.assertIsInstance(record, E9MineRLRunRecord)
            self.assertTrue(record.success)
            self.assertFalse(record.real_execution_performed)
            payload = json.loads((output / "e9_fluid_truth.json").read_text())
            self.assertEqual(payload["outcome"], "fluid_truth_ok")
            self.assertEqual(payload["tested_action_count"], 1)
            self.assertFalse(payload["integration_verified"])
            self.assertEqual(cls.instances[0].calls, ["open", "reset", "step", "close"])
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

    def test_lava_stub_success(self):
        record, output, _, temporary = self.run_stub(
            variant="lava", after_blocks=("lava", "air", "air")
        )
        try:
            self.assertTrue(record.success)
            payload = json.loads((output / "e9_fluid_truth.json").read_text())
            self.assertEqual(payload["fluid_variant"], "lava")
            self.assertEqual(payload["after_fluid_truth"][0]["fluid_type"], "lava")
        finally:
            temporary.cleanup()

    def test_failures_cleanup_and_only_one_attempt(self):
        for kwargs, outcome in (
            ({"after_blocks": ("air", "air", "air")}, "truth_after_fluid_mismatch"),
            ({"missing_after": True}, "truth_snapshot_missing"),
            ({"fail_reset": True}, "reset_failed"),
            ({"fail_step": True}, "action_failed"),
            ({"dirty": True}, "cleanup_failed"),
        ):
            reset_authorized_e9_process_guards_for_tests()
            record, _, _, temporary = self.run_stub(**kwargs)
            try:
                self.assertFalse(record.success)
                self.assertEqual(record.outcome, outcome)
            finally:
                temporary.cleanup()
        reset_authorized_e9_process_guards_for_tests()
        record, _, _, temporary = self.run_stub()
        try:
            root = Path(temporary.name) / "p1_e9_fluid_truth"
            with patch("obsidianlink.env.integration.e9_run.FORMAL_E9_RUNS_ROOT", root.resolve()), self.assertRaisesRegex(
                E9AuthorizationError, "one real run"
            ):
                run_authorized_e9_minerl(
                    execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E9_WATER,
                    authorized_live_run=AUTHORIZED_LIVE_E9_WATER_RUN_VALUE,
                    output_dir=root / "water" / "run-2",
                    episode_id=EPISODE,
                )
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()

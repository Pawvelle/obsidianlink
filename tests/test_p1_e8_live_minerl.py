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
from obsidianlink.env.integration.e8_run import (
    AUTHORIZED_LIVE_E8_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E8,
    E8AuthorizationError,
    E8MineRLRunRecord,
    main,
    preflight_authorized_e8,
    reset_authorized_e8_process_guards_for_tests,
    run_authorized_e8_minerl,
)
from obsidianlink.env.validation.truth import ServerBlockTruth, ServerTruthSnapshot


EPISODE = "e8-live-offline"
PROBES = ((0, 4, 1), (1, 4, 1), (-1, 4, 1))
GRIDS = ((0, 0, 1), (1, 0, 1), (-1, 0, 1))


def _snapshot(step, blocks):
    return ServerTruthSnapshot(
        EPISODE,
        "agent_1",
        step,
        (0.5, 4.0, 0.5),
        "minecraft:overworld",
        (0, 4, 0),
        "portal_grid_origin",
        tuple(ServerBlockTruth(PROBES[i], GRIDS[i], blocks[i]) for i in range(3)),
        0,
    )


class _Backend:
    instances = []
    after_blocks = ("dirt", "air", "air")
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
        for index, cell in enumerate(cells):
            records.append(
                {
                    "block": blocks[index],
                    "grid_cell": list(GRIDS[index]),
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


class E8LiveGateTests(unittest.TestCase):
    def setUp(self):
        reset_authorized_e8_process_guards_for_tests()

    def tearDown(self):
        reset_authorized_e8_process_guards_for_tests()

    def run_stub(self, **kwargs):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name) / "p1_e8_block_truth"
        root.mkdir()
        output = root / "run-1"
        cls = backend_cls(**kwargs)
        with patch("obsidianlink.env.integration.e8_run.FORMAL_E8_RUNS_ROOT", root.resolve()), patch(
            "obsidianlink.env.integration.e8_run._production_backend_cls", return_value=cls
        ):
            record = run_authorized_e8_minerl(
                execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E8,
                authorized_live_run=AUTHORIZED_LIVE_E8_RUN_VALUE,
                output_dir=output,
                episode_id=EPISODE,
            )
        return record, output, cls, temporary

    def test_import_and_check_do_not_resolve_backend(self):
        module = importlib.import_module("obsidianlink.env.integration.e8_run")
        stdout = io.StringIO()
        with patch.object(module, "_production_backend_cls") as production, contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["--check"]), 0)
            production.assert_not_called()
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["production_backend_constructed"])
        self.assertFalse(payload["integration_verified"])
        self.assertFalse(payload["real_execution_performed"])
        self.assertEqual(payload["check_id"], "E8")
        self.assertEqual(payload["name"], "server_side_block_truth")
        self.assertEqual(payload["probe_count"], 3)
        self.assertEqual(payload["truth_missing_required"], 0)
        self.assertEqual(payload["stimulus"]["target"], "dirt")

    def test_missing_wrong_mode_and_token_refuse_before_backend(self):
        attempts = (
            (None, AUTHORIZED_LIVE_E8_RUN_VALUE),
            ("offline", AUTHORIZED_LIVE_E8_RUN_VALUE),
            (EXECUTION_MODE_AUTHORIZED_LIVE_E8, None),
            (EXECUTION_MODE_AUTHORIZED_LIVE_E8, "E8"),
        )
        with patch("obsidianlink.env.integration.e8_run._production_backend_cls") as production:
            for mode, token in attempts:
                with self.assertRaises(E8AuthorizationError):
                    run_authorized_e8_minerl(
                        execution_mode=mode, authorized_live_run=token, output_dir=Path("/unused")
                    )  # type: ignore[arg-type]
            production.assert_not_called()

    def test_exact_preflight_remains_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "p1_e8_block_truth"
            root.mkdir()
            with patch("obsidianlink.env.integration.e8_run.FORMAL_E8_RUNS_ROOT", root.resolve()), patch(
                "obsidianlink.env.integration.e8_run._production_backend_cls"
            ) as production:
                payload = preflight_authorized_e8(
                    execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E8,
                    authorized_live_run=AUTHORIZED_LIVE_E8_RUN_VALUE,
                    output_dir=root / "preflight",
                )
                production.assert_not_called()
        self.assertEqual(payload["probe_world_cells"], [[0, 4, 1], [1, 4, 1], [-1, 4, 1]])
        self.assertEqual(payload["probe_grid_cells"], [[0, 0, 1], [1, 0, 1], [-1, 0, 1]])

    def test_stub_success_writes_narrow_evidence_and_one_step(self):
        record, output, cls, temporary = self.run_stub()
        try:
            self.assertIsInstance(record, E8MineRLRunRecord)
            self.assertTrue(record.success)
            self.assertFalse(record.real_execution_performed)
            payload = json.loads((output / "e8_block_truth.json").read_text())
            self.assertEqual(payload["outcome"], "block_truth_ok")
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

    def test_failures_cleanup_and_only_one_attempt(self):
        for kwargs, outcome in (
            ({"after_blocks": ("air", "air", "air")}, "truth_after_mismatch"),
            ({"missing_after": True}, "truth_snapshot_missing"),
            ({"fail_step": True}, "action_failed"),
            ({"dirty": True}, "cleanup_failed"),
        ):
            reset_authorized_e8_process_guards_for_tests()
            record, _, _, temporary = self.run_stub(**kwargs)
            try:
                self.assertFalse(record.success)
                self.assertEqual(record.outcome, outcome)
            finally:
                temporary.cleanup()
        reset_authorized_e8_process_guards_for_tests()
        record, _, _, temporary = self.run_stub()
        try:
            root = Path(temporary.name) / "p1_e8_block_truth"
            with patch("obsidianlink.env.integration.e8_run.FORMAL_E8_RUNS_ROOT", root.resolve()), self.assertRaisesRegex(
                E8AuthorizationError, "one real run"
            ):
                run_authorized_e8_minerl(
                    execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E8,
                    authorized_live_run=AUTHORIZED_LIVE_E8_RUN_VALUE,
                    output_dir=root / "run-2",
                    episode_id=EPISODE,
                )
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()

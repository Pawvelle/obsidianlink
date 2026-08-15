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
from obsidianlink.env.integration.e6_run import (
    AUTHORIZED_LIVE_E6_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E6,
    E6AuthorizationError,
    E6MineRLRunRecord,
    main,
    preflight_authorized_e6,
    reset_authorized_e6_process_guards_for_tests,
    run_authorized_e6_minerl,
)


EPISODE = "e6-live-offline"


class _Backend:
    instances = []
    after_block = "dirt"
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

    def get_block_placement_truth(self, cell):
        if self.step_id and type(self).missing_after:
            return None
        return {
            "episode_id": EPISODE,
            "agent_id": "agent_1",
            "step_id": self.step_id,
            "x": cell[0],
            "y": cell[1],
            "z": cell[2],
            "block": "air" if not self.step_id else type(self).after_block,
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


class E6LiveGateTests(unittest.TestCase):
    def setUp(self):
        reset_authorized_e6_process_guards_for_tests()

    def tearDown(self):
        reset_authorized_e6_process_guards_for_tests()

    def run_stub(self, **kwargs):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name) / "p1_e6_block_placement"
        root.mkdir()
        output = root / "run-1"
        cls = backend_cls(**kwargs)
        with patch("obsidianlink.env.integration.e6_run.FORMAL_E6_RUNS_ROOT", root.resolve()), patch(
            "obsidianlink.env.integration.e6_run._production_backend_cls", return_value=cls
        ):
            record = run_authorized_e6_minerl(
                execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E6,
                authorized_live_run=AUTHORIZED_LIVE_E6_RUN_VALUE,
                output_dir=output,
                episode_id=EPISODE,
            )
        return record, output, cls, temporary

    def test_import_and_check_do_not_resolve_backend(self):
        module = importlib.import_module("obsidianlink.env.integration.e6_run")
        stdout = io.StringIO()
        with patch.object(module, "_production_backend_cls") as production, contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["--check"]), 0)
            production.assert_not_called()
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["production_backend_constructed"])
        self.assertFalse(payload["integration_verified"])
        self.assertEqual(payload["calibration_block"], "dirt")

    def test_missing_wrong_mode_and_token_refuse_before_backend(self):
        attempts = (
            (None, AUTHORIZED_LIVE_E6_RUN_VALUE),
            ("offline", AUTHORIZED_LIVE_E6_RUN_VALUE),
            (EXECUTION_MODE_AUTHORIZED_LIVE_E6, None),
            (EXECUTION_MODE_AUTHORIZED_LIVE_E6, "E6"),
        )
        with patch("obsidianlink.env.integration.e6_run._production_backend_cls") as production:
            for mode, token in attempts:
                with self.assertRaises(E6AuthorizationError):
                    run_authorized_e6_minerl(
                        execution_mode=mode, authorized_live_run=token, output_dir=Path("/unused")
                    )  # type: ignore[arg-type]
            production.assert_not_called()

    def test_exact_preflight_remains_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "p1_e6_block_placement"
            root.mkdir()
            with patch("obsidianlink.env.integration.e6_run.FORMAL_E6_RUNS_ROOT", root.resolve()), patch(
                "obsidianlink.env.integration.e6_run._production_backend_cls"
            ) as production:
                payload = preflight_authorized_e6(
                    execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E6,
                    authorized_live_run=AUTHORIZED_LIVE_E6_RUN_VALUE,
                    output_dir=root / "preflight",
                )
                production.assert_not_called()
        self.assertEqual(payload["requested_target"], "dirt")
        self.assertEqual(payload["target_cell"], [0, 4, 1])

    def test_stub_success_writes_narrow_evidence_and_one_step(self):
        record, output, cls, temporary = self.run_stub()
        try:
            self.assertIsInstance(record, E6MineRLRunRecord)
            self.assertTrue(record.success)
            self.assertFalse(record.real_execution_performed)
            payload = json.loads((output / "e6_block_placement.json").read_text())
            self.assertEqual(payload["after_block"], "dirt")
            self.assertEqual(payload["tested_action_count"], 1)
            self.assertFalse(payload["integration_verified"])
            authorization = json.loads((output / "authorization.json").read_text())
            self.assertEqual(authorization["planned_tested_placement_action_count"], 1)
            serialized = json.dumps(payload)
            for forbidden in ("portal_grid", "inventory", "rgb", "messages", "workflow_stage"):
                self.assertNotIn(forbidden, serialized)
            self.assertEqual(cls.instances[0].calls, ["open", "reset", "step", "close"])
            self.assertEqual(cls.instances[0].kwargs["max_reset_attempts"], 1)
            self.assertIn("env_factory", cls.instances[0].kwargs)
        finally:
            temporary.cleanup()

    def test_reset_failure_separates_plan_from_actual_and_persists_traceback(self):
        record, output, cls, temporary = self.run_stub(fail_reset=True)
        try:
            self.assertFalse(record.success)
            self.assertEqual(record.outcome, "reset_failed")
            payload = json.loads((output / "e6_block_placement.json").read_text())
            authorization = json.loads((output / "authorization.json").read_text())
            self.assertEqual(authorization["planned_tested_placement_action_count"], 1)
            self.assertEqual(payload["tested_action_count"], 0)
            self.assertEqual(payload["failure_stage"], "reset")
            self.assertEqual(payload["original_exception_type"], "TypeError")
            self.assertIn("TypeError: a bytes-like object", payload["exception_traceback"])
            self.assertEqual(cls.instances[0].calls, ["open", "reset", "close"])
            self.assertIsNone(payload["world_changed"])
        finally:
            temporary.cleanup()

    def test_failures_cleanup_and_only_one_attempt(self):
        for kwargs, outcome in (
            ({"after_block": "air"}, "placement_no_world_effect"),
            ({"missing_after": True}, "block_after_missing"),
            ({"fail_step": True}, "action_failed"),
            ({"dirty": True}, "cleanup_failed"),
        ):
            reset_authorized_e6_process_guards_for_tests()
            record, _, _, temporary = self.run_stub(**kwargs)
            try:
                self.assertFalse(record.success)
                self.assertEqual(record.outcome, outcome)
            finally:
                temporary.cleanup()
        reset_authorized_e6_process_guards_for_tests()
        record, _, _, temporary = self.run_stub()
        try:
            root = Path(temporary.name) / "p1_e6_block_placement"
            with patch("obsidianlink.env.integration.e6_run.FORMAL_E6_RUNS_ROOT", root.resolve()), self.assertRaisesRegex(
                E6AuthorizationError, "one real run"
            ):
                run_authorized_e6_minerl(
                    execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E6,
                    authorized_live_run=AUTHORIZED_LIVE_E6_RUN_VALUE,
                    output_dir=root / "run-2",
                    episode_id=EPISODE,
                )
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()

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
from obsidianlink.env.integration.e5_run import (
    AUTHORIZED_LIVE_E5_RUN_VALUE, EXECUTION_MODE_AUTHORIZED_LIVE_E5,
    E5AuthorizationError, E5MineRLRunRecord, main, preflight_authorized_e5,
    reset_authorized_e5_process_guards_for_tests, run_authorized_e5_minerl,
    _collect_runtime_log_evidence, _snapshot_runtime_logs,
)


EPISODE = "e5-live-offline"


class _Backend:
    instances = []; after_z = 0.1; missing_after = False; dirty = False; fail_step = False; fail_reset = False
    def __init__(self, **kwargs: Any):
        self._opened = False; self._env = None; self._owner_thread = None; self.step_id = 0; self.calls = []; self.kwargs = dict(kwargs); type(self).instances.append(self)
    def open(self): self.calls.append("open"); self._opened = True
    def reset(self, task):
        self.calls.append("reset"); self._env = object()
        if type(self).fail_reset:
            try: raise TypeError("a bytes-like object is required, not 'NoneType'")
            except TypeError as exc: raise RuntimeError("MineRL reset failed after 1 attempts") from exc
        return {"agent_1": SimpleNamespace(episode_id=EPISODE, agent_id="agent_1", step_id=0)}
    def get_reset_audit(self): return {"reset_attempt_count": 1, "environment_launch_count": 1}
    def get_player_position_truth(self):
        if self.step_id and type(self).missing_after: return None
        return {"episode_id": EPISODE, "agent_id": "agent_1", "step_id": self.step_id, "x": 0.0, "y": 4.0, "z": 0.0 if not self.step_id else type(self).after_z}
    def get_camera_orientation_truth(self): return {"episode_id": EPISODE, "agent_id": "agent_1", "step_id": self.step_id, "yaw": 0.0, "pitch": 0.0}
    def step(self, actions):
        self.calls.append("step")
        if type(self).fail_step: raise RuntimeError("step boom")
        self.step_id = 1; obs = Observation(EPISODE, "agent_1", 1, 0.0, frame="unused")
        return BackendStep(EPISODE, 1, {"agent_1": obs}, {"agent_1": 0.0}, False, False, {"translation_accepted": True})
    def close(self):
        self.calls.append("close")
        if not type(self).dirty: self._opened = False; self._env = None; self._owner_thread = None


def backend_cls(**settings):
    class Configured(_Backend): instances = []
    Configured.__name__ = "RecordingMineRLBackend"
    for key, value in settings.items(): setattr(Configured, key, value)
    return Configured


class E5LiveGateTests(unittest.TestCase):
    def setUp(self): reset_authorized_e5_process_guards_for_tests()
    def tearDown(self): reset_authorized_e5_process_guards_for_tests()
    def run_stub(self, **kwargs):
        temporary = tempfile.TemporaryDirectory(); root = Path(temporary.name) / "p1_e5_movement"; root.mkdir(); output = root / "run-1"; cls = backend_cls(**kwargs)
        with patch("obsidianlink.env.integration.e5_run.FORMAL_E5_RUNS_ROOT", root.resolve()), patch("obsidianlink.env.integration.e5_run._production_backend_cls", return_value=cls):
            record = run_authorized_e5_minerl(execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E5, authorized_live_run=AUTHORIZED_LIVE_E5_RUN_VALUE, output_dir=output, episode_id=EPISODE)
        return record, output, cls, temporary

    def test_import_and_check_do_not_resolve_backend(self):
        module = importlib.import_module("obsidianlink.env.integration.e5_run"); stdout = io.StringIO()
        with patch.object(module, "_production_backend_cls") as production, contextlib.redirect_stdout(stdout): self.assertEqual(main(["--check"]), 0); production.assert_not_called()
        self.assertFalse(json.loads(stdout.getvalue())["production_backend_constructed"])

    def test_missing_wrong_mode_and_token_refuse_before_backend(self):
        attempts = ((None, AUTHORIZED_LIVE_E5_RUN_VALUE), ("offline", AUTHORIZED_LIVE_E5_RUN_VALUE), (EXECUTION_MODE_AUTHORIZED_LIVE_E5, None), (EXECUTION_MODE_AUTHORIZED_LIVE_E5, "E5"))
        with patch("obsidianlink.env.integration.e5_run._production_backend_cls") as production:
            for mode, token in attempts:
                with self.assertRaises(E5AuthorizationError): run_authorized_e5_minerl(execution_mode=mode, authorized_live_run=token, output_dir=Path("/unused"))  # type: ignore[arg-type]
            production.assert_not_called()

    def test_exact_preflight_remains_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "p1_e5_movement"; root.mkdir()
            with patch("obsidianlink.env.integration.e5_run.FORMAL_E5_RUNS_ROOT", root.resolve()), patch("obsidianlink.env.integration.e5_run._production_backend_cls") as production:
                payload = preflight_authorized_e5(execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E5, authorized_live_run=AUTHORIZED_LIVE_E5_RUN_VALUE, output_dir=root / "preflight"); production.assert_not_called()
        self.assertEqual(payload["requested_forward"], 1.0)

    def test_stub_success_writes_narrow_evidence_and_one_step(self):
        record, output, cls, temporary = self.run_stub()
        try:
            self.assertIsInstance(record, E5MineRLRunRecord); self.assertTrue(record.success)
            payload = json.loads((output / "e5_movement.json").read_text()); self.assertAlmostEqual(payload["forward_projection"], 0.1); self.assertEqual(payload["tested_action_count"], 1)
            authorization = json.loads((output / "authorization.json").read_text())
            self.assertEqual(authorization["planned_tested_movement_action_count"], 1)
            self.assertNotIn("tested_movement_action_count", authorization)
            serialized = json.dumps(payload)
            for forbidden in ("location_stats", "inventory", "rgb", "messages", "workflow_stage", "pitch"): self.assertNotIn(forbidden, serialized)
            self.assertEqual(cls.instances[0].calls, ["open", "reset", "step", "close"])
            self.assertEqual(cls.instances[0].kwargs, {"max_reset_attempts": 1})
        finally: temporary.cleanup()

    def test_reset_failure_separates_plan_from_actual_and_persists_traceback(self):
        record, output, cls, temporary = self.run_stub(fail_reset=True)
        try:
            self.assertFalse(record.success); self.assertEqual(record.outcome, "reset_failed")
            payload = json.loads((output / "e5_movement.json").read_text())
            authorization = json.loads((output / "authorization.json").read_text())
            self.assertEqual(authorization["planned_tested_movement_action_count"], 1)
            self.assertEqual(payload["tested_action_count"], 0)
            self.assertEqual(payload["failure_stage"], "reset")
            self.assertEqual(payload["original_exception_type"], "TypeError")
            self.assertEqual(payload["reset_attempt_count"], 1)
            self.assertEqual(payload["environment_launch_count"], 1)
            self.assertIn("The above exception was the direct cause", payload["exception_traceback"])
            self.assertIn("TypeError: a bytes-like object", payload["exception_traceback"])
            self.assertEqual(payload["failure_cause"], "unknown")
            self.assertEqual(payload["evidence_manifest"], {"logs": []})
            self.assertEqual(cls.instances[0].calls, ["open", "reset", "close"])
            self.assertIsNone(payload["moved"]); self.assertIsNone(payload["forward_projection"])
        finally: temporary.cleanup()

    def test_native_crash_classification_requires_explicit_new_log_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            logs = Path(directory) / "logs"; watchers = logs / "minerl_watchers"; watchers.mkdir(parents=True)
            with patch("obsidianlink.env.integration.e5_run.RUNTIME_LOGS_ROOT", logs.resolve()):
                before = _snapshot_runtime_logs()
                crash = Path(directory) / "hs_err_pid123.log"
                crash.write_text("A fatal error has been detected by the Java Runtime Environment\nSIGSEGV\nliblwjgl_stb.dylib\nSound engine\n")
                (logs / "mc_1.log").write_text(f"A fatal error has been detected by the Java Runtime Environment\nSIGSEGV\nliblwjgl_stb.dylib\n{crash}\n")
                (watchers / "watcher_1.log").write_text("Child is not running anymore\n")
                cause, evidence = _collect_runtime_log_evidence(before)
        self.assertEqual(cause, "minecraft_native_crash")
        self.assertEqual({item.kind for item in evidence}, {"minecraft", "jvm_crash", "process_watcher"})
        self.assertTrue(all(len(item.sha256) == 64 for item in evidence))

    def test_failures_cleanup_and_only_one_attempt(self):
        for kwargs, outcome in (({"after_z": 0.0}, "movement_no_displacement"), ({"missing_after": True}, "position_after_missing"), ({"fail_step": True}, "runtime_error"), ({"dirty": True}, "cleanup_failed")):
            reset_authorized_e5_process_guards_for_tests()
            record, _, _, temporary = self.run_stub(**kwargs)
            try: self.assertFalse(record.success); self.assertEqual(record.outcome, outcome)
            finally: temporary.cleanup()
        reset_authorized_e5_process_guards_for_tests(); record, _, _, temporary = self.run_stub()
        try:
            root = Path(temporary.name) / "p1_e5_movement"
            with patch("obsidianlink.env.integration.e5_run.FORMAL_E5_RUNS_ROOT", root.resolve()), self.assertRaisesRegex(E5AuthorizationError, "one real run"):
                run_authorized_e5_minerl(execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E5, authorized_live_run=AUTHORIZED_LIVE_E5_RUN_VALUE, output_dir=root / "run-2", episode_id=EPISODE)
        finally: temporary.cleanup()


if __name__ == "__main__": unittest.main()

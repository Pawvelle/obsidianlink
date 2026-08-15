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
from obsidianlink.env.integration.e7_run import (
    AUTHORIZED_LIVE_E7_LAVA_RUN_VALUE,
    AUTHORIZED_LIVE_E7_WATER_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E7_LAVA,
    EXECUTION_MODE_AUTHORIZED_LIVE_E7_WATER,
    E7AuthorizationError,
    E7MineRLRunRecord,
    main,
    preflight_authorized_e7,
    reset_authorized_e7_process_guards_for_tests,
    run_authorized_e7_minerl,
)


EPISODE = "e7-live-offline"


class _Backend:
    instances = []
    after_fluid = "water"
    after_inventory = {"bucket": 1}
    missing_after = False
    dirty = False
    fail_step = False
    fail_reset = False
    variant = "water"

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
        filled = "water_bucket" if type(self).variant == "water" else "lava_bucket"
        return {
            "agent_1": SimpleNamespace(
                episode_id=EPISODE,
                agent_id="agent_1",
                step_id=0,
                visible_inventory={filled: 1},
                selected_item=filled,
            )
        }

    def get_reset_audit(self):
        return {"reset_attempt_count": 1, "environment_launch_count": 1}

    def get_bucket_fluid_truth(self, cell):
        if self.step_id and type(self).missing_after:
            return None
        fluid = "none" if not self.step_id else type(self).after_fluid
        return {
            "episode_id": EPISODE,
            "agent_id": "agent_1",
            "step_id": self.step_id,
            "world_x": cell[0],
            "world_y": cell[1],
            "world_z": cell[2],
            "grid_x": 0,
            "grid_y": 0,
            "grid_z": 1,
            "fluid": fluid,
            "fluid_present": fluid != "none",
        }

    def step(self, actions):
        self.calls.append("step")
        if type(self).fail_step:
            raise RuntimeError("step boom")
        self.step_id = 1
        obs = Observation(
            EPISODE,
            "agent_1",
            1,
            0.0,
            frame="unused",
            visible_inventory=dict(type(self).after_inventory),
            selected_item="bucket",
        )
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


class E7LiveGateTests(unittest.TestCase):
    def setUp(self):
        reset_authorized_e7_process_guards_for_tests()

    def tearDown(self):
        reset_authorized_e7_process_guards_for_tests()

    def run_stub(self, variant="water", **kwargs):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name) / "p1_e7_bucket_usage"
        (root / variant).mkdir(parents=True)
        output = root / variant / "run-1"
        cls = backend_cls(variant=variant, **kwargs)
        mode = (
            EXECUTION_MODE_AUTHORIZED_LIVE_E7_WATER
            if variant == "water"
            else EXECUTION_MODE_AUTHORIZED_LIVE_E7_LAVA
        )
        token = (
            AUTHORIZED_LIVE_E7_WATER_RUN_VALUE
            if variant == "water"
            else AUTHORIZED_LIVE_E7_LAVA_RUN_VALUE
        )
        with patch("obsidianlink.env.integration.e7_run.FORMAL_E7_RUNS_ROOT", root.resolve()), patch(
            "obsidianlink.env.integration.e7_run._production_backend_cls", return_value=cls
        ):
            record = run_authorized_e7_minerl(
                execution_mode=mode,
                authorized_live_run=token,
                output_dir=output,
                episode_id=EPISODE,
            )
        return record, output, cls, temporary

    def test_import_and_check_do_not_resolve_backend(self):
        module = importlib.import_module("obsidianlink.env.integration.e7_run")
        stdout = io.StringIO()
        with patch.object(module, "_production_backend_cls") as production, contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["--check"]), 0)
            production.assert_not_called()
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["production_backend_constructed"])
        self.assertFalse(payload["integration_verified"])
        self.assertEqual(payload["variants"]["water"]["bucket_item"], "water_bucket")
        self.assertEqual(payload["variants"]["lava"]["bucket_item"], "lava_bucket")

    def test_missing_wrong_mode_and_token_refuse_before_backend(self):
        attempts = (
            (None, AUTHORIZED_LIVE_E7_WATER_RUN_VALUE),
            ("offline", AUTHORIZED_LIVE_E7_WATER_RUN_VALUE),
            (EXECUTION_MODE_AUTHORIZED_LIVE_E7_WATER, None),
            (EXECUTION_MODE_AUTHORIZED_LIVE_E7_WATER, "E7"),
            (EXECUTION_MODE_AUTHORIZED_LIVE_E7_WATER, AUTHORIZED_LIVE_E7_LAVA_RUN_VALUE),
            (EXECUTION_MODE_AUTHORIZED_LIVE_E7_LAVA, AUTHORIZED_LIVE_E7_WATER_RUN_VALUE),
        )
        with patch("obsidianlink.env.integration.e7_run._production_backend_cls") as production:
            for mode, token in attempts:
                with self.assertRaises(E7AuthorizationError):
                    run_authorized_e7_minerl(
                        execution_mode=mode, authorized_live_run=token, output_dir=Path("/unused")
                    )  # type: ignore[arg-type]
            production.assert_not_called()

    def test_exact_preflight_remains_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "p1_e7_bucket_usage"
            (root / "water").mkdir(parents=True)
            with patch("obsidianlink.env.integration.e7_run.FORMAL_E7_RUNS_ROOT", root.resolve()), patch(
                "obsidianlink.env.integration.e7_run._production_backend_cls"
            ) as production:
                payload = preflight_authorized_e7(
                    execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E7_WATER,
                    authorized_live_run=AUTHORIZED_LIVE_E7_WATER_RUN_VALUE,
                    output_dir=root / "water" / "preflight",
                )
                production.assert_not_called()
        self.assertEqual(payload["variant"], "water")
        self.assertEqual(payload["bucket_item"], "water_bucket")
        self.assertEqual(payload["target_world_cell"], [0, 4, 1])
        self.assertEqual(payload["target_grid_cell"], [0, 0, 1])

    def test_stub_success_writes_narrow_evidence_and_one_step(self):
        record, output, cls, temporary = self.run_stub()
        try:
            self.assertIsInstance(record, E7MineRLRunRecord)
            self.assertTrue(record.success)
            self.assertFalse(record.real_execution_performed)
            payload = json.loads((output / "e7_bucket_usage.json").read_text())
            self.assertEqual(payload["after_fluid"], "water")
            self.assertEqual(payload["after_inventory"], {"bucket": 1})
            self.assertEqual(payload["target_world_cell"], [0, 4, 1])
            self.assertEqual(payload["target_grid_cell"], [0, 0, 1])
            self.assertEqual(payload["tested_action_count"], 1)
            self.assertFalse(payload["integration_verified"])
            authorization = json.loads((output / "authorization.json").read_text())
            self.assertEqual(authorization["planned_tested_bucket_action_count"], 1)
            self.assertEqual(authorization["variant"], "water")
            serialized = json.dumps(payload)
            for forbidden in ("portal_grid", "messages", "workflow_stage"):
                self.assertNotIn(forbidden, serialized)
            self.assertEqual(cls.instances[0].calls, ["open", "reset", "step", "close"])
            self.assertEqual(cls.instances[0].kwargs["max_reset_attempts"], 1)
            self.assertIn("env_factory", cls.instances[0].kwargs)
        finally:
            temporary.cleanup()

    def test_lava_token_cannot_be_reused_for_water(self):
        record, _, _, temporary = self.run_stub(variant="lava", after_fluid="lava")
        try:
            self.assertTrue(record.success)
            self.assertEqual(record.variant, "lava")
            self.assertEqual(record.lifecycle.after_fluid, "lava")
        finally:
            temporary.cleanup()

    def test_reset_failure_separates_plan_from_actual_and_persists_traceback(self):
        record, output, cls, temporary = self.run_stub(fail_reset=True)
        try:
            self.assertFalse(record.success)
            self.assertEqual(record.outcome, "reset_failed")
            payload = json.loads((output / "e7_bucket_usage.json").read_text())
            authorization = json.loads((output / "authorization.json").read_text())
            self.assertEqual(authorization["planned_tested_bucket_action_count"], 1)
            self.assertEqual(payload["tested_action_count"], 0)
            self.assertEqual(payload["failure_stage"], "reset")
            self.assertEqual(payload["original_exception_type"], "TypeError")
            self.assertIn("TypeError: a bytes-like object", payload["exception_traceback"])
            self.assertEqual(cls.instances[0].calls, ["open", "reset", "close"])
            self.assertIsNone(payload["fluid_changed"])
        finally:
            temporary.cleanup()

    def test_failures_cleanup_and_only_one_attempt(self):
        for kwargs, outcome in (
            ({"after_fluid": "none"}, "bucket_no_world_effect"),
            ({"missing_after": True}, "fluid_after_missing"),
            ({"fail_step": True}, "action_failed"),
            ({"dirty": True}, "cleanup_failed"),
        ):
            reset_authorized_e7_process_guards_for_tests()
            record, _, _, temporary = self.run_stub(**kwargs)
            try:
                self.assertFalse(record.success)
                self.assertEqual(record.outcome, outcome)
            finally:
                temporary.cleanup()
        reset_authorized_e7_process_guards_for_tests()
        record, _, _, temporary = self.run_stub()
        try:
            root = Path(temporary.name) / "p1_e7_bucket_usage"
            with patch("obsidianlink.env.integration.e7_run.FORMAL_E7_RUNS_ROOT", root.resolve()), self.assertRaisesRegex(
                E7AuthorizationError, "one real run"
            ):
                run_authorized_e7_minerl(
                    execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E7_WATER,
                    authorized_live_run=AUTHORIZED_LIVE_E7_WATER_RUN_VALUE,
                    output_dir=root / "water" / "run-2",
                    episode_id=EPISODE,
                )
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()

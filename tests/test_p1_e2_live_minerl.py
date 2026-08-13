from __future__ import annotations

import ast
import contextlib
import importlib
import inspect
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from obsidianlink.env.integration import (
    AUTHORIZED_LIVE_E2_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E2,
    E2AuthorizationError,
    E2MineRLRunRecord,
    preflight_authorized_e2,
    run_authorized_e2_minerl,
)
from obsidianlink.env.integration.e2_config import E2_CALIBRATION_INVENTORY
from obsidianlink.env.integration.e2_run import (
    build_parser,
    main,
    reset_authorized_e2_process_guards_for_tests,
)
from obsidianlink.env.validation import p1_validation_manifest
from obsidianlink.env.validation.result import UNIT_VERIFIED


ROOT = Path(__file__).resolve().parents[1]
EPISODE_ID = "e2-live-offline-episode"


class _RecordingMineRLBackend:
    instances: list["_RecordingMineRLBackend"] = []
    reset_result: object = None
    close_error: Exception | None = None
    leave_cleanup_dirty = False

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self.calls: list[object] = []
        self._opened = False
        self._env: object | None = None
        self._owner_thread: int | None = None
        type(self).instances.append(self)

    def open(self) -> None:
        self.calls.append("open")
        self._opened = True
        self._owner_thread = 1

    def reset(self, task: object) -> object:
        self.calls.append(("reset", getattr(task, "task_id", None)))
        self._env = object()
        if isinstance(type(self).reset_result, BaseException):
            raise type(self).reset_result
        if type(self).reset_result is not None:
            return type(self).reset_result
        return _observation(dict(E2_CALIBRATION_INVENTORY))

    def close(self) -> None:
        self.calls.append("close")
        if type(self).close_error is not None:
            raise type(self).close_error
        if not type(self).leave_cleanup_dirty:
            self._env = None
            self._owner_thread = None
            self._opened = False

    def step(self, action: object) -> None:
        self.calls.append(("step", action))
        raise AssertionError("E2 live bridge must not execute actions")


def _backend_cls(
    reset_result: object,
    *,
    close_error: Exception | None = None,
    leave_cleanup_dirty: bool = False,
) -> type[_RecordingMineRLBackend]:
    class _Configured(_RecordingMineRLBackend):
        instances: list[_RecordingMineRLBackend] = []

    _Configured.__name__ = "RecordingMineRLBackend"
    _Configured.reset_result = reset_result
    _Configured.close_error = close_error
    _Configured.leave_cleanup_dirty = leave_cleanup_dirty
    return _Configured


def _observation(inventory: object, **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "agent_id": "agent_1",
        "episode_id": EPISODE_ID,
        "step_id": 0,
        "visible_inventory": inventory,
        "frame": "must-not-be-evidence",
        "selected_item": "must-not-be-evidence",
        "portal_grid": "must-not-be-evidence",
    }
    payload.update(extra)
    return {"agent_1": SimpleNamespace(**payload)}


class E2LiveRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_authorized_e2_process_guards_for_tests()

    def tearDown(self) -> None:
        reset_authorized_e2_process_guards_for_tests()

    def _run_stub(
        self,
        reset_result: object,
        *,
        close_error: Exception | None = None,
        leave_cleanup_dirty: bool = False,
    ) -> tuple[
        E2MineRLRunRecord,
        Path,
        type[_RecordingMineRLBackend],
        tempfile.TemporaryDirectory[str],
    ]:
        temporary = tempfile.TemporaryDirectory()
        runs_root = Path(temporary.name) / "p1_e2_inventory_observation"
        runs_root.mkdir()
        output_dir = runs_root / "run-1"
        backend_cls = _backend_cls(
            reset_result,
            close_error=close_error,
            leave_cleanup_dirty=leave_cleanup_dirty,
        )
        with patch(
            "obsidianlink.env.integration.e2_run.FORMAL_E2_RUNS_ROOT",
            runs_root.resolve(),
        ), patch(
            "obsidianlink.env.integration.e2_run._production_backend_cls",
            return_value=backend_cls,
        ):
            record = run_authorized_e2_minerl(
                execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E2,
                authorized_live_run=AUTHORIZED_LIVE_E2_RUN_VALUE,
                output_dir=output_dir,
                episode_id=EPISODE_ID,
            )
        self.assertIsInstance(record, E2MineRLRunRecord)
        return record, output_dir, backend_cls, temporary

    def test_import_and_check_do_not_resolve_or_construct_backend(self) -> None:
        module = importlib.import_module("obsidianlink.env.integration.e2_run")
        self.assertTrue(hasattr(module, "run_authorized_e2_minerl"))
        stdout = io.StringIO()
        with patch.object(module, "_production_backend_cls") as production:
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(main(["--check"]), 0)
            production.assert_not_called()
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["production_backend_constructed"])
        self.assertFalse(payload["real_execution_performed"])
        self.assertFalse(payload["integration_verified"])

    def test_authorization_requires_both_exact_values_before_backend(self) -> None:
        attempts = (
            (None, AUTHORIZED_LIVE_E2_RUN_VALUE, "execution_mode"),
            ("offline", AUTHORIZED_LIVE_E2_RUN_VALUE, "execution_mode"),
            (EXECUTION_MODE_AUTHORIZED_LIVE_E2, None, "authorized_live_run"),
            (EXECUTION_MODE_AUTHORIZED_LIVE_E2, "E2", "authorized_live_run"),
        )
        with patch(
            "obsidianlink.env.integration.e2_run._production_backend_cls"
        ) as production:
            for execution_mode, live_value, message in attempts:
                with self.subTest(execution_mode=execution_mode, live_value=live_value):
                    with self.assertRaisesRegex(E2AuthorizationError, message):
                        run_authorized_e2_minerl(
                            execution_mode=execution_mode,  # type: ignore[arg-type]
                            authorized_live_run=live_value,  # type: ignore[arg-type]
                            output_dir=Path("/unused"),
                        )
            production.assert_not_called()

    def test_preflight_is_lazy_and_checks_frozen_e2_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory) / "p1_e2_inventory_observation"
            runs_root.mkdir()
            output_dir = runs_root / "preflight"
            with patch(
                "obsidianlink.env.integration.e2_run.FORMAL_E2_RUNS_ROOT",
                runs_root.resolve(),
            ), patch(
                "obsidianlink.env.integration.e2_run._production_backend_cls"
            ) as production:
                payload = preflight_authorized_e2(
                    execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E2,
                    authorized_live_run=AUTHORIZED_LIVE_E2_RUN_VALUE,
                    output_dir=output_dir,
                )
                production.assert_not_called()
        self.assertEqual(payload["check_id"], "E2")
        self.assertEqual(payload["name"], "inventory_observation")
        self.assertEqual(
            payload["expected_inventory"], dict(E2_CALIBRATION_INVENTORY)
        )
        self.assertFalse(payload["requires_server_truth"])
        self.assertTrue(payload["calibration_only"])
        self.assertFalse(payload["integration_verified"])

    def test_cli_has_no_expected_inventory_override(self) -> None:
        actions = {action.dest for action in build_parser()._actions}
        self.assertNotIn("expected_inventory", actions)
        self.assertNotIn("inventory_json", actions)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["--expected-inventory", "{}"])

    def test_exact_inventory_writes_narrow_offline_evidence(self) -> None:
        record, output_dir, backend_cls, temporary = self._run_stub(
            _observation(dict(E2_CALIBRATION_INVENTORY))
        )
        try:
            self.assertTrue(record.success)
            self.assertEqual(record.outcome, "inventory_ok")
            self.assertEqual(
                record.expected_inventory, dict(E2_CALIBRATION_INVENTORY)
            )
            self.assertEqual(
                record.observed_inventory, dict(E2_CALIBRATION_INVENTORY)
            )
            self.assertFalse(record.real_execution_performed)
            self.assertFalse(record.integration_verified)
            self.assertEqual(record.verification_level, UNIT_VERIFIED)
            self.assertTrue(record.calibration_only)
            backend = backend_cls.instances[0]
            self.assertEqual(backend.calls[0], "open")
            self.assertEqual(backend.calls[-1], "close")
            self.assertFalse(any(call == "step" for call in backend.calls))

            payload = json.loads(
                (output_dir / "e2_inventory.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload["expected_inventory"], dict(E2_CALIBRATION_INVENTORY)
            )
            self.assertEqual(
                payload["observed_inventory"], dict(E2_CALIBRATION_INVENTORY)
            )
            serialized = json.dumps(payload)
            for forbidden in (
                "rgb",
                "frame",
                "selected_item",
                "portal_grid",
                "evaluator",
            ):
                self.assertNotIn(forbidden, serialized)
            self.assertEqual(
                set(payload["cleanup"]),
                {
                    "backend_marked_closed",
                    "close_returned",
                    "environment_reference_cleared",
                    "limitation",
                    "owner_cleared",
                    "process_release_proven",
                },
            )
        finally:
            temporary.cleanup()

    def test_observed_inventory_is_not_copied_from_expected(self) -> None:
        record, _, backend_cls, temporary = self._run_stub(
            _observation({"dirt": 2})
        )
        try:
            self.assertFalse(record.success)
            self.assertEqual(record.outcome, "inventory_mismatch")
            self.assertEqual(record.observed_inventory, {"dirt": 2})
            self.assertEqual(
                record.expected_inventory, dict(E2_CALIBRATION_INVENTORY)
            )
            task = backend_cls.instances[0].calls[1]
            self.assertEqual(task[0], "reset")
        finally:
            temporary.cleanup()

    def test_inventory_mismatch_variants_fail_closed(self) -> None:
        variants = (
            {"dirt": 8, "obsidian": 4, "flint_and_steel": 1},
            {"dirt": 7, "obsidian": 4},
            {**dict(E2_CALIBRATION_INVENTORY), "cobblestone": 3},
            {},
        )
        for observed in variants:
            reset_authorized_e2_process_guards_for_tests()
            with self.subTest(observed=observed):
                record, _, _, temporary = self._run_stub(_observation(observed))
                try:
                    self.assertFalse(record.success)
                    self.assertEqual(record.outcome, "inventory_mismatch")
                    self.assertEqual(record.observed_inventory, observed)
                finally:
                    temporary.cleanup()

    def test_invalid_quantity_is_structural_failure_without_coercion(self) -> None:
        record, _, _, temporary = self._run_stub(
            _observation({"obsidian": "4"})
        )
        try:
            self.assertFalse(record.success)
            self.assertEqual(record.outcome, "inventory_quantity_invalid")
            self.assertIsNone(record.observed_inventory)
        finally:
            temporary.cleanup()

    def test_close_and_cleanup_failures_override_inventory_match(self) -> None:
        cases = (
            {"close_error": RuntimeError("close failed")},
            {"leave_cleanup_dirty": True},
        )
        for kwargs in cases:
            reset_authorized_e2_process_guards_for_tests()
            with self.subTest(kwargs=kwargs):
                record, _, _, temporary = self._run_stub(
                    _observation(dict(E2_CALIBRATION_INVENTORY)), **kwargs
                )
                try:
                    self.assertFalse(record.success)
                    if kwargs.get("leave_cleanup_dirty"):
                        self.assertEqual(record.outcome, "cleanup_failed")
                        self.assertTrue(record.cleanup.has_explicit_failure())
                    else:
                        self.assertEqual(record.outcome, "close_failed")
                        self.assertIn("close failed", record.close_error or "")
                finally:
                    temporary.cleanup()

    def test_reset_failure_is_recorded_and_closed(self) -> None:
        record, output_dir, backend_cls, temporary = self._run_stub(
            RuntimeError("reset failed")
        )
        try:
            self.assertFalse(record.success)
            self.assertEqual(record.outcome, "reset_failed")
            self.assertIn("reset failed", record.error or "")
            self.assertTrue(record.closed)
            self.assertEqual(backend_cls.instances[0].calls[-1], "close")
            self.assertTrue((output_dir / "e2_inventory.json").is_file())
        finally:
            temporary.cleanup()

    def test_evidence_write_failure_cannot_return_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory) / "p1_e2_inventory_observation"
            runs_root.mkdir()
            backend_cls = _backend_cls(
                _observation(dict(E2_CALIBRATION_INVENTORY))
            )
            with patch(
                "obsidianlink.env.integration.e2_run.FORMAL_E2_RUNS_ROOT",
                runs_root.resolve(),
            ), patch(
                "obsidianlink.env.integration.e2_run._production_backend_cls",
                return_value=backend_cls,
            ), patch(
                "obsidianlink.env.integration.e2_run._write_evidence",
                side_effect=OSError("evidence write failed"),
            ):
                with self.assertRaisesRegex(OSError, "evidence write failed"):
                    run_authorized_e2_minerl(
                        execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E2,
                        authorized_live_run=AUTHORIZED_LIVE_E2_RUN_VALUE,
                        output_dir=runs_root / "run-1",
                        episode_id=EPISODE_ID,
                    )
            self.assertFalse((runs_root / "run-1").exists())

    def test_second_live_attempt_in_process_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory) / "p1_e2_inventory_observation"
            runs_root.mkdir()
            backend_cls = _backend_cls(
                _observation(dict(E2_CALIBRATION_INVENTORY))
            )
            with patch(
                "obsidianlink.env.integration.e2_run.FORMAL_E2_RUNS_ROOT",
                runs_root.resolve(),
            ), patch(
                "obsidianlink.env.integration.e2_run._production_backend_cls",
                return_value=backend_cls,
            ) as production:
                run_authorized_e2_minerl(
                    execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E2,
                    authorized_live_run=AUTHORIZED_LIVE_E2_RUN_VALUE,
                    output_dir=runs_root / "run-1",
                    episode_id=EPISODE_ID,
                )
                with self.assertRaisesRegex(E2AuthorizationError, "one real run"):
                    run_authorized_e2_minerl(
                        execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E2,
                        authorized_live_run=AUTHORIZED_LIVE_E2_RUN_VALUE,
                        output_dir=runs_root / "run-2",
                        episode_id=EPISODE_ID,
                    )
                self.assertEqual(production.call_count, 1)
                self.assertEqual(len(backend_cls.instances), 1)

    def test_source_is_lazy_and_contains_no_gradle_action_or_model_execution(self) -> None:
        path = ROOT / "obsidianlink/env/integration/e2_run.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        top_level_imports = [
            node
            for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        imported = "\n".join(ast.unparse(node) for node in top_level_imports)
        self.assertNotIn("obsidianlink.env.minerl_backend", imported)
        self.assertNotIn("subprocess", imported)
        self.assertNotIn("gradlew", source.lower())
        run_source = inspect.getsource(run_authorized_e2_minerl)
        self.assertNotIn(".step(", run_source)
        self.assertNotIn("model", run_source.lower())
        self.assertIn("_production_backend_cls()", run_source)

    def test_manifest_stays_not_run_and_e3_is_unimplemented(self) -> None:
        manifest = p1_validation_manifest()
        self.assertEqual(manifest[2]["status"], "not_run")
        self.assertEqual(manifest[3]["status"], "not_run")
        self.assertEqual(manifest[3]["check_id"], "E3")
        self.assertEqual(manifest[3]["name"], "selected_item")


if __name__ == "__main__":
    unittest.main()

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
    AUTHORIZED_LIVE_E3_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E3,
    E3AuthorizationError,
    E3MineRLRunRecord,
    preflight_authorized_e3,
    run_authorized_e3_minerl,
)
from obsidianlink.env.integration.e3_config import E3_EXPECTED_SELECTED_ITEM
from obsidianlink.env.integration.e3_run import main, reset_authorized_e3_process_guards_for_tests


EPISODE_ID = "e3-live-offline-episode"


def _observation(item: object) -> dict[str, object]:
    return {"agent_1": SimpleNamespace(
        agent_id="agent_1", episode_id=EPISODE_ID, step_id=0,
        selected_item=item, visible_inventory={E3_EXPECTED_SELECTED_ITEM: 1},
        frame="drop", portal_grid="drop", equipped_items="drop",
    )}


class _Backend:
    instances = []
    reset_result: object = None
    close_error = None
    dirty = False
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls = []
        self._opened = False
        self._env = None
        self._owner_thread = None
        type(self).instances.append(self)
    def open(self):
        self.calls.append("open"); self._opened = True
    def reset(self, task):
        self.calls.append("reset"); self._env = object()
        if isinstance(type(self).reset_result, BaseException): raise type(self).reset_result
        return type(self).reset_result
    def close(self):
        self.calls.append("close")
        if type(self).close_error: raise type(self).close_error
        if not type(self).dirty:
            self._opened = False; self._env = None; self._owner_thread = None
    def step(self, action):
        raise AssertionError("E3 live runner must not step")


def _backend_cls(result: object, *, close_error=None, dirty=False):
    class Configured(_Backend):
        instances = []
    Configured.__name__ = "RecordingMineRLBackend"
    Configured.reset_result = result
    Configured.close_error = close_error
    Configured.dirty = dirty
    return Configured


class E3LiveGateTests(unittest.TestCase):
    def setUp(self): reset_authorized_e3_process_guards_for_tests()
    def tearDown(self): reset_authorized_e3_process_guards_for_tests()

    def _run(self, result: object, **kwargs):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name) / "p1_e3_selected_item_observation"
        root.mkdir()
        output = root / "run-1"
        backend = _backend_cls(result, **kwargs)
        with patch("obsidianlink.env.integration.e3_run.FORMAL_E3_RUNS_ROOT", root.resolve()), patch("obsidianlink.env.integration.e3_run._production_backend_cls", return_value=backend):
            record = run_authorized_e3_minerl(
                execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E3,
                authorized_live_run=AUTHORIZED_LIVE_E3_RUN_VALUE,
                output_dir=output, episode_id=EPISODE_ID,
            )
        return record, output, backend, temporary

    def test_import_and_check_do_not_start_or_resolve_minerl(self) -> None:
        module = importlib.import_module("obsidianlink.env.integration.e3_run")
        stdout = io.StringIO()
        with patch.object(module, "_production_backend_cls") as production, contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["--check"]), 0)
            production.assert_not_called()
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["production_backend_constructed"])
        self.assertFalse(payload["real_execution_performed"])

    def test_missing_or_wrong_gate_refuses_before_backend(self) -> None:
        attempts = ((None, AUTHORIZED_LIVE_E3_RUN_VALUE), ("offline", AUTHORIZED_LIVE_E3_RUN_VALUE), (EXECUTION_MODE_AUTHORIZED_LIVE_E3, None), (EXECUTION_MODE_AUTHORIZED_LIVE_E3, "E3"))
        with patch("obsidianlink.env.integration.e3_run._production_backend_cls") as production:
            for mode, token in attempts:
                with self.assertRaises(E3AuthorizationError):
                    run_authorized_e3_minerl(execution_mode=mode, authorized_live_run=token, output_dir=Path("/unused"))  # type: ignore[arg-type]
            production.assert_not_called()

    def test_exact_gate_preflight_remains_offline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "p1_e3_selected_item_observation"; root.mkdir()
            with patch("obsidianlink.env.integration.e3_run.FORMAL_E3_RUNS_ROOT", root.resolve()), patch("obsidianlink.env.integration.e3_run._production_backend_cls") as production:
                payload = preflight_authorized_e3(execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E3, authorized_live_run=AUTHORIZED_LIVE_E3_RUN_VALUE, output_dir=root / "preflight")
                production.assert_not_called()
        self.assertEqual(payload["expected_selected_item"], E3_EXPECTED_SELECTED_ITEM)
        self.assertFalse(payload["real_execution_performed"])

    def test_stub_success_writes_narrow_deterministic_evidence(self) -> None:
        record, output, backend, temporary = self._run(_observation(E3_EXPECTED_SELECTED_ITEM))
        try:
            self.assertIsInstance(record, E3MineRLRunRecord)
            self.assertTrue(record.success)
            self.assertFalse(record.real_execution_performed)
            payload = json.loads((output / "e3_selected_item.json").read_text())
            self.assertEqual(payload["observed_selected_item"], E3_EXPECTED_SELECTED_ITEM)
            serialized = json.dumps(payload)
            for forbidden in ("inventory", "rgb", "frame", "portal_grid", "equipped_items"):
                self.assertNotIn(forbidden, serialized)
            self.assertEqual(backend.instances[0].calls, ["open", "reset", "close"])
        finally: temporary.cleanup()

    def test_mismatch_none_malformed_lifecycle_and_cleanup_fail_closed(self) -> None:
        cases = (
            (_observation("obsidian"), {}, "selected_item_mismatch"),
            (_observation(None), {}, "selected_item_none"),
            (_observation(4), {}, "selected_item_type_invalid"),
            (RuntimeError("reset"), {}, "reset_failed"),
            (_observation(E3_EXPECTED_SELECTED_ITEM), {"close_error": RuntimeError("close")}, "close_failed"),
            (_observation(E3_EXPECTED_SELECTED_ITEM), {"dirty": True}, "cleanup_failed"),
        )
        for result, kwargs, outcome in cases:
            reset_authorized_e3_process_guards_for_tests()
            with self.subTest(outcome=outcome):
                record, _, _, temporary = self._run(result, **kwargs)
                try:
                    self.assertFalse(record.success)
                    self.assertEqual(record.outcome, outcome)
                finally: temporary.cleanup()

    def test_only_one_execution_attempt_per_process(self) -> None:
        record, _, _, temporary = self._run(_observation(E3_EXPECTED_SELECTED_ITEM))
        try:
            self.assertTrue(record.success)
            root = Path(temporary.name) / "p1_e3_selected_item_observation"
            with patch(
                "obsidianlink.env.integration.e3_run.FORMAL_E3_RUNS_ROOT",
                root.resolve(),
            ):
                with self.assertRaisesRegex(E3AuthorizationError, "one real run"):
                    run_authorized_e3_minerl(execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E3, authorized_live_run=AUTHORIZED_LIVE_E3_RUN_VALUE, output_dir=root / "run-2", episode_id=EPISODE_ID)
        finally: temporary.cleanup()

    def test_source_is_lazy_and_has_no_action_model_or_gradle(self) -> None:
        source = inspect.getsource(run_authorized_e3_minerl)
        self.assertNotIn(".step(", source)
        self.assertNotIn("model", source.lower())
        path = Path(__file__).resolve().parents[1] / "obsidianlink/env/integration/e3_run.py"
        full = path.read_text()
        tree = ast.parse(full)
        imports = "\n".join(ast.unparse(n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))
        self.assertNotIn("obsidianlink.env.minerl_backend", imports)
        self.assertNotIn("subprocess", imports)
        self.assertNotIn("gradlew", full.lower())


if __name__ == "__main__": unittest.main()

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from obsidianlink.env.integration import (
    AUTHORIZED_LIVE_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E0,
    E0AuthorizationError,
    E0CleanupStatus,
    E0MineRLRunRecord,
    MineRLE0LifecycleAdapter,
    preflight_authorized_e0,
    run_authorized_e0_minerl,
)
from obsidianlink.env.integration.e0_cleanup import inspect_minerl_cleanup
from obsidianlink.env.integration.e0_config import (
    E0_COMPATIBILITY_WORKFLOW,
    build_e0_compatibility_task,
)
from obsidianlink.env.integration.e0_run import reset_authorized_e0_process_guards_for_tests
from obsidianlink.env.validation import (
    E0_LIFECYCLE_CASE,
    EnvironmentValidationRunner,
    p1_validation_manifest,
)
from obsidianlink.env.validation.result import UNIT_VERIFIED


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PACKAGE = ROOT / "obsidianlink/env/validation"
INTEGRATION_PACKAGE = ROOT / "obsidianlink/env/integration"
EPISODE_ID = "e0-bridge-episode"

BANNED_VALIDATION_PREFIXES = (
    "obsidianlink.drivers",
    "obsidianlink.runners",
    "obsidianlink.evaluation",
    "obsidianlink.agents",
    "obsidianlink.benchmark.evaluator",
    "obsidianlink.env.fake",
    "obsidianlink.env.minerl_backend",
    "obsidianlink.env.integration",
    "minerl",
)
BANNED_INTEGRATION_PREFIXES = (
    "obsidianlink.drivers",
    "obsidianlink.runners",
    "obsidianlink.evaluation",
    "obsidianlink.agents",
)


def _top_level_imported_modules(source: Path) -> tuple[str, ...]:
    modules: list[str] = []
    tree = ast.parse(source.read_text(encoding="utf-8"))
    relative = source.relative_to(ROOT).with_suffix("")
    package_parts = relative.parts if source.name == "__init__.py" else relative.parts[:-1]
    package = ".".join(package_parts)
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative_name = "." * node.level + (node.module or "")
                modules.append(importlib.util.resolve_name(relative_name, package))
            elif node.module:
                modules.append(node.module)
    return tuple(modules)


def _imported_modules(source: Path) -> tuple[str, ...]:
    modules: list[str] = []
    tree = ast.parse(source.read_text(encoding="utf-8"))
    relative = source.relative_to(ROOT).with_suffix("")
    package_parts = relative.parts if source.name == "__init__.py" else relative.parts[:-1]
    package = ".".join(package_parts)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative_name = "." * node.level + (node.module or "")
                modules.append(importlib.util.resolve_name(relative_name, package))
            elif node.module:
                modules.append(node.module)
    return tuple(modules)


def _matches_prefix(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


class _RecordingMineRLBackend:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self.calls: list[object] = []
        self._opened = False
        self._env: object | None = None
        self._task = None
        self._owner_thread: int | None = None
        self.open_error: Exception | None = None
        self.reset_error: Exception | None = None
        self.close_error: Exception | None = None
        self.reset_result: object | None = None

    def open(self) -> None:
        self.calls.append("open")
        if self.open_error is not None:
            raise self.open_error
        self._opened = True
        self._owner_thread = 1

    def reset(self, task: object) -> object:
        self.calls.append(("reset", type(task).__name__, getattr(task, "task_id", None)))
        if self.reset_error is not None:
            raise self.reset_error
        self._task = task
        self._env = object()
        if self.reset_result is not None:
            return self.reset_result
        return {
            "agent_1": SimpleNamespace(
                episode_id=getattr(task, "task_id", EPISODE_ID),
                step_id=0,
                agent_id="agent_1",
                frame={"pixels": "not-for-e0"},
                visible_inventory={"obsidian": 14},
                selected_item="flint_and_steel",
                workflow_stage="route_a_a0",
            )
        }

    def close(self) -> None:
        self.calls.append("close")
        if self.close_error is not None:
            raise self.close_error
        self._env = None
        self._task = None
        self._owner_thread = None
        self._opened = False


def _backend_cls(
    *,
    open_error: Exception | None = None,
    reset_error: Exception | None = None,
    close_error: Exception | None = None,
    reset_result: object | None = None,
) -> type[_RecordingMineRLBackend]:
    class _Configured(_RecordingMineRLBackend):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.open_error = open_error
            self.reset_error = reset_error
            self.close_error = close_error
            self.reset_result = reset_result

    _Configured.__name__ = "RecordingMineRLBackend"
    return _Configured


class AdapterLifecycleTests(unittest.TestCase):
    def test_construct_open_reset_close_order(self) -> None:
        adapter = MineRLE0LifecycleAdapter(
            episode_id=EPISODE_ID,
            backend_cls=_backend_cls(),
        )
        state = adapter.reset()
        adapter.close()
        backend = adapter._backend
        assert isinstance(backend, _RecordingMineRLBackend)
        self.assertEqual(
            [item if isinstance(item, str) else item[0] for item in backend.calls],
            ["open", "reset", "close"],
        )
        self.assertEqual(state["agent_1"]["episode_id"], EPISODE_ID)
        self.assertEqual(state["agent_1"]["step_id"], 0)
        self.assertNotIn("frame", state["agent_1"])
        self.assertNotIn("visible_inventory", state["agent_1"])
        self.assertNotIn("selected_item", state["agent_1"])
        self.assertNotIn("workflow_stage", state["agent_1"])
        self.assertTrue(backend.calls[1][1] == "TaskInstance")
        self.assertFalse(adapter.opened)
        self.assertTrue(adapter.open_succeeded)
        cleanup = adapter.cleanup_status()
        self.assertTrue(cleanup.close_returned)
        self.assertTrue(cleanup.backend_marked_closed)
        self.assertTrue(cleanup.environment_reference_cleared)
        self.assertTrue(cleanup.owner_cleared)
        self.assertFalse(cleanup.process_release_proven)

    def test_open_failure_fails_closed_and_still_closes(self) -> None:
        factory = MineRLE0LifecycleAdapter.lifecycle_factory(
            episode_id=EPISODE_ID,
            backend_cls=_backend_cls(open_error=RuntimeError("open exploded")),
        )
        holder: dict[str, MineRLE0LifecycleAdapter | None] = {"adapter": None}

        def capturing() -> MineRLE0LifecycleAdapter:
            adapter = factory()
            holder["adapter"] = adapter
            return adapter

        result = EnvironmentValidationRunner().run(
            E0_LIFECYCLE_CASE,
            capturing,
            episode_id=EPISODE_ID,
        )
        adapter = holder["adapter"]
        assert adapter is not None
        backend = adapter._backend
        assert isinstance(backend, _RecordingMineRLBackend)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "reset_failed")
        self.assertFalse(result.reset_completed)
        self.assertTrue(result.closed)
        self.assertIn("close", backend.calls)
        self.assertFalse(any(item[0] == "reset" for item in backend.calls if isinstance(item, tuple)))
        self.assertFalse(result.integration_verified)
        self.assertFalse(result.real_execution_performed)

    def test_reset_failure_still_closes(self) -> None:
        factory = MineRLE0LifecycleAdapter.lifecycle_factory(
            episode_id=EPISODE_ID,
            backend_cls=_backend_cls(reset_error=RuntimeError("reset exploded")),
        )
        holder: dict[str, MineRLE0LifecycleAdapter | None] = {"adapter": None}

        def capturing() -> MineRLE0LifecycleAdapter:
            adapter = factory()
            holder["adapter"] = adapter
            return adapter

        result = EnvironmentValidationRunner().run(
            E0_LIFECYCLE_CASE,
            capturing,
            episode_id=EPISODE_ID,
        )
        adapter = holder["adapter"]
        assert adapter is not None
        backend = adapter._backend
        assert isinstance(backend, _RecordingMineRLBackend)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "reset_failed")
        self.assertTrue(result.closed)
        self.assertIn("close", backend.calls)
        self.assertIn("reset exploded", result.error or "")

    def test_missing_initial_state_fails_closed(self) -> None:
        factory = MineRLE0LifecycleAdapter.lifecycle_factory(
            episode_id=EPISODE_ID,
            backend_cls=_backend_cls(reset_result={}),
        )
        result = EnvironmentValidationRunner().run(
            E0_LIFECYCLE_CASE,
            factory,
            episode_id=EPISODE_ID,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "initial_state_missing")
        self.assertTrue(result.closed)

    def test_close_failure_preserves_cleanup_error(self) -> None:
        factory = MineRLE0LifecycleAdapter.lifecycle_factory(
            episode_id=EPISODE_ID,
            backend_cls=_backend_cls(close_error=OSError("close failed")),
        )
        holder: dict[str, MineRLE0LifecycleAdapter | None] = {"adapter": None}

        def capturing() -> MineRLE0LifecycleAdapter:
            adapter = factory()
            holder["adapter"] = adapter
            return adapter

        result = EnvironmentValidationRunner().run(
            E0_LIFECYCLE_CASE,
            capturing,
            episode_id=EPISODE_ID,
        )
        adapter = holder["adapter"]
        assert adapter is not None
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "close_failed")
        self.assertFalse(result.closed)
        self.assertIn("close failed", result.close_error or "")
        self.assertFalse(adapter.cleanup_status().close_returned)
        self.assertFalse(adapter.cleanup_status().process_release_proven)

    def test_reset_signature_does_not_take_a_task(self) -> None:
        parameters = inspect.signature(MineRLE0LifecycleAdapter.reset).parameters
        self.assertEqual(tuple(parameters), ("self",))


class CompatibilityIsolationTests(unittest.TestCase):
    def test_legacy_task_stays_inside_integration(self) -> None:
        task = build_e0_compatibility_task(EPISODE_ID)
        self.assertEqual(task.workflow, E0_COMPATIBILITY_WORKFLOW)
        self.assertEqual(task.task_id, EPISODE_ID)
        import obsidianlink.env.integration as integration

        self.assertNotIn("TaskInstance", integration.__all__)
        self.assertFalse(hasattr(integration, "TaskInstance"))
        self.assertFalse(hasattr(integration, "build_e0_compatibility_task"))

    def test_validation_package_does_not_import_task_instance_or_integration(self) -> None:
        for source in VALIDATION_PACKAGE.rglob("*.py"):
            imported = _imported_modules(source)
            for module in imported:
                for prefix in BANNED_VALIDATION_PREFIXES:
                    self.assertFalse(
                        _matches_prefix(module, prefix),
                        f"{source.relative_to(ROOT)} imports {module}",
                    )
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    names = {alias.name for alias in node.names}
                    self.assertTrue(
                        names.isdisjoint({"TaskInstance", "LegacyTaskInstance"}),
                        f"{source.relative_to(ROOT)} imports legacy TaskInstance",
                    )


class SolverIndependenceTests(unittest.TestCase):
    def test_integration_does_not_import_drivers_evaluators_or_agents(self) -> None:
        sources = tuple(sorted(INTEGRATION_PACKAGE.rglob("*.py")))
        self.assertTrue(sources)
        for source in sources:
            for module in _imported_modules(source):
                for prefix in BANNED_INTEGRATION_PREFIXES:
                    self.assertFalse(
                        _matches_prefix(module, prefix),
                        f"{source.relative_to(ROOT)} imports {module}",
                    )


class ImportSafetyTests(unittest.TestCase):
    def test_minerl_backend_is_not_imported_at_module_level(self) -> None:
        for relative in (
            "obsidianlink/env/integration/__init__.py",
            "obsidianlink/env/integration/e0_adapter.py",
            "obsidianlink/env/integration/e0_run.py",
            "obsidianlink/env/integration/e0_config.py",
            "obsidianlink/env/integration/e0_cleanup.py",
        ):
            source = ROOT / relative
            imported = _top_level_imported_modules(source)
            for module in imported:
                self.assertFalse(
                    _matches_prefix(module, "obsidianlink.env.minerl_backend"),
                    f"{relative} top-level imports {module}",
                )
                self.assertFalse(
                    _matches_prefix(module, "minerl"),
                    f"{relative} top-level imports {module}",
                )


class AuthorizationTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_authorized_e0_process_guards_for_tests()

    def tearDown(self) -> None:
        reset_authorized_e0_process_guards_for_tests()

    def test_entrypoint_refuses_without_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            with patch(
                "obsidianlink.env.integration.e0_run._production_backend_cls"
            ) as production:
                with self.assertRaisesRegex(E0AuthorizationError, "execution_mode"):
                    run_authorized_e0_minerl(
                        execution_mode="offline",
                        authorized_live_run=AUTHORIZED_LIVE_RUN_VALUE,
                        output_dir=output_dir,
                    )
                production.assert_not_called()

    def test_preflight_accepts_flags_without_starting_backend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory) / "p1_e0_reset_close"
            runs_root.mkdir()
            output_dir = runs_root / "run-1"
            with patch(
                "obsidianlink.env.integration.e0_run.FORMAL_E0_RUNS_ROOT",
                runs_root.resolve(),
            ):
                with patch(
                    "obsidianlink.env.integration.e0_run._production_backend_cls"
                ) as production:
                    payload = preflight_authorized_e0(
                        execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E0,
                        authorized_live_run=AUTHORIZED_LIVE_RUN_VALUE,
                        output_dir=output_dir,
                    )
                    production.assert_not_called()
            self.assertFalse(payload["integration_verified"])
            self.assertFalse(payload["real_execution_performed"])
            self.assertEqual(payload["verification_level"], UNIT_VERIFIED)

    def test_authorized_stub_run_stays_unit_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory) / "p1_e0_reset_close"
            runs_root.mkdir()
            output_dir = runs_root / "run-stub"
            with patch(
                "obsidianlink.env.integration.e0_run.FORMAL_E0_RUNS_ROOT",
                runs_root.resolve(),
            ), patch(
                "obsidianlink.env.integration.e0_run._production_backend_cls",
                return_value=_backend_cls(),
            ):
                record = run_authorized_e0_minerl(
                    execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E0,
                    authorized_live_run=AUTHORIZED_LIVE_RUN_VALUE,
                    output_dir=output_dir,
                    episode_id=EPISODE_ID,
                )
                self.assertIsInstance(record, E0MineRLRunRecord)
                self.assertTrue(record.success)
                self.assertTrue(record.authorization_accepted)
                self.assertFalse(record.real_execution_performed)
                self.assertFalse(record.integration_verified)
                self.assertEqual(record.verification_level, UNIT_VERIFIED)
                self.assertFalse(record.cleanup.process_release_proven)
                payload = json.loads(
                    (output_dir / "e0_lifecycle.json").read_text(encoding="utf-8")
                )
                self.assertFalse(payload["integration_verified"])
                self.assertEqual(payload["check_id"], "E0")
                self.assertEqual(payload["name"], "reset_close")
                manifest = p1_validation_manifest()
                self.assertTrue(all(item["status"] == "not_run" for item in manifest))

    def test_record_rejects_integration_claim(self) -> None:
        cleanup = inspect_minerl_cleanup(None, close_returned=True)
        with self.assertRaisesRegex(ValueError, "integration_verified"):
            E0MineRLRunRecord(
                check_id="E0",
                name="reset_close",
                episode_id=EPISODE_ID,
                step_id=0,
                backend_identity="RecordingMineRLBackend",
                opened=True,
                created=True,
                reset_completed=True,
                initial_state_present=True,
                closed=True,
                success=True,
                outcome="lifecycle_ok",
                cleanup=cleanup,
                integration_verified=True,
            )
        with self.assertRaisesRegex(ValueError, "process release"):
            E0CleanupStatus(
                close_returned=True,
                backend_marked_closed=True,
                environment_reference_cleared=True,
                owner_cleared=True,
                process_release_proven=True,
            )


class PublicApiTests(unittest.TestCase):
    def test_cli_check_does_not_run_e0_minerl(self) -> None:
        from obsidianlink.cli import main
        import io
        from contextlib import redirect_stdout

        output = io.StringIO()
        with redirect_stdout(output):
            code = main(["--check"])
        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertFalse(payload["p1_validation"]["real_execution_performed"])
        self.assertFalse(payload["p1_validation"]["integration_verified"])
        self.assertEqual(payload["p1_validation"]["cases"][0]["status"], "not_run")


if __name__ == "__main__":
    unittest.main()

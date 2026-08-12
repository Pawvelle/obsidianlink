"""Offline tests for the authorized C1 live MineRL entry.

These tests must never start real MineRL/Minecraft. They patch the production
env factory and formal runs root so evidence lands in temporary directories.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
from PIL import Image

from obsidianlink.core.task_catalog import load_task_catalog
from obsidianlink.core.types import MacroAction, TaskInstance
from obsidianlink.drivers.casting_c1 import build_casting_action_plan
from obsidianlink.runners import casting_c1_live as live_mod
from obsidianlink.runners.casting_c1_live import (
    AUTHORIZED_LIVE_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_C1,
    REQUIRED_LIVE_EVIDENCE_FILES,
    C1LiveAuthorizationError,
    C1LivePreflightError,
    allocate_live_run_dir,
    collect_runtime_preflight,
    preflight_authorized_c1_live,
    reset_authorized_live_process_guards_for_tests,
    run_casting_c1_authorized_live,
)
from obsidianlink.runners.casting_c1_live_smoke import (
    C1ReactiveStubEnv,
    EVALUATOR_ONLY_TOKENS,
    FROZEN_TARGET_CELL,
    OfflineC1StubEnvFactory,
    build_offline_stub_env_factory,
    load_frozen_c1_task,
    run_casting_c1_live_smoke,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "benchmark/catalog/tasks.json"
TASK_PATH = ROOT / "benchmark/instances/active/casting_c1_fixed.json"


class _CountingStubFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, task: TaskInstance) -> C1ReactiveStubEnv:
        self.calls += 1
        return C1ReactiveStubEnv(task)


class AuthorizedLivePreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_authorized_live_process_guards_for_tests()
        self.tmp = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.tmp.name) / "casting_c1_fixed"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.runs_root / "run-preflight"
        self._patcher = mock.patch.object(
            live_mod, "FORMAL_C1_RUNS_ROOT", self.runs_root.resolve()
        )
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        self.tmp.cleanup()
        reset_authorized_live_process_guards_for_tests()

    def test_import_and_preflight_do_not_start_env(self) -> None:
        factory = _CountingStubFactory()
        with mock.patch.object(
            live_mod.minerl_backend_module,
            "_default_env_factory",
            factory,
        ):
            payload = collect_runtime_preflight(dry_run=True)
            self.assertTrue(payload["dry_run"])
            self.assertEqual(factory.calls, 0)
            preflight_authorized_c1_live(
                output_dir=self.output_dir,
                execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_C1,
                authorized_live_run=AUTHORIZED_LIVE_RUN_VALUE,
            )
            self.assertEqual(factory.calls, 0)

    def test_muted_runtime_copy_preserves_jar_and_disables_sound(self) -> None:
        source = Path(self.tmp.name) / "source-runtime"
        destination = Path(self.tmp.name) / "isolated-runtime"
        jar = source / "build" / "libs" / "mcprec-6.13.jar"
        jar.parent.mkdir(parents=True)
        jar.write_bytes(b"exact-vendored-jar-fixture")
        launch = source / "launchClient.sh"
        launch.write_text("#!/bin/sh\n", encoding="utf-8")

        live_mod._copy_muted_minecraft_runtime(source, destination)

        self.assertEqual(
            (destination / "build" / "libs" / jar.name).read_bytes(),
            jar.read_bytes(),
        )
        options = (destination / "options.txt").read_text(encoding="utf-8")
        self.assertIn("version:2586", options)
        self.assertIn("soundCategory_master:0.0", options)
        self.assertIn("soundCategory_ambient:0.0", options)
    def test_missing_authorization_fails(self) -> None:
        with self.assertRaises(C1LiveAuthorizationError):
            preflight_authorized_c1_live(
                output_dir=self.output_dir,
                execution_mode="offline_stub",
                authorized_live_run=AUTHORIZED_LIVE_RUN_VALUE,
            )
        with self.assertRaises(C1LiveAuthorizationError):
            preflight_authorized_c1_live(
                output_dir=self.output_dir,
                execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_C1,
                authorized_live_run="casting_c3_fixed",
            )

    def test_wrong_task_identity_fails(self) -> None:
        payload = json.loads(TASK_PATH.read_text(encoding="utf-8"))
        payload["workflow"] = "casting_c3_fixed"
        task = TaskInstance.from_dict(payload)
        with self.assertRaises(C1LivePreflightError):
            preflight_authorized_c1_live(
                output_dir=self.output_dir,
                execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_C1,
                authorized_live_run=AUTHORIZED_LIVE_RUN_VALUE,
                task=task,
            )

    def test_c2_to_c5_requests_fail(self) -> None:
        for bad in (
            "casting_c3_fixed",
            "casting_s_c3_fixed",
            "casting_s_c4_fixed",
            "casting_s_c5_fixed",
        ):
            with self.assertRaises(C1LiveAuthorizationError):
                preflight_authorized_c1_live(
                    output_dir=self.output_dir,
                    execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_C1,
                    authorized_live_run=bad,
                )

    def test_injected_factory_or_backend_rejected(self) -> None:
        with self.assertRaises(C1LivePreflightError):
            preflight_authorized_c1_live(
                output_dir=self.output_dir,
                execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_C1,
                authorized_live_run=AUTHORIZED_LIVE_RUN_VALUE,
                env_factory=lambda task: None,
            )
        with self.assertRaises(C1LivePreflightError):
            preflight_authorized_c1_live(
                output_dir=self.output_dir,
                execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_C1,
                authorized_live_run=AUTHORIZED_LIVE_RUN_VALUE,
                backend=object(),
            )

    def test_plan_tamper_rejected(self) -> None:
        plan = build_casting_action_plan()[:-1]
        with self.assertRaises(C1LivePreflightError):
            preflight_authorized_c1_live(
                output_dir=self.output_dir,
                execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_C1,
                authorized_live_run=AUTHORIZED_LIVE_RUN_VALUE,
                plan=plan,
            )

    def test_output_already_exists_rejected(self) -> None:
        self.output_dir.mkdir()
        with self.assertRaises(C1LivePreflightError):
            preflight_authorized_c1_live(
                output_dir=self.output_dir,
                execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_C1,
                authorized_live_run=AUTHORIZED_LIVE_RUN_VALUE,
            )

    def test_output_outside_formal_path_rejected(self) -> None:
        outside = Path(self.tmp.name) / "other" / "run"
        outside.parent.mkdir(parents=True, exist_ok=True)
        with self.assertRaises(C1LivePreflightError):
            preflight_authorized_c1_live(
                output_dir=outside,
                execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_C1,
                authorized_live_run=AUTHORIZED_LIVE_RUN_VALUE,
            )

    def test_gradle_and_model_rejected(self) -> None:
        with self.assertRaises(C1LivePreflightError):
            preflight_authorized_c1_live(
                output_dir=self.output_dir,
                execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_C1,
                authorized_live_run=AUTHORIZED_LIVE_RUN_VALUE,
                allow_gradle=True,
            )
        with self.assertRaises(C1LivePreflightError):
            preflight_authorized_c1_live(
                output_dir=self.output_dir,
                execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_C1,
                authorized_live_run=AUTHORIZED_LIVE_RUN_VALUE,
                request_model=True,
            )


class AuthorizedLiveRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_authorized_live_process_guards_for_tests()
        self.tmp = tempfile.TemporaryDirectory()
        self.runs_root = Path(self.tmp.name) / "casting_c1_fixed"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.output_dir = self.runs_root / "run-live-stub"
        self._root_patcher = mock.patch.object(
            live_mod, "FORMAL_C1_RUNS_ROOT", self.runs_root.resolve()
        )
        self._root_patcher.start()
        self.factory = _CountingStubFactory()
        self._factory_patcher = mock.patch.object(
            live_mod.minerl_backend_module,
            "_default_env_factory",
            self.factory,
        )
        self._factory_patcher.start()

    def tearDown(self) -> None:
        self._factory_patcher.stop()
        self._root_patcher.stop()
        self.tmp.cleanup()
        reset_authorized_live_process_guards_for_tests()

    def test_positive_path_with_stubbed_production_factory(self) -> None:
        result = run_casting_c1_authorized_live(
            output_dir=self.output_dir,
            execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_C1,
            authorized_live_run=AUTHORIZED_LIVE_RUN_VALUE,
            wall_clock_seconds=120,
        )
        self.assertEqual(self.factory.calls, 1)
        self.assertEqual(result.real_env_factory_calls, 1)
        self.assertEqual(result.real_episode_count, 1)
        self.assertTrue(result.driver_completed)
        self.assertTrue(result.evaluator_success)
        self.assertTrue(result.evidence_complete)
        self.assertEqual(result.close_status, "closed")
        self.assertTrue(result.overall_success)
        for filename in REQUIRED_LIVE_EVIDENCE_FILES:
            path = self.output_dir / filename
            self.assertTrue(path.is_file(), msg=filename)
            self.assertGreater(path.stat().st_size, 0, msg=filename)
        auth = json.loads((self.output_dir / "authorization.json").read_text())
        self.assertFalse(auth["scope"]["gradle_authorized"])
        self.assertTrue(auth["catalog_live_run_allowed_remains_false"])
        with Image.open(self.output_dir / "initial.png") as image:
            self.assertEqual(image.size, (640, 360))
        events = [
            json.loads(line)
            for line in (self.output_dir / "events.jsonl").read_text().splitlines()
            if line.strip()
        ]
        for event in events:
            payload = json.dumps(event)
            for token in EVALUATOR_ONLY_TOKENS:
                self.assertNotIn(token, payload)
        catalog = load_task_catalog(CATALOG_PATH)
        entry = catalog.entry_for_compatibility_id("casting_c1_fixed")
        self.assertFalse(entry.live_run_allowed)

    def test_max_reset_attempts_is_one(self) -> None:
        created: list[Any] = []

        class BoomFactory:
            def __call__(self, task: TaskInstance) -> Any:
                created.append(task)
                raise RuntimeError("boom")

        self._factory_patcher.stop()
        boom = BoomFactory()
        self._factory_patcher = mock.patch.object(
            live_mod.minerl_backend_module,
            "_default_env_factory",
            boom,
        )
        self._factory_patcher.start()
        result = run_casting_c1_authorized_live(
            output_dir=self.output_dir,
            execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_C1,
            authorized_live_run=AUTHORIZED_LIVE_RUN_VALUE,
            wall_clock_seconds=120,
        )
        self.assertEqual(len(created), 1)
        self.assertEqual(result.real_env_factory_calls, 1)
        self.assertFalse(result.overall_success)
        self.assertIsNotNone(result.failure_reason)

    def test_second_process_run_rejected(self) -> None:
        first = run_casting_c1_authorized_live(
            output_dir=self.output_dir,
            execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_C1,
            authorized_live_run=AUTHORIZED_LIVE_RUN_VALUE,
            wall_clock_seconds=120,
        )
        self.assertTrue(first.evidence_complete)
        with self.assertRaises(C1LivePreflightError):
            run_casting_c1_authorized_live(
                output_dir=self.runs_root / "run-live-stub-2",
                execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_C1,
                authorized_live_run=AUTHORIZED_LIVE_RUN_VALUE,
                wall_clock_seconds=120,
            )

    def test_driver_completed_evaluator_failure_is_not_success(self) -> None:
        class NoObsidianFactory:
            def __call__(self, task: TaskInstance) -> C1ReactiveStubEnv:
                return C1ReactiveStubEnv(task, produce_obsidian=False)

        self._factory_patcher.stop()
        self._factory_patcher = mock.patch.object(
            live_mod.minerl_backend_module,
            "_default_env_factory",
            NoObsidianFactory(),
        )
        self._factory_patcher.start()
        result = run_casting_c1_authorized_live(
            output_dir=self.output_dir,
            execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_C1,
            authorized_live_run=AUTHORIZED_LIVE_RUN_VALUE,
            wall_clock_seconds=120,
        )
        self.assertTrue(result.driver_completed)
        self.assertFalse(result.evaluator_success)
        self.assertFalse(result.overall_success)

    def test_allocate_live_run_dir_under_formal_root(self) -> None:
        path = allocate_live_run_dir()
        self.assertEqual(path.parent, self.runs_root.resolve())
        self.assertFalse(path.exists())


class OfflineRunnerSafetyRegressionTests(unittest.TestCase):
    def test_offline_runner_still_rejects_live_and_arbitrary_factory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "offline"
            with self.assertRaises(Exception):
                run_casting_c1_live_smoke(
                    output_dir=output_dir,
                    env_factory=build_offline_stub_env_factory(),
                    request_live=True,
                )
            with self.assertRaises(Exception):
                run_casting_c1_live_smoke(
                    output_dir=output_dir,
                    env_factory=lambda task: C1ReactiveStubEnv(task),  # type: ignore[arg-type]
                )
            # Controlled offline factory still works and stays outside runs/.
            result = run_casting_c1_live_smoke(
                output_dir=output_dir,
                env_factory=build_offline_stub_env_factory(),
            )
            self.assertTrue(result.overall_success)
            formal_runs = (ROOT / "runs").resolve()
            self.assertFalse(formal_runs in output_dir.resolve().parents)

    def test_catalog_unchanged_after_live_module_import(self) -> None:
        catalog = load_task_catalog(CATALOG_PATH)
        self.assertEqual(
            catalog.active_phase, "P1-REAL-MINERL-ENVIRONMENT-VALIDATION"
        )
        self.assertIsNone(catalog.active_benchmark_task_id)
        self.assertFalse(
            catalog.entry_for_compatibility_id("casting_c1_fixed").live_run_allowed
        )
        self.assertEqual(
            catalog.entry_for_compatibility_id("casting_s_c5_fixed").implementation_status,
            "legacy_regression",
        )
        self.assertEqual(
            catalog.entry_for_compatibility_id("casting_s_c5_fixed").kind,
            "legacy",
        )
        self.assertFalse(
            catalog.entry_for_compatibility_id("casting_s_c5_fixed").live_run_allowed
        )


class CliHelpTests(unittest.TestCase):
    def test_cli_help_does_not_start_env(self) -> None:
        import scripts.run_c1_live as cli

        factory = _CountingStubFactory()
        with mock.patch.object(
            live_mod.minerl_backend_module,
            "_default_env_factory",
            factory,
        ):
            parser = cli.build_parser()
            with self.assertRaises(SystemExit) as ctx:
                parser.parse_args(["--help"])
            self.assertEqual(ctx.exception.code, 0)
            self.assertEqual(factory.calls, 0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from obsidianlink.env.integration.e0_cleanup import inspect_os_process_release
from obsidianlink.env.integration.e11_run import DEPLOYED_E11_FIXTURE_JAR_SHA256
from obsidianlink.env.integration.e12_run import DEPLOYED_E12_FIXTURE_JAR_SHA256
from obsidianlink.env.integration.p1_suite import (
    AUTHORIZED_LIVE_P1_SUITE_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_P1_SUITE,
    P1CaseSummary,
    VERDICT_VALIDATION_FAILED,
    check_p1_suite,
    p1_suite_steps,
    reset_authorized_p1_suite_process_guards_for_tests,
    run_authorized_p1_suite,
)
from obsidianlink.env.integration.p1_suite_runtime import (
    CANONICAL_JAR_SHA256,
    E11_COMPLETION_BARRIER_JAR_SHA256,
    E12_PORTAL_FIXTURE_JAR_SHA256,
    P1SuiteRuntimeError,
    RUNTIME_CANONICAL,
    RUNTIME_E11_COMPLETION_BARRIER,
    RUNTIME_E12_PORTAL_FIXTURE,
    activate_required_runtime,
    find_verified_jar,
    required_runtime,
)
from obsidianlink.env.validation import p1_validation_manifest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "obsidianlink" / "env" / "integration" / "p1_suite_runtime.py"
CANONICAL_BUILD = ROOT / "scripts" / "build_p1_e11_completion_barrier_runtime.py"
SUITE_SOURCE = ROOT / "obsidianlink" / "env" / "integration" / "p1_suite.py"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_jar(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256(payload)


class RequiredRuntimeMappingTests(unittest.TestCase):
    def test_hashes_lock_to_existing_records(self) -> None:
        self.assertEqual(CANONICAL_JAR_SHA256, "684c20ec533897b44e9f2f73340f66ab41a6f61e7c9ae7e0f1db6fae7430751e")
        self.assertIn(CANONICAL_JAR_SHA256, CANONICAL_BUILD.read_text(encoding="utf-8"))
        self.assertEqual(E11_COMPLETION_BARRIER_JAR_SHA256, DEPLOYED_E11_FIXTURE_JAR_SHA256)
        self.assertEqual(E12_PORTAL_FIXTURE_JAR_SHA256, DEPLOYED_E12_FIXTURE_JAR_SHA256)

    def test_e0_through_e10_select_canonical(self) -> None:
        for index in range(11):
            self.assertEqual(
                required_runtime(f"E{index}"),
                (RUNTIME_CANONICAL, CANONICAL_JAR_SHA256),
            )

    def test_e11_selects_completion_barrier(self) -> None:
        self.assertEqual(
            required_runtime("E11"),
            (RUNTIME_E11_COMPLETION_BARRIER, E11_COMPLETION_BARRIER_JAR_SHA256),
        )

    def test_e12_selects_portal_fixture(self) -> None:
        self.assertEqual(
            required_runtime("E12"),
            (RUNTIME_E12_PORTAL_FIXTURE, E12_PORTAL_FIXTURE_JAR_SHA256),
        )

    def test_unknown_check_id_fails_closed(self) -> None:
        with self.assertRaises(P1SuiteRuntimeError):
            required_runtime("E13")

    def test_check_payload_lists_required_runtimes_without_hashing(self) -> None:
        with patch(
            "obsidianlink.env.integration.p1_suite.activate_required_runtime"
        ) as activate, patch(
            "obsidianlink.env.integration.p1_suite_runtime.deployed_mcp_libs"
        ) as libs, patch(
            "obsidianlink.env.integration.p1_suite_runtime._sha256_file"
        ) as hashed:
            payload = check_p1_suite()
        activate.assert_not_called()
        libs.assert_not_called()
        hashed.assert_not_called()
        steps = {item["check_id"]: item for item in payload["steps"]}
        self.assertEqual(steps["E0"]["required_runtime"], RUNTIME_CANONICAL)
        self.assertEqual(steps["E0"]["required_runtime_sha256"], CANONICAL_JAR_SHA256)
        self.assertEqual(steps["E10"]["required_runtime"], RUNTIME_CANONICAL)
        self.assertEqual(steps["E11"]["required_runtime"], RUNTIME_E11_COMPLETION_BARRIER)
        self.assertEqual(steps["E11"]["required_runtime_sha256"], E11_COMPLETION_BARRIER_JAR_SHA256)
        self.assertEqual(steps["E12"]["required_runtime"], RUNTIME_E12_PORTAL_FIXTURE)
        self.assertEqual(steps["E12"]["required_runtime_sha256"], E12_PORTAL_FIXTURE_JAR_SHA256)
        for item in payload["steps"]:
            if item["check_id"] in {"E11", "E12"}:
                continue
            self.assertEqual(item["required_runtime"], RUNTIME_CANONICAL)
            self.assertEqual(item["required_runtime_sha256"], CANONICAL_JAR_SHA256)
        self.assertTrue(all(item["status"] == "not_run" for item in p1_validation_manifest()))

    def test_helper_never_invokes_gradle_or_minerl(self) -> None:
        source = HELPER.read_text(encoding="utf-8")
        self.assertNotIn("gradlew", source.lower())
        self.assertNotIn("shadowjar", source.lower())
        self.assertNotIn("subprocess", source)
        self.assertNotIn("import minerl", source)
        self.assertNotIn("gym.make", source)
        suite = SUITE_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("allow_gradle=True", suite)


class AlreadyBuiltJarActivationTests(unittest.TestCase):
    def test_filename_prefix_is_not_enough_without_sha(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            libs = Path(directory)
            expected = "a" * 64
            _write_jar(libs / "backups" / f"mcprec-6.13.jar.pre-{expected[:8]}", b"wrong-bytes")
            with self.assertRaises(P1SuiteRuntimeError):
                find_verified_jar(libs, expected)

    def test_missing_jar_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(P1SuiteRuntimeError):
                activate_required_runtime("E0", mcp_libs=Path(directory))

    def test_already_active_verified_jar_is_not_copied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            libs = Path(directory)
            payload = b"canonical-already-active"
            sha = _write_jar(libs / "mcprec-6.13.jar", payload)
            with patch(
                "obsidianlink.env.integration.p1_suite_runtime.required_runtime",
                return_value=(RUNTIME_CANONICAL, sha),
            ), patch("shutil.copy2") as copied:
                record = activate_required_runtime("E0", mcp_libs=libs)
            copied.assert_not_called()
            self.assertTrue(record["verified"])
            self.assertTrue(record["already_active"])
            self.assertFalse(record["gradle_invoked"])
            self.assertEqual((libs / "mcprec-6.13.jar").read_bytes(), payload)

    def test_switches_among_existing_jars_and_backs_up_active(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            libs = Path(directory)
            backups = libs / "backups"
            canonical = b"canonical-runtime"
            barrier = b"e11-completion-barrier"
            fixture = b"e12-portal-fixture"
            canonical_sha = _sha256(canonical)
            barrier_sha = _sha256(barrier)
            fixture_sha = _sha256(fixture)
            _write_jar(backups / f"mcprec-6.13.jar.pre-e11-{canonical_sha[:8]}", canonical)
            _write_jar(backups / f"mcprec-6.13.jar.pre-e12-{barrier_sha[:8]}", barrier)
            _write_jar(libs / "mcprec-6.13.jar", fixture)
            mapping = {
                "E0": (RUNTIME_CANONICAL, canonical_sha),
                "E11": (RUNTIME_E11_COMPLETION_BARRIER, barrier_sha),
                "E12": (RUNTIME_E12_PORTAL_FIXTURE, fixture_sha),
            }

            def required(check_id: str) -> tuple[str, str]:
                return mapping[check_id]

            with patch(
                "obsidianlink.env.integration.p1_suite_runtime.required_runtime",
                side_effect=required,
            ), patch("subprocess.Popen") as popen, patch("subprocess.run") as run:
                first = activate_required_runtime("E0", mcp_libs=libs)
                self.assertEqual((libs / "mcprec-6.13.jar").read_bytes(), canonical)
                self.assertFalse(first["already_active"])
                self.assertTrue(
                    (backups / f"mcprec-6.13.jar.p1-suite-{fixture_sha[:8]}").is_file()
                )
                second = activate_required_runtime("E11", mcp_libs=libs)
                self.assertEqual((libs / "mcprec-6.13.jar").read_bytes(), barrier)
                self.assertEqual(second["runtime"], RUNTIME_E11_COMPLETION_BARRIER)
                third = activate_required_runtime("E12", mcp_libs=libs)
                self.assertEqual((libs / "mcprec-6.13.jar").read_bytes(), fixture)
                self.assertEqual(third["sha256"], fixture_sha)
                self.assertTrue(third["verified"])
            popen.assert_not_called()
            run.assert_not_called()

    def test_mismatch_after_copy_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            libs = Path(directory)
            sha = _write_jar(libs / "backups" / "mcprec-6.13.jar.pre-deadbeef", b"source")
            _write_jar(libs / "mcprec-6.13.jar", b"other")

            def corrupt_replace(self: Path, target: Path) -> Path:
                Path(target).write_bytes(b"corrupted")
                return Path(target)

            with patch(
                "obsidianlink.env.integration.p1_suite_runtime.required_runtime",
                return_value=(RUNTIME_CANONICAL, sha),
            ), patch.object(Path, "replace", corrupt_replace):
                with self.assertRaises(P1SuiteRuntimeError):
                    activate_required_runtime("E0", mcp_libs=libs)


class SuiteStopsBeforeLaunchTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_authorized_p1_suite_process_guards_for_tests()

    def tearDown(self) -> None:
        reset_authorized_p1_suite_process_guards_for_tests()

    def test_runtime_mismatch_does_not_launch_the_case(self) -> None:
        called: list[str] = []

        def execute(step, output_dir, episode_id):
            called.append(step.step_key)
            raise AssertionError("case runner must not start after runtime failure")

        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory) / "p1_validation_suite"
            runs_root.mkdir()
            output_dir = runs_root / "pilot-runtime-mismatch"
            with patch(
                "obsidianlink.env.integration.p1_suite.FORMAL_SUITE_RUNS_ROOT",
                runs_root.resolve(),
            ), patch(
                "obsidianlink.env.integration.p1_suite.activate_required_runtime",
                side_effect=P1SuiteRuntimeError("required already-built JAR missing"),
            ):
                result = run_authorized_p1_suite(
                    execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_P1_SUITE,
                    authorized_live_run=AUTHORIZED_LIVE_P1_SUITE_RUN_VALUE,
                    output_dir=output_dir,
                    execute_step=execute,
                )
        self.assertEqual(called, [])
        self.assertEqual(result.verdict, VERDICT_VALIDATION_FAILED)
        self.assertEqual(result.stopped_after, "E0")
        self.assertFalse(result.cases[0].success)
        self.assertEqual(result.cases[0].outcome, "runtime_not_verified")
        self.assertFalse(result.cases[0].runtime["verified"])
        self.assertFalse(result.p1_hard_gate_passed)
        self.assertFalse(result.integration_verified)

    def test_success_without_verified_runtime_is_rejected(self) -> None:
        step = p1_suite_steps()[0]
        with self.assertRaises(ValueError):
            P1CaseSummary(
                check_id=step.check_id,
                name=step.name,
                variant=step.variant,
                success=True,
                outcome="ok",
                requires_server_truth=step.requires_server_truth,
                truth_missing_count=None,
                cleanup_failed=False,
                process_release=inspect_os_process_release(subprocess_exited=True),
                real_execution_performed=True,
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from obsidianlink.env.integration.e0_cleanup import (
    PROCESS_RELEASE_NOT_OBSERVED,
    PROCESS_RELEASE_RESIDUAL,
    PROCESS_RELEASE_SUBPROCESS_ALIVE,
    descendant_pids,
    inspect_os_process_release,
    is_minerl_runtime_command,
    merge_tracked_descendants,
    residual_descendants,
    snapshot_process_table,
)
from obsidianlink.env.integration.e0_run import (
    AUTHORIZED_LIVE_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E0,
)
from obsidianlink.env.integration.e7_run import (
    AUTHORIZED_LIVE_E7_LAVA_RUN_VALUE,
    AUTHORIZED_LIVE_E7_WATER_RUN_VALUE,
)
from obsidianlink.env.integration.e9_run import (
    AUTHORIZED_LIVE_E9_LAVA_RUN_VALUE,
    AUTHORIZED_LIVE_E9_WATER_RUN_VALUE,
)
from obsidianlink.env.integration.e12_run import (
    AUTHORIZED_LIVE_E12_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_E12,
)
from obsidianlink.env.integration.p1_suite import (
    AUTHORIZED_LIVE_P1_SUITE_RUN_VALUE,
    EXECUTION_MODE_AUTHORIZED_LIVE_P1_SUITE,
    P1CaseSummary,
    P1SuiteAuthorizationError,
    VERDICT_CLEANUP_FAILED,
    VERDICT_HARD_GATE_SUCCESS,
    VERDICT_PROCESS_RELEASE_NOT_PROVEN,
    VERDICT_TRUTH_MISSING,
    VERDICT_VALIDATION_FAILED,
    aggregate_p1_suite,
    check_p1_suite,
    main,
    p1_suite_steps,
    preflight_authorized_p1_suite,
    requires_counted_server_truth,
    reset_authorized_p1_suite_process_guards_for_tests,
    run_authorized_p1_suite,
)
from obsidianlink.env.integration.p1_suite_runtime import required_runtime
from obsidianlink.env.validation import P1_VALIDATION_CASES, p1_validation_manifest


def _release(*, proven: bool) -> object:
    if proven:
        return inspect_os_process_release(
            tracked_children=[{"pid": 4242, "command": "/usr/bin/java -jar mcprec.jar"}],
            residual_children=(),
            subprocess_exited=True,
        )
    return inspect_os_process_release(subprocess_exited=True)


def _runtime(check_id: str) -> dict:
    name, sha256 = required_runtime(check_id)
    return {
        "already_active": True,
        "check_id": check_id,
        "gradle_invoked": False,
        "jar_path": "/tmp/fake-mcprec-6.13.jar",
        "runtime": name,
        "sha256": sha256,
        "source_path": "/tmp/fake-mcprec-6.13.jar",
        "verified": True,
    }


def _case(
    step,
    *,
    success: bool = True,
    outcome: str = "ok",
    truth_missing_count: int | None = 0,
    cleanup_failed: bool = False,
    proven: bool = True,
    real: bool = True,
) -> P1CaseSummary:
    missing = (
        truth_missing_count if requires_counted_server_truth(step.check_id) else None
    )
    return P1CaseSummary(
        check_id=step.check_id,
        name=step.name,
        variant=step.variant,
        success=success,
        outcome=outcome,
        requires_server_truth=step.requires_server_truth,
        truth_missing_count=missing,
        cleanup_failed=cleanup_failed,
        process_release=_release(proven=proven),
        real_execution_performed=real,
        runtime=_runtime(step.check_id),
    )


def _all_ok(**kwargs) -> list[P1CaseSummary]:
    return [_case(step, **kwargs) for step in p1_suite_steps()]


class ProcessReleaseInspectionTests(unittest.TestCase):
    def test_close_returning_is_not_process_release(self) -> None:
        status = inspect_os_process_release(subprocess_exited=True)
        self.assertFalse(status.process_release_proven)
        self.assertEqual(status.limitation, PROCESS_RELEASE_NOT_OBSERVED)

    def test_observed_java_child_gone_is_proven(self) -> None:
        status = inspect_os_process_release(
            tracked_children=[{"pid": 9, "command": "/opt/java/bin/java -Xmx2G"}],
            residual_children=(),
            subprocess_exited=True,
        )
        self.assertTrue(status.process_release_proven)
        self.assertTrue(status.minerl_runtime_observed)

    def test_residual_java_child_is_not_proven(self) -> None:
        child = {"pid": 9, "command": "/usr/bin/java -jar minecraft.jar"}
        status = inspect_os_process_release(
            tracked_children=[child],
            residual_children=[child],
            subprocess_exited=True,
        )
        self.assertFalse(status.process_release_proven)

    def test_python_descendant_without_java_is_not_proven(self) -> None:
        status = inspect_os_process_release(
            tracked_children=[{"pid": 8, "command": sys.executable + " -m helper"}],
            residual_children=(),
            subprocess_exited=True,
        )
        self.assertFalse(status.process_release_proven)

    def test_process_table_drops_a_terminated_pid(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(0.4)"]
        )
        self.assertIn(process.pid, snapshot_process_table())
        self.assertEqual(process.wait(timeout=2), 0)
        self.assertNotIn(process.pid, snapshot_process_table())

    def test_descendant_walk_follows_pid_tree(self) -> None:
        table = {
            1: (0, "init"),
            10: (1, "python"),
            11: (10, "/usr/bin/java -jar mcprec.jar"),
            12: (11, "watcher"),
            99: (1, "unrelated"),
        }
        self.assertEqual(descendant_pids(10, table), {11, 12})

    def test_retained_java_identity_survives_degraded_command(self) -> None:
        tracked: dict[int, str] = {}
        merge_tracked_descendants(
            tracked,
            {2785: "java -Xmx4G -XstartOnFirstThread -jar mcprec-6.13.jar --envPort=9594"},
        )
        merge_tracked_descendants(tracked, {2785: "(java)"})
        self.assertTrue(is_minerl_runtime_command(tracked[2785]))
        self.assertNotEqual(tracked[2785], "(java)")
        status = inspect_os_process_release(
            tracked_children=[{"pid": 2785, "command": tracked[2785]}],
            residual_children=(),
            subprocess_exited=True,
        )
        self.assertTrue(status.minerl_runtime_observed)
        self.assertTrue(status.process_release_proven)

    def test_parenthesized_java_alone_is_not_runtime_identity(self) -> None:
        self.assertFalse(is_minerl_runtime_command("(java)"))
        tracked: dict[int, str] = {}
        merge_tracked_descendants(tracked, {2785: "(java)"})
        status = inspect_os_process_release(
            tracked_children=[{"pid": 2785, "command": tracked[2785]}],
            residual_children=(),
            subprocess_exited=True,
        )
        self.assertFalse(status.minerl_runtime_observed)
        self.assertFalse(status.process_release_proven)
        self.assertEqual(status.limitation, PROCESS_RELEASE_NOT_OBSERVED)

    def test_observed_java_with_residual_pid_is_not_proven(self) -> None:
        child = {"pid": 9, "command": "java -Xmx4G -jar mcprec-6.13.jar"}
        residual = residual_descendants(
            {9: child["command"]},
            table={9: (1, "(java)")},
        )
        self.assertEqual(residual, {9: "(java)"})
        status = inspect_os_process_release(
            tracked_children=[child],
            residual_children=[{"pid": 9, "command": residual[9]}],
            subprocess_exited=True,
        )
        self.assertTrue(status.minerl_runtime_observed)
        self.assertFalse(status.process_release_proven)
        self.assertEqual(status.limitation, PROCESS_RELEASE_RESIDUAL)

    def test_live_subprocess_is_not_proven(self) -> None:
        status = inspect_os_process_release(
            tracked_children=[
                {"pid": 9, "command": "java -Xmx4G -jar mcprec-6.13.jar"}
            ],
            residual_children=(),
            subprocess_exited=False,
        )
        self.assertFalse(status.process_release_proven)
        self.assertEqual(status.limitation, PROCESS_RELEASE_SUBPROCESS_ALIVE)

    def test_weaker_command_can_upgrade_to_java_identity(self) -> None:
        tracked: dict[int, str] = {}
        merge_tracked_descendants(tracked, {7: "(java)"})
        merge_tracked_descendants(tracked, {7: "/usr/bin/java -jar mcprec.jar"})
        self.assertEqual(tracked[7], "/usr/bin/java -jar mcprec.jar")

    def test_residual_uses_pid_presence_not_command_text(self) -> None:
        tracked = {9: "java -Xmx4G -jar mcprec-6.13.jar"}
        self.assertEqual(residual_descendants(tracked, table={}), {})
        self.assertEqual(
            residual_descendants(tracked, table={9: (1, "(java)")}),
            {9: "(java)"},
        )


class SuiteOrderingTests(unittest.TestCase):
    def test_steps_follow_p1_cases_and_required_variants(self) -> None:
        steps = p1_suite_steps()
        self.assertEqual(
            [case.check_id.value for case in P1_VALIDATION_CASES],
            [f"E{index}" for index in range(13)],
        )
        self.assertEqual(steps[0].check_id, "E0")
        self.assertEqual(steps[-1].check_id, "E12")
        self.assertEqual([step.check_id for step in steps if step.check_id == "E7"], ["E7", "E7"])
        self.assertEqual([step.variant for step in steps if step.check_id == "E7"], ["water", "lava"])
        self.assertEqual([step.variant for step in steps if step.check_id == "E9"], ["water", "lava"])
        self.assertEqual(
            [step.check_id for step in steps],
            ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E7", "E8", "E9", "E9", "E10", "E11", "E12"],
        )

    def test_steps_reuse_existing_runner_tokens(self) -> None:
        steps = {step.step_key: step for step in p1_suite_steps()}
        self.assertEqual(steps["E0"].execution_mode, EXECUTION_MODE_AUTHORIZED_LIVE_E0)
        self.assertEqual(steps["E0"].authorized_live_run, AUTHORIZED_LIVE_RUN_VALUE)
        self.assertEqual(steps["E0"].module, "obsidianlink.env.integration.e0_run")
        self.assertEqual(steps["E7:water"].authorized_live_run, AUTHORIZED_LIVE_E7_WATER_RUN_VALUE)
        self.assertEqual(steps["E7:lava"].authorized_live_run, AUTHORIZED_LIVE_E7_LAVA_RUN_VALUE)
        self.assertEqual(steps["E9:water"].authorized_live_run, AUTHORIZED_LIVE_E9_WATER_RUN_VALUE)
        self.assertEqual(steps["E9:lava"].authorized_live_run, AUTHORIZED_LIVE_E9_LAVA_RUN_VALUE)
        self.assertEqual(steps["E12"].execution_mode, EXECUTION_MODE_AUTHORIZED_LIVE_E12)
        self.assertEqual(steps["E12"].authorized_live_run, AUTHORIZED_LIVE_E12_RUN_VALUE)


class AggregateVerdictTests(unittest.TestCase):
    def test_hard_gate_requires_real_complete_proven_suite(self) -> None:
        result = aggregate_p1_suite(_all_ok(), real_execution_performed=True)
        self.assertEqual(result.verdict, VERDICT_HARD_GATE_SUCCESS)
        self.assertTrue(result.p1_hard_gate_passed)
        self.assertFalse(result.integration_verified)
        self.assertTrue(all(item["status"] == "not_run" for item in p1_validation_manifest()))

    def test_offline_success_does_not_pass_hard_gate(self) -> None:
        result = aggregate_p1_suite(_all_ok(real=False), real_execution_performed=False)
        self.assertEqual(result.verdict, VERDICT_PROCESS_RELEASE_NOT_PROVEN)
        self.assertFalse(result.p1_hard_gate_passed)
        self.assertFalse(result.integration_verified)

    def test_validation_failure_propagates_and_blocks_later_reasons(self) -> None:
        cases = _all_ok()
        cases[4] = _case(p1_suite_steps()[4], success=False, outcome="camera_no_change")
        result = aggregate_p1_suite(cases, real_execution_performed=True)
        self.assertEqual(result.verdict, VERDICT_VALIDATION_FAILED)
        self.assertFalse(result.p1_hard_gate_passed)

    def test_truth_missing_is_distinct_from_validation_failure(self) -> None:
        cases = _all_ok()
        e8 = next(step for step in p1_suite_steps() if step.check_id == "E8")
        index = [step.step_key for step in p1_suite_steps()].index("E8")
        cases[index] = _case(
            e8,
            success=False,
            outcome="truth_snapshot_missing",
            truth_missing_count=2,
        )
        result = aggregate_p1_suite(cases, real_execution_performed=True)
        self.assertEqual(result.verdict, VERDICT_TRUTH_MISSING)
        self.assertTrue(result.truth_missing)
        self.assertFalse(result.p1_hard_gate_passed)

    def test_cleanup_failure_is_distinct(self) -> None:
        cases = _all_ok()
        cases[0] = _case(
            p1_suite_steps()[0],
            success=False,
            outcome="cleanup_failed",
            cleanup_failed=True,
        )
        result = aggregate_p1_suite(cases, real_execution_performed=True)
        self.assertEqual(result.verdict, VERDICT_CLEANUP_FAILED)
        self.assertTrue(result.cleanup_failed)

    def test_process_release_not_proven_when_jvm_was_not_observed(self) -> None:
        result = aggregate_p1_suite(_all_ok(proven=False), real_execution_performed=True)
        self.assertEqual(result.verdict, VERDICT_PROCESS_RELEASE_NOT_PROVEN)
        self.assertFalse(result.process_release_proven)
        self.assertFalse(result.p1_hard_gate_passed)

    def test_incomplete_suite_fails_closed(self) -> None:
        result = aggregate_p1_suite(_all_ok()[:3], real_execution_performed=True)
        self.assertFalse(result.all_required_cases_present)
        self.assertEqual(result.verdict, VERDICT_VALIDATION_FAILED)
        self.assertFalse(result.p1_hard_gate_passed)

    def test_first_truth_missing_wins_even_if_later_cases_are_absent(self) -> None:
        e8 = next(step for step in p1_suite_steps() if step.check_id == "E8")
        prefix = [
            _case(step)
            for step in p1_suite_steps()
            if step.check_id in {"E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"}
        ]
        prefix.append(
            _case(e8, success=False, outcome="truth_snapshot_missing", truth_missing_count=1)
        )
        result = aggregate_p1_suite(prefix, real_execution_performed=True)
        self.assertEqual(result.verdict, VERDICT_TRUTH_MISSING)

    def test_e4_through_e7_success_does_not_require_truth_missing_count(self) -> None:
        for check_id in ("E4", "E5", "E6", "E7"):
            self.assertFalse(requires_counted_server_truth(check_id))
            self.assertTrue(requires_counted_server_truth("E8"))
        cases = _all_ok()
        typed = [case for case in cases if case.check_id in {"E4", "E5", "E6", "E7"}]
        self.assertTrue(typed)
        self.assertTrue(all(case.truth_missing_count is None for case in typed))
        result = aggregate_p1_suite(cases, real_execution_performed=True)
        self.assertEqual(result.verdict, VERDICT_HARD_GATE_SUCCESS)
        self.assertFalse(result.truth_missing)

    def test_e4_camera_ok_without_count_is_not_truth_missing(self) -> None:
        cases = _all_ok()
        e4 = next(step for step in p1_suite_steps() if step.check_id == "E4")
        cases[4] = _case(e4, outcome="camera_ok")
        self.assertIsNone(cases[4].truth_missing_count)
        result = aggregate_p1_suite(cases, real_execution_performed=True)
        self.assertEqual(result.verdict, VERDICT_HARD_GATE_SUCCESS)
        self.assertFalse(result.truth_missing)

    def test_e4_orientation_missing_is_validation_failure(self) -> None:
        cases = _all_ok()
        e4 = next(step for step in p1_suite_steps() if step.check_id == "E4")
        cases[4] = _case(e4, success=False, outcome="orientation_before_missing")
        result = aggregate_p1_suite(cases, real_execution_performed=True)
        self.assertEqual(result.verdict, VERDICT_VALIDATION_FAILED)
        self.assertFalse(result.truth_missing)

    def test_e5_e7_truth_absence_outcomes_fail_closed(self) -> None:
        failures = (
            ("E5", "position_before_missing"),
            ("E6", "block_before_missing"),
            ("E7", "fluid_before_missing"),
        )
        for check_id, outcome in failures:
            cases = _all_ok()
            step = next(step for step in p1_suite_steps() if step.check_id == check_id)
            index = next(
                i for i, item in enumerate(p1_suite_steps()) if item.check_id == check_id
            )
            cases[index] = _case(step, success=False, outcome=outcome)
            result = aggregate_p1_suite(cases, real_execution_performed=True)
            self.assertEqual(result.verdict, VERDICT_VALIDATION_FAILED, check_id)
            self.assertFalse(result.truth_missing, check_id)

    def test_e8_success_with_nonzero_count_is_truth_missing(self) -> None:
        cases = _all_ok()
        e8 = next(step for step in p1_suite_steps() if step.check_id == "E8")
        index = [step.step_key for step in p1_suite_steps()].index("E8")
        cases[index] = _case(
            e8, success=True, outcome="block_truth_ok", truth_missing_count=1
        )
        result = aggregate_p1_suite(cases, real_execution_performed=True)
        self.assertEqual(result.verdict, VERDICT_TRUTH_MISSING)
        self.assertTrue(result.truth_missing)

    def test_e8_success_with_missing_count_is_truth_missing(self) -> None:
        cases = _all_ok()
        e8 = next(step for step in p1_suite_steps() if step.check_id == "E8")
        index = [step.step_key for step in p1_suite_steps()].index("E8")
        cases[index] = _case(
            e8, success=True, outcome="block_truth_ok", truth_missing_count=None
        )
        result = aggregate_p1_suite(cases, real_execution_performed=True)
        self.assertEqual(result.verdict, VERDICT_TRUTH_MISSING)
        self.assertTrue(result.truth_missing)

    def test_e12_zero_count_does_not_block_hard_gate(self) -> None:
        cases = _all_ok()
        e12 = next(step for step in p1_suite_steps() if step.check_id == "E12")
        self.assertEqual(cases[-1].check_id, "E12")
        self.assertEqual(cases[-1].truth_missing_count, 0)
        cases[-1] = _case(e12, outcome="dimension_transition_ok", truth_missing_count=0)
        result = aggregate_p1_suite(cases, real_execution_performed=True)
        self.assertEqual(result.verdict, VERDICT_HARD_GATE_SUCCESS)


class SuiteEntrypointTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_authorized_p1_suite_process_guards_for_tests()
        patcher = patch(
            "obsidianlink.env.integration.p1_suite.activate_required_runtime",
            side_effect=lambda check_id, **kwargs: _runtime(check_id),
        )
        self.activate_runtime = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        reset_authorized_p1_suite_process_guards_for_tests()

    def test_check_is_offline_and_does_not_promote(self) -> None:
        payload = check_p1_suite()
        self.assertEqual(payload["status"], "ok")
        self.assertFalse(payload["integration_verified"])
        self.assertFalse(payload["p1_hard_gate_passed"])
        self.assertFalse(payload["real_execution_performed"])
        self.assertFalse(payload["process_release_proven"])
        self.assertEqual(payload["p1_validation_manifest_status"], "not_run")
        self.assertTrue(all(item["status"] == "not_run" for item in p1_validation_manifest()))

    def test_preflight_does_not_launch_case_runners(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory) / "p1_validation_suite"
            runs_root.mkdir()
            output_dir = runs_root / "pilot-preflight"
            with patch(
                "obsidianlink.env.integration.p1_suite.FORMAL_SUITE_RUNS_ROOT",
                runs_root.resolve(),
            ), patch(
                "obsidianlink.env.integration.p1_suite.execute_case_subprocess"
            ) as live:
                payload = run_authorized_p1_suite(
                    execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_P1_SUITE,
                    authorized_live_run=AUTHORIZED_LIVE_P1_SUITE_RUN_VALUE,
                    output_dir=output_dir,
                    preflight_only=True,
                )
            self.assertIsInstance(payload, dict)
            self.assertFalse(payload["real_execution_performed"])
            self.assertFalse(payload["p1_hard_gate_passed"])
            live.assert_not_called()
            self.activate_runtime.assert_not_called()
            self.assertFalse(output_dir.exists())

    def test_injected_failure_stops_the_suite(self) -> None:
        called: list[str] = []

        def execute(step, output_dir, episode_id):
            called.append(step.step_key)
            success = step.check_id != "E4"
            payload = {
                "success": success,
                "outcome": "camera_ok" if success else "camera_no_change",
                "truth_missing_count": 0 if step.requires_server_truth else None,
                "real_execution_performed": True,
                "cleanup": {
                    "close_returned": True,
                    "backend_marked_closed": True,
                    "environment_reference_cleared": True,
                    "owner_cleared": True,
                },
            }
            return payload, _release(proven=True)

        with tempfile.TemporaryDirectory() as directory:
            runs_root = Path(directory) / "p1_validation_suite"
            runs_root.mkdir()
            output_dir = runs_root / "pilot-stop"
            with patch(
                "obsidianlink.env.integration.p1_suite.FORMAL_SUITE_RUNS_ROOT",
                runs_root.resolve(),
            ):
                result = run_authorized_p1_suite(
                    execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_P1_SUITE,
                    authorized_live_run=AUTHORIZED_LIVE_P1_SUITE_RUN_VALUE,
                    output_dir=output_dir,
                    execute_step=execute,
                )
            self.assertEqual(result.verdict, VERDICT_VALIDATION_FAILED)
            self.assertEqual(result.stopped_after, "E4")
            self.assertEqual(called, ["E0", "E1", "E2", "E3", "E4"])
            self.assertEqual(
                [call.args[0] for call in self.activate_runtime.call_args_list],
                ["E0", "E1", "E2", "E3", "E4"],
            )
            self.assertTrue(all(case.runtime["verified"] for case in result.cases))
            self.assertFalse(result.p1_hard_gate_passed)
            self.assertFalse(result.integration_verified)
            self.assertTrue((output_dir / "p1_suite.json").is_file())

    def test_cli_check_does_not_start_minerl(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(["--check"])
        payload = json.loads(output.getvalue())
        self.assertEqual(status, 0)
        self.assertFalse(payload["integration_verified"])
        self.assertFalse(payload["p1_hard_gate_passed"])

    def test_wrong_authorization_is_rejected(self) -> None:
        with self.assertRaises(P1SuiteAuthorizationError):
            preflight_authorized_p1_suite(
                execution_mode="authorized_live_e0",
                authorized_live_run=AUTHORIZED_LIVE_P1_SUITE_RUN_VALUE,
            )


if __name__ == "__main__":
    unittest.main()

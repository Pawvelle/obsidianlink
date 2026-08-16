from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from obsidianlink.env.integration.startup_reliability import (
    FINGERPRINT_MALMO,
    FINGERPRINT_NATIVE,
    aggregate_attempts,
    build_attempt_record,
    classify_failure,
    create_unique_run_dir,
    extract_jvm_crash_details,
    load_child_evidence,
)


def _child(*, success: bool = True, reset: bool = True) -> dict[str, object]:
    return {
        "attempt_id": "attempt-001",
        "episode_id": "startup-001",
        "backend_opened": True,
        "environment_created": True,
        "reset_completed": reset,
        "initial_state_present": success,
        "close_returned": True,
        "success": success,
        "outcome": "lifecycle_ok" if success else "reset_failed",
        "error": None if success else "reset error",
        "close_error": None,
        "reset_attempt_count": 1,
        "environment_launch_count": 1,
        "cleanup": {
            "close_returned": True,
            "backend_marked_closed": True,
            "environment_reference_cleared": True,
            "owner_cleared": True,
            "process_release_proven": False,
        },
    }


def _record(
    *,
    child: dict[str, object] | None = None,
    text: str = "",
    exit_code: int = 0,
    timed_out: bool = False,
    duration: float = 2.0,
) -> dict[str, object]:
    value = _child() if child is None else child
    return build_attempt_record(
        attempt_id="attempt-001",
        episode_id="startup-001",
        started_at="2026-08-16T00:00:00Z",
        finished_at="2026-08-16T00:00:02Z",
        duration_seconds=duration,
        exit_code=exit_code,
        timed_out=timed_out,
        child=value,
        child_error=None,
        combined_text=text,
    )


class AggregationTests(unittest.TestCase):
    def test_twenty_attempt_aggregation(self) -> None:
        attempts = [_record(duration=float(index)) for index in range(1, 21)]
        summary = aggregate_attempts(attempts)
        self.assertEqual(summary["total_attempts"], 20)
        self.assertEqual(summary["successful_attempts"], 20)
        self.assertEqual(summary["failed_attempts"], 0)

    def test_all_success_aggregation(self) -> None:
        summary = aggregate_attempts([_record(), _record()])
        self.assertEqual(summary["first_attempt_success_rate"], 1.0)
        self.assertEqual(summary["engineering_interpretation"], "stable")

    def test_mixed_aggregation(self) -> None:
        failure = _record(child=_child(success=False, reset=False))
        summary = aggregate_attempts([_record(), failure])
        self.assertEqual(summary["successful_attempts"], 1)
        self.assertEqual(summary["failed_attempts"], 1)
        self.assertEqual(summary["failure_counts_by_class"], {"reset_failure": 1})

    def test_duration_statistics(self) -> None:
        summary = aggregate_attempts(
            [_record(duration=1.0), _record(duration=2.0), _record(duration=9.0)]
        )
        self.assertEqual(summary["mean_startup_duration_seconds"], 4.0)
        self.assertEqual(summary["median_startup_duration_seconds"], 2.0)
        self.assertEqual(summary["min_startup_duration_seconds"], 1.0)
        self.assertEqual(summary["max_startup_duration_seconds"], 9.0)

    def test_failure_stage_aggregation(self) -> None:
        failure = _record(child=_child(success=False, reset=False))
        summary = aggregate_attempts([failure])
        self.assertEqual(summary["failure_counts_by_stage"], {"reset": 1})


class ClassificationTests(unittest.TestCase):
    def test_native_crash_fingerprint(self) -> None:
        result = classify_failure(
            "SIGSEGV liblwjgl_stb.dylib Sound engine STBVorbis",
            child=_child(success=False, reset=False),
            timed_out=False,
            exit_code=1,
        )
        self.assertEqual(
            result[:3],
            ("minecraft_startup", "minecraft_native_crash", FINGERPRINT_NATIVE),
        )

    def test_malmo_eof_fingerprint(self) -> None:
        result = classify_failure(
            "Malmo connection EOF",
            child=_child(success=False, reset=False),
            timed_out=False,
            exit_code=1,
        )
        self.assertEqual(result[:3], ("minecraft_startup", "malmo_eof", FINGERPRINT_MALMO))

    def test_native_crash_is_primary_over_malmo_eof(self) -> None:
        result = classify_failure(
            "SIGSEGV liblwjgl_stb Sound engine; Malmo EOF",
            child=_child(success=False, reset=False),
            timed_out=False,
            exit_code=1,
        )
        self.assertEqual(result[1], "minecraft_native_crash")
        self.assertIn("malmo_eof", result[3])

    def test_hs_err_fields_are_normalized(self) -> None:
        details = extract_jvm_crash_details(
            '# SIGSEGV\n# Problematic frame:\n# C  [liblwjgl_stb.dylib+0x4c158]\n'
            'Current thread (0x1): JavaThread "Sound engine" daemon'
        )
        self.assertEqual(details["signal"], "SIGSEGV")
        self.assertEqual(details["problematic_frame"], "liblwjgl_stb.dylib+0x4c158")
        self.assertEqual(details["native_library"], "liblwjgl_stb.dylib")
        self.assertEqual(details["thread_name"], "Sound engine")

    def test_java_pid_and_environment_port_are_preserved(self) -> None:
        result = build_attempt_record(
            attempt_id="attempt-006",
            episode_id="startup-006",
            started_at="2026-08-16T00:00:00Z",
            finished_at="2026-08-16T00:00:02Z",
            duration_seconds=2.0,
            exit_code=1,
            timed_out=False,
            child=_child(success=False, reset=False),
            child_error=None,
            combined_text="reset failed",
            subprocess_pid=100,
            tracked_descendants=[
                {
                    "pid": 200,
                    "command": "java -XstartOnFirstThread -jar mcprec-6.13.jar --envPort=9573",
                }
            ],
        )
        self.assertEqual(result["subprocess_pid"], 100)
        self.assertEqual(result["launch_processes"][0]["pid"], 200)
        self.assertEqual(result["launch_processes"][0]["environment_port"], 9573)

    def test_mission_xml_does_not_create_false_malmo_eof(self) -> None:
        result = classify_failure(
            "ProjectMalmo <AllowPassageOfTime>false</AllowPassageOfTime>",
            child=_child(success=False, reset=False),
            timed_out=False,
            exit_code=1,
        )
        self.assertEqual(result[1], "reset_failure")
        self.assertNotIn("malmo_eof", result[3])

    def test_malmo_mission_reply_none_uses_specific_fingerprint(self) -> None:
        child = _child(success=False, reset=False)
        child["error_traceback"] = (
            "File minerl/env/_multiagent.py, in _send_mission\n"
            "TypeError: a bytes-like object is required, not 'NoneType'"
        )
        result = _record(child=child)
        self.assertEqual(result["failure_stage"], "reset")
        self.assertEqual(result["failure_class"], "reset_failure")
        self.assertEqual(result["failure_fingerprint"], "malmo_mission_reply_missing")

    def test_timeout(self) -> None:
        result = _record(timed_out=True, exit_code=-15)
        self.assertEqual(result["failure_class"], "timeout")
        self.assertEqual(result["failure_stage"], "minecraft_startup")

    def test_nonzero_subprocess_exit_code(self) -> None:
        result = _record(exit_code=17)
        self.assertFalse(result["success"])
        self.assertEqual(result["failure_fingerprint"], "nonzero_subprocess_exit")


class EvidenceTests(unittest.TestCase):
    def test_missing_and_malformed_child_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "child.json"
            self.assertIsNone(load_child_evidence(path)[0])
            path.write_text("not json", encoding="utf-8")
            self.assertIsNone(load_child_evidence(path)[0])
            path.write_text(json.dumps({"attempt_id": "attempt-001"}), encoding="utf-8")
            self.assertIn("missing fields", load_child_evidence(path)[1] or "")

    def test_each_run_creates_unique_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            first = create_unique_run_dir(root)
            second = create_unique_run_dir(root)
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())

    def test_existing_evidence_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            first = create_unique_run_dir(root)
            marker = first / "summary.json"
            marker.write_text("historical", encoding="utf-8")
            second = create_unique_run_dir(root)
            self.assertEqual(marker.read_text(encoding="utf-8"), "historical")
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()

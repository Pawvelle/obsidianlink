"""P1 MineRL first-attempt startup reliability calibration.

The parent runner launches one fresh Python process per attempt.  Each child
reuses the E0 lifecycle adapter and fixes ``max_reset_attempts`` to one.  This
module does not implement E10 or any agent/benchmark action.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from obsidianlink.env.integration.e0_adapter import MineRLE0LifecycleAdapter
from obsidianlink.env.integration.e0_cleanup import descendant_pids, snapshot_process_table
from obsidianlink.env.validation import E0_LIFECYCLE_CASE, EnvironmentValidationRunner


ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = (ROOT / "runs" / "p1_startup_reliability").resolve()
DEFAULT_EPISODES = 20
# Minecraft commonly needs several minutes for a cold launch/world creation.
# Ten minutes bounds genuine hangs while staying well above observed starts.
DEFAULT_TIMEOUT_SECONDS = 600.0
MAX_RESET_ATTEMPTS = 1
FINGERPRINT_NATIVE = "lwjgl_stb_sound_engine_sigsegv"
FINGERPRINT_MALMO = "malmo_eof"
FINGERPRINT_RESET = "other_reset_error"


def extract_jvm_crash_details(text: str) -> dict[str, str | None]:
    """Extract stable fields from an ``hs_err_pid`` report or launch log."""

    signal_match = re.search(r"(?:^|\n)#?\s*(SIG[A-Z]+)\b", text)
    frame_match = re.search(
        r"(?:Problematic frame:\s*\n)?#?\s*C\s+\[([^\]]+)\]",
        text,
    )
    thread_match = re.search(
        r'Current thread[^\n]*JavaThread\s+"([^"]+)"',
        text,
    )
    frame = frame_match.group(1).strip() if frame_match else None
    library_match = re.search(r"([^/\s+]+\.dylib)", frame or "")
    return {
        "signal": signal_match.group(1) if signal_match else None,
        "problematic_frame": frame,
        "thread_name": thread_match.group(1) if thread_match else None,
        "native_library": library_match.group(1) if library_match else None,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def create_unique_run_dir(root: Path = RUNS_ROOT) -> Path:
    """Create a never-overwritten run directory."""

    root.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        candidate = root / f"startup-reliability-{stamp}-{uuid.uuid4().hex[:8]}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("could not allocate a unique reliability run directory")


def load_child_evidence(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return None, f"child evidence unavailable or malformed: {error}"
    if not isinstance(value, dict):
        return None, "child evidence must be a JSON object"
    required = {
        "attempt_id",
        "episode_id",
        "backend_opened",
        "environment_created",
        "reset_completed",
        "initial_state_present",
        "close_returned",
        "success",
        "outcome",
        "reset_attempt_count",
        "environment_launch_count",
        "cleanup",
    }
    missing = sorted(required.difference(value))
    if missing:
        return None, "child evidence missing fields: " + ", ".join(missing)
    return value, None


def classify_failure(
    text: str,
    *,
    child: Mapping[str, Any] | None,
    timed_out: bool,
    exit_code: int | None,
    cleanup_anomaly: bool = False,
) -> tuple[str, str, str | None, list[str]]:
    """Return stage, primary class, normalized fingerprint, secondary signs."""

    lowered = text.lower()
    sigsegv = "sigsegv" in lowered
    lwjgl_stb = "liblwjgl_stb" in lowered or "stbvorbis" in lowered
    sound_engine = "sound engine" in lowered
    generic_lwjgl = "liblwjgl" in lowered
    native = sigsegv and (lwjgl_stb or sound_engine or generic_lwjgl)
    explicit_eof = bool(
        re.search(r"\beof\b", lowered)
        or "eoferror" in lowered
        or "end of file" in lowered
    )
    # MineRL mission XML always contains ``ProjectMalmo`` and may contain
    # ordinary text such as ``PassageOfTime``.  Substring matching "eof"
    # across those words would be a false EOF symptom, so require an explicit
    # EOF token/error phrase.
    malmo_eof = "malmo" in lowered and explicit_eof
    secondary = ["malmo_eof"] if malmo_eof else []
    if timed_out:
        return "minecraft_startup", "timeout", "startup_timeout", secondary
    if native:
        return "minecraft_startup", "minecraft_native_crash", FINGERPRINT_NATIVE, secondary
    if malmo_eof:
        return "minecraft_startup", "malmo_eof", FINGERPRINT_MALMO, []
    if cleanup_anomaly:
        return "process_cleanup", "cleanup_failure", "process_cleanup_anomaly", secondary
    if child is None:
        return "unknown", "unknown_infrastructure_failure", "missing_or_malformed_child_evidence", secondary
    outcome = child.get("outcome")
    if child.get("close_error") or outcome == "close_failed":
        return "close", "cleanup_failure", "close_error", secondary
    if not child.get("reset_completed", False):
        missing_protocol_reply = (
            "bytes-like object is required" in lowered and "nonetype" in lowered
        )
        if missing_protocol_reply and "_send_mission" in lowered:
            return "reset", "reset_failure", "malmo_mission_reply_missing", secondary
        if missing_protocol_reply and "_to_move_quit_current_episode" in lowered:
            return "minecraft_startup", "reset_failure", "malmo_quit_reply_missing", secondary
        if missing_protocol_reply and "_peek_obs" in lowered:
            return (
                "observation",
                "observation_failure",
                "malmo_observation_reply_missing",
                secondary,
            )
        stage = "environment_creation" if not child.get("environment_created", False) else "reset"
        return stage, "reset_failure", FINGERPRINT_RESET, secondary
    if not child.get("initial_state_present", False):
        return "observation", "observation_failure", "initial_observation_missing", secondary
    if exit_code not in (0, None):
        return "unknown", "unknown_infrastructure_failure", "nonzero_subprocess_exit", secondary
    return "unknown", "unknown_infrastructure_failure", "unknown_infrastructure_failure", secondary


def _run_child(attempt_dir: Path, attempt_id: str, episode_id: str) -> int:
    started_at = _utc_now()
    start = time.monotonic()
    holder: dict[str, MineRLE0LifecycleAdapter | None] = {"adapter": None}

    def factory() -> MineRLE0LifecycleAdapter:
        adapter = MineRLE0LifecycleAdapter(
            episode_id=episode_id,
            backend_kwargs={"max_reset_attempts": MAX_RESET_ATTEMPTS},
        )
        holder["adapter"] = adapter
        return adapter

    lifecycle = EnvironmentValidationRunner().run(
        E0_LIFECYCLE_CASE,
        factory,
        episode_id=episode_id,
    )
    adapter = holder["adapter"]
    cleanup = None if adapter is None else adapter.cleanup_status()
    audit = (
        {"reset_attempt_count": 0, "environment_launch_count": 0}
        if adapter is None
        else adapter.reset_audit()
    )
    reset_diagnostics = (
        {"traceback": None, "exception_chain": []}
        if adapter is None
        else adapter.reset_failure_diagnostics()
    )
    cleanup_payload = (
        {
            "close_returned": lifecycle.closed,
            "backend_marked_closed": None,
            "environment_reference_cleared": None,
            "owner_cleared": None,
            "process_release_proven": False,
        }
        if cleanup is None
        else cleanup.as_dict()
    )
    success = lifecycle.success and not (
        cleanup is not None and cleanup.has_explicit_failure()
    )
    payload: dict[str, Any] = {
        "attempt_id": attempt_id,
        "episode_id": episode_id,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": round(time.monotonic() - start, 6),
        "backend_opened": False if adapter is None else adapter.open_succeeded,
        "environment_created": audit["environment_launch_count"] > 0,
        "reset_completed": lifecycle.reset_completed,
        "initial_state_present": lifecycle.initial_state_present,
        "close_returned": cleanup_payload["close_returned"],
        "success": success,
        "outcome": lifecycle.outcome if success else (
            "cleanup_failed" if lifecycle.success else lifecycle.outcome
        ),
        "error": lifecycle.error,
        "error_traceback": reset_diagnostics["traceback"],
        "exception_chain": reset_diagnostics["exception_chain"],
        "close_error": lifecycle.close_error,
        "reset_attempt_count": audit["reset_attempt_count"],
        "environment_launch_count": audit["environment_launch_count"],
        "cleanup": cleanup_payload,
        "max_reset_attempts": MAX_RESET_ATTEMPTS,
        "validation_action_executed": False,
        "integration_verified": False,
        "calibration_only": True,
    }
    _write_json(attempt_dir / "child_evidence.json", payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if success else 1


def _snapshot_runtime_logs() -> dict[str, tuple[int, int]]:
    paths = list((ROOT / "logs").glob("mc_*.log"))
    paths.extend((ROOT / "logs" / "minerl_watchers").glob("watcher_*.log"))
    return {
        str(path.resolve()): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in paths
        if path.is_file()
    }


def _collect_logs(
    attempt_dir: Path,
    before: Mapping[str, tuple[int, int]],
    combined_text: str,
    tracked_pids: set[int],
) -> list[dict[str, Any]]:
    candidates: list[tuple[Path, str]] = []
    for path in list((ROOT / "logs").glob("mc_*.log")) + list(
        (ROOT / "logs" / "minerl_watchers").glob("watcher_*.log")
    ):
        if path.is_file() and before.get(str(path.resolve())) != (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        ):
            candidates.append((path.resolve(), "runtime_log"))
    reference_text = combined_text
    for path, _ in candidates:
        reference_text += "\n" + path.read_text(encoding="utf-8", errors="replace")
    for match in re.findall(r"(/[^\s]+/hs_err_pid(\d+)\.log)", reference_text):
        path = Path(match[0]).resolve()
        pid = int(match[1])
        if path.is_file() and (pid in tracked_pids or any(str(path) in text for text in (reference_text,))):
            candidates.append((path, "jvm_crash"))
    evidence_dir = attempt_dir / "logs"
    evidence: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for source, kind in candidates:
        if source in seen:
            continue
        seen.add(source)
        evidence_dir.mkdir(exist_ok=True)
        target = evidence_dir / f"{len(evidence) + 1:02d}-{source.name}"
        shutil.copy2(source, target)
        data = target.read_bytes()
        evidence.append(
            {
                "kind": kind,
                "source_path": str(source),
                "copied_path": str(target.resolve()),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return evidence


def build_attempt_record(
    *,
    attempt_id: str,
    episode_id: str,
    started_at: str,
    finished_at: str,
    duration_seconds: float,
    exit_code: int | None,
    timed_out: bool,
    child: Mapping[str, Any] | None,
    child_error: str | None,
    combined_text: str,
    tracked_descendants: Sequence[Mapping[str, Any]] = (),
    residual_descendants: Sequence[Mapping[str, Any]] = (),
    cleanup_actions: Sequence[str] = (),
    log_evidence: Sequence[Mapping[str, Any]] = (),
    subprocess_pid: int | None = None,
) -> dict[str, Any]:
    cleanup_payload = child.get("cleanup", {}) if child is not None else {}
    explicit_cleanup_failure = any(
        cleanup_payload.get(name) is False
        for name in (
            "close_returned",
            "backend_marked_closed",
            "environment_reference_cleared",
            "owner_cleared",
        )
    )
    cleanup_anomaly = explicit_cleanup_failure or bool(residual_descendants)
    nominal_success = bool(child and child.get("success"))
    success = nominal_success and exit_code == 0 and not timed_out and not cleanup_anomaly
    diagnostic_text = (
        combined_text
        + "\n"
        + (child_error or "")
        + "\n"
        + ("" if child is None else str(child.get("error_traceback") or ""))
    )
    if success:
        stage, failure_class, fingerprint, secondary = (
            "none",
            "success",
            None,
            [],
        )
    else:
        stage, failure_class, fingerprint, secondary = classify_failure(
            diagnostic_text,
            child=child,
            timed_out=timed_out,
            exit_code=exit_code,
            cleanup_anomaly=cleanup_anomaly,
        )
    error = child_error if child is None else child.get("error")
    if timed_out:
        error = f"subprocess exceeded timeout"
    launch_processes = []
    for process in tracked_descendants:
        command = str(process.get("command", ""))
        if not re.search(r"(?:^|/)java(?:\s|$)", command):
            continue
        port_match = re.search(r"--envPort=(\d+)", command)
        launch_processes.append(
            {
                "pid": process.get("pid"),
                "command": command,
                "environment_port": int(port_match.group(1)) if port_match else None,
            }
        )
    crash_details = extract_jvm_crash_details(diagnostic_text) if not success else None
    return {
        "attempt_id": attempt_id,
        "episode_id": episode_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(duration_seconds, 6),
        "startup_duration_seconds": round(duration_seconds, 6),
        "subprocess_started": True,
        "subprocess_pid": subprocess_pid,
        "subprocess_exit_code": exit_code,
        "backend_opened": bool(child and child.get("backend_opened")),
        "environment_created": bool(child and child.get("environment_created")),
        "reset_completed": bool(child and child.get("reset_completed")),
        "initial_state_present": bool(child and child.get("initial_state_present")),
        "close_returned": bool(child and child.get("close_returned")),
        "success": success,
        "outcome": "lifecycle_ok" if success else failure_class,
        "error": error,
        "error_traceback": None if child is None else child.get("error_traceback"),
        "exception_chain": [] if child is None else child.get("exception_chain", []),
        "close_error": None if child is None else child.get("close_error"),
        "reset_attempt_count": 0 if child is None else child.get("reset_attempt_count", 0),
        "environment_launch_count": 0 if child is None else child.get("environment_launch_count", 0),
        "max_reset_attempts": MAX_RESET_ATTEMPTS,
        "failure_stage": stage,
        "failure_class": failure_class,
        "failure_fingerprint": fingerprint,
        "secondary_symptoms": secondary,
        "jvm_crash_details": crash_details,
        "launch_processes": launch_processes,
        "timed_out": timed_out,
        "cleanup": {
            **cleanup_payload,
            "tracked_descendants": list(tracked_descendants),
            "residual_descendants": list(residual_descendants),
            "cleanup_actions": list(cleanup_actions),
            "process_release_proven": False,
            "process_release_limitation": (
                "PID-tree observation can detect tracked residuals but cannot prove "
                "that no process escaped or was reparented before observation"
            ),
        },
        "cleanup_anomaly": cleanup_anomaly,
        "log_evidence": list(log_evidence),
        "validation_action_executed": False,
        "infrastructure_only": True,
        "integration_verified": False,
    }


def aggregate_attempts(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(attempts)
    successes = sum(bool(item.get("success")) for item in attempts)
    durations = [float(item["startup_duration_seconds"]) for item in attempts]
    classes = Counter(str(item.get("failure_class")) for item in attempts if not item.get("success"))
    stages = Counter(str(item.get("failure_stage")) for item in attempts if not item.get("success"))
    fingerprints = Counter(
        str(item["failure_fingerprint"])
        for item in attempts
        if not item.get("success") and item.get("failure_fingerprint")
    )
    native = classes.get("minecraft_native_crash", 0)
    malmo = sum(
        item.get("failure_class") == "malmo_eof"
        or "malmo_eof" in item.get("secondary_symptoms", [])
        for item in attempts
    )
    summary: dict[str, Any] = {
        "total_attempts": total,
        "successful_attempts": successes,
        "failed_attempts": total - successes,
        "first_attempt_success_rate": successes / total if total else 0.0,
        "native_crash_count": native,
        "native_crash_rate": native / total if total else 0.0,
        "malmo_eof_count": malmo,
        "timeout_count": classes.get("timeout", 0),
        "cleanup_failure_count": sum(bool(item.get("cleanup_anomaly")) for item in attempts),
        "failure_counts_by_class": dict(sorted(classes.items())),
        "failure_counts_by_stage": dict(sorted(stages.items())),
        "failure_counts_by_fingerprint": dict(sorted(fingerprints.items())),
        "process_release_proven_attempts": sum(
            item.get("cleanup", {}).get("process_release_proven") is True
            for item in attempts
        ),
        "validation_action_executed_attempts": sum(
            item.get("validation_action_executed") is True for item in attempts
        ),
        "integration_verified": False,
        "p1_hard_gate_passed": False,
    }
    if durations:
        summary.update(
            {
                "mean_startup_duration_seconds": statistics.mean(durations),
                "median_startup_duration_seconds": statistics.median(durations),
                "min_startup_duration_seconds": min(durations),
                "max_startup_duration_seconds": max(durations),
            }
        )
    else:
        summary.update(
            {
                "mean_startup_duration_seconds": None,
                "median_startup_duration_seconds": None,
                "min_startup_duration_seconds": None,
                "max_startup_duration_seconds": None,
            }
        )
    if total == 0:
        interpretation = "inconclusive"
    elif successes == total:
        interpretation = "stable"
    elif successes / total >= 0.8 and native > 0:
        interpretation = "mostly_stable_with_known_native_failures"
    else:
        interpretation = "unstable"
    summary["engineering_interpretation"] = interpretation
    summary["interpretation_is_heuristic_not_benchmark_threshold"] = True
    return summary


def run_one_attempt(
    attempt_dir: Path,
    attempt_id: str,
    episode_id: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    attempt_dir.mkdir(exist_ok=False)
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    child_path = attempt_dir / "child_evidence.json"
    before_logs = _snapshot_runtime_logs()
    started_at = _utc_now()
    started = time.monotonic()
    command = [
        sys.executable,
        "-m",
        "obsidianlink.env.integration.startup_reliability",
        "--child",
        "--attempt-dir",
        str(attempt_dir.resolve()),
        "--attempt-id",
        attempt_id,
        "--episode-id",
        episode_id,
    ]
    tracked: dict[int, str] = {}
    subprocess_pid: int | None = None
    timed_out = False
    exit_code: int | None = None
    cleanup_actions: list[str] = []
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
        )
        subprocess_pid = process.pid
        deadline = started + timeout_seconds
        while process.poll() is None and time.monotonic() < deadline:
            table = snapshot_process_table()
            for pid in descendant_pids(process.pid, table):
                tracked[pid] = table[pid][1]
            time.sleep(0.25)
        if process.poll() is None:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            cleanup_actions.append(f"SIGTERM process group {process.pid}")
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                cleanup_actions.append(f"SIGKILL process group {process.pid}")
                process.wait(timeout=10)
        exit_code = process.returncode
    # Allow normal close/watchers a bounded release window, then inspect only
    # PIDs previously proven to be descendants of this attempt's child.
    residual: dict[int, str] = {}
    release_deadline = time.monotonic() + 5.0
    while time.monotonic() < release_deadline:
        table = snapshot_process_table()
        residual = {
            pid: command_text
            for pid, command_text in tracked.items()
            if pid in table and table[pid][1] == command_text
        }
        if not residual:
            break
        time.sleep(0.25)
    if residual:
        for pid in sorted(residual):
            try:
                os.kill(pid, signal.SIGTERM)
                cleanup_actions.append(f"SIGTERM confirmed descendant {pid}")
            except ProcessLookupError:
                pass
    finished_at = _utc_now()
    duration = time.monotonic() - started
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    child, child_error = load_child_evidence(child_path)
    tracked_payload = [
        {"pid": pid, "command": command_text}
        for pid, command_text in sorted(tracked.items())
    ]
    residual_payload = [
        {"pid": pid, "command": command_text}
        for pid, command_text in sorted(residual.items())
    ]
    log_evidence = _collect_logs(
        attempt_dir,
        before_logs,
        stdout_text + "\n" + stderr_text,
        set(tracked),
    )
    record = build_attempt_record(
        attempt_id=attempt_id,
        episode_id=episode_id,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration,
        exit_code=exit_code,
        timed_out=timed_out,
        child=child,
        child_error=child_error,
        combined_text=stdout_text + "\n" + stderr_text + "\n" + "\n".join(
            Path(item["copied_path"]).read_text(encoding="utf-8", errors="replace")
            for item in log_evidence
        ),
        tracked_descendants=tracked_payload,
        residual_descendants=residual_payload,
        cleanup_actions=cleanup_actions,
        log_evidence=log_evidence,
        subprocess_pid=subprocess_pid,
    )
    _write_json(attempt_dir / "attempt.json", record)
    return record


def run_reliability(
    *,
    episodes: int = DEFAULT_EPISODES,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    runs_root: Path = RUNS_ROOT,
) -> tuple[Path, dict[str, Any]]:
    if type(episodes) is not int or episodes < 1:
        raise ValueError("episodes must be a positive integer")
    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    run_dir = create_unique_run_dir(runs_root.resolve())
    config = {
        "created_at": _utc_now(),
        "episodes": episodes,
        "timeout_seconds_per_subprocess": float(timeout_seconds),
        "max_reset_attempts": MAX_RESET_ATTEMPTS,
        "fresh_python_process_per_attempt": True,
        "lifecycle": "E0 open/reset/initial observation/close only",
        "validation_action_executed": False,
        "e10_started": False,
        "gradle_authorized": False,
        "model_api_authorized": False,
        "integration_verified": False,
        "p1_hard_gate_passed": False,
        "startup_duration_basis": "parent subprocess wall time through close and cleanup inspection",
    }
    _write_json(run_dir / "config.json", config)
    attempts: list[dict[str, Any]] = []
    jsonl_path = run_dir / "attempts.jsonl"
    for index in range(1, episodes + 1):
        attempt_id = f"attempt-{index:03d}"
        episode_id = f"startup-{index:03d}"
        print(f"[{index}/{episodes}] {episode_id}", flush=True)
        record = run_one_attempt(
            run_dir / attempt_id,
            attempt_id,
            episode_id,
            float(timeout_seconds),
        )
        attempts.append(record)
        with jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        print(
            f"[{index}/{episodes}] {record['outcome']} "
            f"({record['duration_seconds']:.1f}s)",
            flush=True,
        )
    summary = aggregate_attempts(attempts)
    summary.update({"run_dir": str(run_dir), "finished_at": _utc_now()})
    _write_json(run_dir / "summary.json", summary)
    return run_dir, summary


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="P1 first-attempt MineRL startup reliability calibration"
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--attempt-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--attempt-id", help=argparse.SUPPRESS)
    parser.add_argument("--episode-id", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.child:
        if args.attempt_dir is None or not args.attempt_id or not args.episode_id:
            raise ValueError("child mode requires attempt directory and identifiers")
        return _run_child(args.attempt_dir, args.attempt_id, args.episode_id)
    if any(value is not None for value in (args.attempt_dir, args.attempt_id, args.episode_id)):
        raise ValueError("child-only arguments require --child")
    run_dir, summary = run_reliability(
        episodes=args.episodes,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

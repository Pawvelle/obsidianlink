"""Ordered P1 E0--E12 validation suite and process-release Hard Gate.

This is orchestration only. Case science stays in the existing E0--E12
runners. Imports, ``--check``, and ``--preflight-only`` never start
MineRL. A later authorized live command is required for a single pilot.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from obsidianlink.core.task_catalog import load_task_catalog
from obsidianlink.env.integration.e0_cleanup import (
    ProcessReleaseStatus,
    inspect_os_process_release,
    merge_tracked_descendants,
    residual_descendants,
    tracked_descendants,
)
from obsidianlink.env.integration.p1_suite_runtime import (
    P1SuiteRuntimeError,
    activate_required_runtime,
    failed_runtime_record,
    is_verified_runtime,
    required_runtime,
)
from obsidianlink.env.validation.contract import P1_VALIDATION_CASES
from obsidianlink.env.validation.result import UNIT_VERIFIED


ROOT = Path(__file__).resolve().parents[3]
FORMAL_SUITE_RUNS_ROOT = (ROOT / "runs" / "p1_validation_suite").resolve()
EXECUTION_MODE_AUTHORIZED_LIVE_P1_SUITE = "authorized_live_p1_suite"
AUTHORIZED_LIVE_P1_SUITE_RUN_VALUE = "p1_e0_e12_validation_suite"
DEFAULT_CASE_TIMEOUT_SECONDS = 600.0
PROCESS_RELEASE_WAIT_SECONDS = 5.0

VERDICT_VALIDATION_FAILED = "validation_failed"
VERDICT_TRUTH_MISSING = "truth_missing"
VERDICT_CLEANUP_FAILED = "cleanup_failed"
VERDICT_PROCESS_RELEASE_NOT_PROVEN = "process_release_not_proven"
VERDICT_HARD_GATE_SUCCESS = "hard_gate_success"
SUITE_VERDICTS = frozenset(
    {
        VERDICT_VALIDATION_FAILED,
        VERDICT_TRUTH_MISSING,
        VERDICT_CLEANUP_FAILED,
        VERDICT_PROCESS_RELEASE_NOT_PROVEN,
        VERDICT_HARD_GATE_SUCCESS,
    }
)

_PROCESS_LIVE_RUN_STARTED = False
_PROCESS_LIVE_RUN_LOCK = threading.Lock()


class P1SuiteAuthorizationError(ValueError):
    """Raised when the exact P1 suite live gate or preflight is invalid."""


@dataclass(frozen=True)
class P1SuiteStep:
    check_id: str
    name: str
    variant: str | None
    requires_server_truth: bool
    execution_mode: str
    authorized_live_run: str
    module: str
    runs_root_relative: str

    @property
    def step_key(self) -> str:
        return self.check_id if self.variant is None else f"{self.check_id}:{self.variant}"


def _gates_for_case(check_id: str) -> tuple[tuple[str | None, str, str, str, str], ...]:
    """Return ``(variant, execution_mode, token, module, runs_root_relative)``."""

    module = f"obsidianlink.env.integration.{check_id.lower()}_run"
    if check_id == "E7":
        return (
            (
                "water",
                "authorized_live_e7_water",
                "e7_water_bucket",
                module,
                "p1_e7_bucket_usage/water",
            ),
            (
                "lava",
                "authorized_live_e7_lava",
                "e7_lava_bucket",
                module,
                "p1_e7_bucket_usage/lava",
            ),
        )
    if check_id == "E9":
        return (
            (
                "water",
                "authorized_live_e9_water",
                "e9_water_fluid_truth",
                module,
                "p1_e9_fluid_truth/water",
            ),
            (
                "lava",
                "authorized_live_e9_lava",
                "e9_lava_fluid_truth",
                module,
                "p1_e9_fluid_truth/lava",
            ),
        )
    tokens = {
        "E0": ("e0_reset_close", "p1_e0_reset_close"),
        "E1": ("e1_rgb_observation", "p1_e1_rgb_observation"),
        "E2": ("e2_inventory_observation", "p1_e2_inventory_observation"),
        "E3": ("e3_selected_item", "p1_e3_selected_item_observation"),
        "E4": ("e4_camera_control", "p1_e4_camera_control"),
        "E5": ("e5_movement", "p1_e5_movement"),
        "E6": ("e6_block_placement", "p1_e6_block_placement"),
        "E8": ("e8_block_truth", "p1_e8_block_truth"),
        "E10": ("e10_obsidian_conversion", "p1_e10_obsidian_conversion"),
        "E11": ("e11_portal_activation", "p1_e11_portal_activation"),
        "E12": ("e12_dimension_transition", "p1_e12_dimension_transition"),
    }
    token, runs_root = tokens[check_id]
    return (
        (None, f"authorized_live_{check_id.lower()}", token, module, runs_root),
    )


def p1_suite_steps() -> tuple[P1SuiteStep, ...]:
    """Ordered suite derived from ``P1_VALIDATION_CASES`` plus E7/E9 variants."""

    steps: list[P1SuiteStep] = []
    for case in P1_VALIDATION_CASES:
        for variant, mode, token, module, runs_root in _gates_for_case(case.check_id.value):
            steps.append(
                P1SuiteStep(
                    check_id=case.check_id.value,
                    name=case.name,
                    variant=variant,
                    requires_server_truth=case.requires_server_truth,
                    execution_mode=mode,
                    authorized_live_run=token,
                    module=module,
                    runs_root_relative=runs_root,
                )
            )
    return tuple(steps)


P1_SUITE_STEPS = p1_suite_steps()


@dataclass(frozen=True)
class P1CaseSummary:
    check_id: str
    name: str
    variant: str | None
    success: bool
    outcome: str
    requires_server_truth: bool
    truth_missing_count: int | None
    cleanup_failed: bool
    process_release: ProcessReleaseStatus
    real_execution_performed: bool
    integration_verified: bool = False
    runtime: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.check_id not in {case.check_id.value for case in P1_VALIDATION_CASES}:
            raise ValueError("check_id must be a P1 validation case")
        if type(self.success) is not bool or type(self.cleanup_failed) is not bool:
            raise ValueError("success and cleanup_failed must be bool")
        if type(self.real_execution_performed) is not bool:
            raise ValueError("real_execution_performed must be bool")
        if type(self.integration_verified) is not bool:
            raise ValueError("integration_verified must be bool")
        if self.integration_verified:
            raise ValueError("suite records cannot claim integration_verified")
        if not isinstance(self.process_release, ProcessReleaseStatus):
            raise ValueError("process_release must be ProcessReleaseStatus")
        if self.truth_missing_count is not None and (
            type(self.truth_missing_count) is not int or self.truth_missing_count < 0
        ):
            raise ValueError("truth_missing_count must be a non-negative int or None")
        if self.success:
            runtime = self.runtime
            if not is_verified_runtime(runtime):
                raise ValueError("success requires verified runtime identity")
            name, sha256 = required_runtime(self.check_id)
            if (
                runtime.get("check_id") != self.check_id
                or runtime.get("runtime") != name
                or runtime.get("sha256") != sha256
            ):
                raise ValueError("verified runtime does not match the required mapping")

    @property
    def step_key(self) -> str:
        return self.check_id if self.variant is None else f"{self.check_id}:{self.variant}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "cleanup_failed": self.cleanup_failed,
            "integration_verified": False,
            "name": self.name,
            "outcome": self.outcome,
            "process_release": self.process_release.as_dict(),
            "real_execution_performed": self.real_execution_performed,
            "requires_server_truth": self.requires_server_truth,
            "runtime": dict(self.runtime) if isinstance(self.runtime, Mapping) else None,
            "success": self.success,
            "truth_missing_count": self.truth_missing_count,
            "variant": self.variant,
        }


def _require_bool(value: object, field_name: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be bool")


def _cleanup_failed_from_payload(payload: Mapping[str, Any]) -> bool:
    if payload.get("outcome") == "cleanup_failed":
        return True
    cleanup = payload.get("cleanup")
    if not isinstance(cleanup, Mapping):
        return False
    return any(
        cleanup.get(name) is False
        for name in (
            "close_returned",
            "backend_marked_closed",
            "environment_reference_cleared",
            "owner_cleared",
        )
    )


def case_summary_from_payload(
    payload: Mapping[str, Any],
    step: P1SuiteStep,
    process_release: ProcessReleaseStatus,
    runtime: Mapping[str, Any] | None = None,
) -> P1CaseSummary:
    truth_missing_count = payload.get("truth_missing_count")
    if truth_missing_count is not None and type(truth_missing_count) is not int:
        raise ValueError("truth_missing_count must be a non-negative int or None")
    success = bool(payload.get("success"))
    outcome = str(payload.get("outcome") or "runtime_error")
    name, sha256 = required_runtime(step.check_id)
    if success and (
        not is_verified_runtime(runtime)
        or runtime.get("check_id") != step.check_id
        or runtime.get("runtime") != name
        or runtime.get("sha256") != sha256
    ):
        success = False
        outcome = "runtime_not_verified"
    return P1CaseSummary(
        check_id=step.check_id,
        name=step.name,
        variant=step.variant,
        success=success,
        outcome=outcome,
        requires_server_truth=step.requires_server_truth,
        truth_missing_count=truth_missing_count,
        cleanup_failed=_cleanup_failed_from_payload(payload),
        process_release=process_release,
        real_execution_performed=bool(payload.get("real_execution_performed")),
        integration_verified=False,
        runtime=runtime,
    )


def _case_issue(summary: P1CaseSummary) -> str | None:
    if summary.requires_server_truth and (
        summary.truth_missing_count is None or summary.truth_missing_count != 0
    ):
        return VERDICT_TRUTH_MISSING
    if summary.cleanup_failed:
        return VERDICT_CLEANUP_FAILED
    if not summary.success:
        return VERDICT_VALIDATION_FAILED
    if not summary.process_release.process_release_proven:
        return VERDICT_PROCESS_RELEASE_NOT_PROVEN
    return None


@dataclass(frozen=True)
class P1SuiteResult:
    steps: tuple[P1SuiteStep, ...]
    cases: tuple[P1CaseSummary, ...]
    verdict: str
    all_required_cases_present: bool
    all_cases_succeeded: bool
    truth_missing: bool
    cleanup_failed: bool
    process_release_proven: bool
    human_intervention: bool
    real_execution_performed: bool
    p1_hard_gate_passed: bool
    integration_verified: bool = False
    verification_level: str = UNIT_VERIFIED
    calibration_only: bool = True
    stopped_after: str | None = None

    def __post_init__(self) -> None:
        if self.verdict not in SUITE_VERDICTS:
            raise ValueError(f"unknown suite verdict: {self.verdict!r}")
        for field_name in (
            "all_required_cases_present",
            "all_cases_succeeded",
            "truth_missing",
            "cleanup_failed",
            "process_release_proven",
            "human_intervention",
            "real_execution_performed",
            "p1_hard_gate_passed",
            "integration_verified",
            "calibration_only",
        ):
            _require_bool(getattr(self, field_name), field_name)
        if self.verification_level != UNIT_VERIFIED:
            raise ValueError("this runtime may only emit unit_verified")
        if self.integration_verified:
            raise ValueError("this runtime cannot claim integration_verified")
        if not self.calibration_only:
            raise ValueError("P1 suite records must remain calibration-only")
        if self.p1_hard_gate_passed:
            if self.verdict != VERDICT_HARD_GATE_SUCCESS:
                raise ValueError("hard gate success requires verdict hard_gate_success")
            if self.integration_verified:
                raise ValueError("hard gate success still cannot claim integration_verified")

    def as_dict(self) -> dict[str, Any]:
        return {
            "all_cases_succeeded": self.all_cases_succeeded,
            "all_required_cases_present": self.all_required_cases_present,
            "calibration_only": True,
            "cases": [case.as_dict() for case in self.cases],
            "cleanup_failed": self.cleanup_failed,
            "human_intervention": self.human_intervention,
            "integration_verified": False,
            "p1_hard_gate_passed": self.p1_hard_gate_passed,
            "process_release_proven": self.process_release_proven,
            "real_execution_performed": self.real_execution_performed,
            "stopped_after": self.stopped_after,
            "steps": [step.step_key for step in self.steps],
            "truth_missing": self.truth_missing,
            "verdict": self.verdict,
            "verification_level": UNIT_VERIFIED,
        }


def aggregate_p1_suite(
    cases: Sequence[P1CaseSummary],
    *,
    steps: Sequence[P1SuiteStep] = P1_SUITE_STEPS,
    real_execution_performed: bool,
    human_intervention: bool = False,
    stopped_after: str | None = None,
) -> P1SuiteResult:
    """Fail-closed aggregate. First blocking issue in suite order wins."""

    _require_bool(real_execution_performed, "real_execution_performed")
    _require_bool(human_intervention, "human_intervention")
    expected = tuple(step.step_key for step in steps)
    observed = tuple(case.step_key for case in cases)
    complete = observed == expected
    truth_missing = any(
        case.requires_server_truth
        and (case.truth_missing_count is None or case.truth_missing_count != 0)
        for case in cases
    )
    cleanup_failed = any(case.cleanup_failed for case in cases)
    all_succeeded = complete and all(case.success for case in cases) and not cleanup_failed
    process_release_proven = complete and all(
        case.process_release.process_release_proven for case in cases
    )
    verdict = VERDICT_VALIDATION_FAILED
    for case in cases:
        issue = _case_issue(case)
        if issue is not None:
            verdict = issue
            break
    else:
        if not complete:
            verdict = VERDICT_VALIDATION_FAILED
        elif (
            real_execution_performed
            and not human_intervention
            and process_release_proven
            and all_succeeded
            and not truth_missing
        ):
            verdict = VERDICT_HARD_GATE_SUCCESS
        else:
            verdict = VERDICT_PROCESS_RELEASE_NOT_PROVEN
    hard_gate = verdict == VERDICT_HARD_GATE_SUCCESS
    return P1SuiteResult(
        steps=tuple(steps),
        cases=tuple(cases),
        verdict=verdict,
        all_required_cases_present=complete,
        all_cases_succeeded=all_succeeded,
        truth_missing=truth_missing,
        cleanup_failed=cleanup_failed,
        process_release_proven=process_release_proven,
        human_intervention=human_intervention,
        real_execution_performed=real_execution_performed,
        p1_hard_gate_passed=hard_gate,
        stopped_after=stopped_after,
    )


def reset_authorized_p1_suite_process_guards_for_tests() -> None:
    global _PROCESS_LIVE_RUN_STARTED
    with _PROCESS_LIVE_RUN_LOCK:
        _PROCESS_LIVE_RUN_STARTED = False


def assert_p1_suite_live_authorized(
    *,
    execution_mode: object,
    authorized_live_run: object,
    allow_gradle: object = False,
) -> None:
    if execution_mode != EXECUTION_MODE_AUTHORIZED_LIVE_P1_SUITE:
        raise P1SuiteAuthorizationError(
            "execution_mode must be exactly authorized_live_p1_suite"
        )
    if authorized_live_run != AUTHORIZED_LIVE_P1_SUITE_RUN_VALUE:
        raise P1SuiteAuthorizationError(
            "authorized_live_run must be exactly p1_e0_e12_validation_suite"
        )
    if allow_gradle is not False:
        raise P1SuiteAuthorizationError(
            "Gradle is not authorized for the P1 suite; allow_gradle must be False"
        )


def _validate_catalog_policy() -> None:
    catalog = load_task_catalog(ROOT / "benchmark/catalog/tasks.json")
    if catalog.active_phase != "P1-REAL-MINERL-ENVIRONMENT-VALIDATION":
        raise P1SuiteAuthorizationError(
            "active catalog phase must remain P1 environment validation"
        )
    if any(entry.live_run_allowed for entry in catalog.entries):
        raise P1SuiteAuthorizationError(
            "catalog live_run_allowed must remain false; authorization is per-run only"
        )


def _validate_output_dir(output_dir: Path) -> Path:
    if not isinstance(output_dir, Path):
        raise P1SuiteAuthorizationError("output_dir must be a pathlib.Path")
    if not output_dir.is_absolute():
        raise P1SuiteAuthorizationError("output_dir must be an absolute path")
    resolved = output_dir.resolve()
    if resolved.exists() or resolved.is_symlink():
        raise P1SuiteAuthorizationError(f"output_dir must not already exist: {resolved}")
    try:
        resolved.relative_to(FORMAL_SUITE_RUNS_ROOT)
    except ValueError as error:
        raise P1SuiteAuthorizationError(
            f"output_dir must be under {FORMAL_SUITE_RUNS_ROOT}"
        ) from error
    if resolved == FORMAL_SUITE_RUNS_ROOT or resolved.parent != FORMAL_SUITE_RUNS_ROOT:
        raise P1SuiteAuthorizationError(
            "output_dir must be a unique direct child of runs/p1_validation_suite/"
        )
    return resolved


def check_p1_suite() -> dict[str, Any]:
    _validate_catalog_policy()
    steps = p1_suite_steps()
    case_ids = tuple(case.check_id.value for case in P1_VALIDATION_CASES)
    step_ids = tuple(step.check_id for step in steps)
    if case_ids != tuple(f"E{index}" for index in range(13)):
        raise P1SuiteAuthorizationError("P1_VALIDATION_CASES must remain E0 through E12")
    if step_ids[0] != "E0" or step_ids[-1] != "E12":
        raise P1SuiteAuthorizationError("suite order must start at E0 and end at E12")
    if set(step_ids) != set(case_ids):
        raise P1SuiteAuthorizationError("suite must cover every E0-E12 case exactly")
    return {
        "authorized_live_run_required": AUTHORIZED_LIVE_P1_SUITE_RUN_VALUE,
        "calibration_only": True,
        "case_timeout_seconds": DEFAULT_CASE_TIMEOUT_SECONDS,
        "execution_mode_required": EXECUTION_MODE_AUTHORIZED_LIVE_P1_SUITE,
        "gradle_authorized": False,
        "hard_gate_conditions": [
            "complete ordered E0-E12 suite including E7 water/lava and E9 water/lava",
            "every required case success",
            "truth_missing_count==0 where requires_server_truth",
            "no explicit cleanup failure",
            "OS process_release_proven for every case",
            "real_execution_performed",
            "no human intervention",
        ],
        "human_intervention": False,
        "integration_verified": False,
        "p1_hard_gate_passed": False,
        "p1_validation_manifest_status": "not_run",
        "process_release_proven": False,
        "real_execution_performed": False,
        "status": "ok",
        "step_keys": [step.step_key for step in steps],
        "steps": [
            {
                "authorized_live_run": step.authorized_live_run,
                "check_id": step.check_id,
                "execution_mode": step.execution_mode,
                "module": step.module,
                "name": step.name,
                "required_runtime": required_runtime(step.check_id)[0],
                "required_runtime_sha256": required_runtime(step.check_id)[1],
                "requires_server_truth": step.requires_server_truth,
                "runs_root_relative": step.runs_root_relative,
                "variant": step.variant,
            }
            for step in steps
        ],
        "verification_level": UNIT_VERIFIED,
    }


def preflight_authorized_p1_suite(
    *,
    execution_mode: str,
    authorized_live_run: str,
    output_dir: Path | None = None,
    allow_gradle: bool = False,
) -> dict[str, Any]:
    assert_p1_suite_live_authorized(
        execution_mode=execution_mode,
        authorized_live_run=authorized_live_run,
        allow_gradle=allow_gradle,
    )
    payload = check_p1_suite()
    payload.update(
        {
            "authorized_live_run": AUTHORIZED_LIVE_P1_SUITE_RUN_VALUE,
            "execution_mode": EXECUTION_MODE_AUTHORIZED_LIVE_P1_SUITE,
            "pilot_command": (
                "conda run -n mc-agent python -m obsidianlink.env.integration.p1_suite "
                "--execution-mode authorized_live_p1_suite "
                "--authorized-live-run p1_e0_e12_validation_suite "
                "--output-dir runs/p1_validation_suite/<unique-pilot-id>"
            ),
        }
    )
    if output_dir is not None:
        payload["output_dir"] = str(_validate_output_dir(output_dir))
    return payload


def _payload_from_result(result: object) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return dict(result)
    as_dict = getattr(result, "as_dict", None)
    if callable(as_dict):
        payload = as_dict()
        if isinstance(payload, Mapping):
            return dict(payload)
    raise P1SuiteAuthorizationError("case runner did not return a mapping record")


def execute_existing_case_runner(
    step: P1SuiteStep,
    *,
    output_dir: Path,
    episode_id: str,
) -> dict[str, Any]:
    """Call the existing authorized E-case runner in-process. Tests inject this."""

    module = importlib.import_module(step.module)
    runner = getattr(module, f"run_authorized_{step.check_id.lower()}_minerl")
    result = runner(
        execution_mode=step.execution_mode,
        authorized_live_run=step.authorized_live_run,
        output_dir=output_dir,
        episode_id=episode_id,
        allow_gradle=False,
        preflight_only=False,
    )
    return _payload_from_result(result)


def execute_case_subprocess(
    step: P1SuiteStep,
    *,
    output_dir: Path,
    episode_id: str,
    timeout_seconds: float = DEFAULT_CASE_TIMEOUT_SECONDS,
) -> tuple[dict[str, Any], ProcessReleaseStatus]:
    """Launch one existing case CLI in a fresh process and inspect OS release."""

    command = [
        sys.executable,
        "-m",
        step.module,
        "--execution-mode",
        step.execution_mode,
        "--authorized-live-run",
        step.authorized_live_run,
        "--output-dir",
        str(output_dir),
        "--episode-id",
        episode_id,
    ]
    stdout_path = output_dir.parent / f"{output_dir.name}.stdout.log"
    stderr_path = output_dir.parent / f"{output_dir.name}.stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    tracked: dict[int, str] = {}
    timed_out = False
    with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout_seconds
        while process.poll() is None and time.monotonic() < deadline:
            merge_tracked_descendants(tracked, tracked_descendants(process.pid))
            time.sleep(0.25)
        if process.poll() is None:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        exit_code = process.returncode
    residual: dict[int, str] = dict(tracked)
    release_deadline = time.monotonic() + PROCESS_RELEASE_WAIT_SECONDS
    while time.monotonic() < release_deadline:
        residual = residual_descendants(tracked)
        if not residual:
            break
        time.sleep(0.25)
    release = inspect_os_process_release(
        tracked_children=[
            {"pid": pid, "command": command_text}
            for pid, command_text in sorted(tracked.items())
        ],
        residual_children=[
            {"pid": pid, "command": command_text}
            for pid, command_text in sorted(residual.items())
        ],
        subprocess_exited=exit_code is not None,
    )
    payload: dict[str, Any] = {
        "success": False,
        "outcome": "timeout" if timed_out else "runtime_error",
        "real_execution_performed": False,
        "subprocess_exit_code": exit_code,
    }
    if output_dir.is_dir():
        for path in output_dir.glob("*.json"):
            if path.name == "authorization.json":
                continue
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if isinstance(loaded, Mapping) and "success" in loaded:
                payload = dict(loaded)
                payload["subprocess_exit_code"] = exit_code
                break
    return payload, release


def _write_suite_evidence(result: P1SuiteResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "p1_suite.json"
    path.write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "authorization.json").write_text(
        json.dumps(
            {
                "authorized_live_run": AUTHORIZED_LIVE_P1_SUITE_RUN_VALUE,
                "catalog_live_run_allowed_remains_false": True,
                "execution_mode": EXECUTION_MODE_AUTHORIZED_LIVE_P1_SUITE,
                "gradle_authorized": False,
                "integration_verified": False,
                "model_api_authorized": False,
                "p1_hard_gate_passed": result.p1_hard_gate_passed,
                "p1_validation_manifest_status": "not_run",
                "note": (
                    "Per-run suite authorization only. Does not mark E0-E12 "
                    "integration_verified. p1_validation_manifest() remains not_run."
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


ExecuteStep = Callable[[P1SuiteStep, Path, str], tuple[Mapping[str, Any], ProcessReleaseStatus]]


def run_authorized_p1_suite(
    *,
    execution_mode: str,
    authorized_live_run: str,
    output_dir: Path,
    allow_gradle: bool = False,
    preflight_only: bool = False,
    execute_step: ExecuteStep | None = None,
    case_timeout_seconds: float = DEFAULT_CASE_TIMEOUT_SECONDS,
) -> P1SuiteResult | dict[str, Any]:
    preflight = preflight_authorized_p1_suite(
        execution_mode=execution_mode,
        authorized_live_run=authorized_live_run,
        output_dir=output_dir,
        allow_gradle=allow_gradle,
    )
    if preflight_only:
        return preflight
    global _PROCESS_LIVE_RUN_STARTED
    with _PROCESS_LIVE_RUN_LOCK:
        if _PROCESS_LIVE_RUN_STARTED:
            raise P1SuiteAuthorizationError("authorized P1 suite allows one real run per process")
        _PROCESS_LIVE_RUN_STARTED = True

    def default_execute(
        step: P1SuiteStep, suite_dir: Path, episode_id: str
    ) -> tuple[Mapping[str, Any], ProcessReleaseStatus]:
        case_root = (ROOT / "runs" / step.runs_root_relative).resolve()
        case_root.mkdir(parents=True, exist_ok=True)
        case_output = case_root / f"{suite_dir.name}-{step.step_key.replace(':', '-').lower()}"
        return execute_case_subprocess(
            step,
            output_dir=case_output,
            episode_id=episode_id,
            timeout_seconds=case_timeout_seconds,
        )

    runner = execute_step if execute_step is not None else default_execute
    steps = p1_suite_steps()
    cases: list[P1CaseSummary] = []
    stopped_after: str | None = None
    for step in steps:
        episode_id = f"p1-suite-{step.step_key.replace(':', '-').lower()}"
        try:
            runtime = activate_required_runtime(step.check_id)
        except P1SuiteRuntimeError as error:
            cases.append(
                P1CaseSummary(
                    check_id=step.check_id,
                    name=step.name,
                    variant=step.variant,
                    success=False,
                    outcome="runtime_not_verified",
                    requires_server_truth=step.requires_server_truth,
                    truth_missing_count=None,
                    cleanup_failed=False,
                    process_release=inspect_os_process_release(subprocess_exited=False),
                    real_execution_performed=False,
                    runtime=failed_runtime_record(step.check_id, error),
                )
            )
            stopped_after = step.step_key
            break
        payload, release = runner(step, output_dir, episode_id)
        summary = case_summary_from_payload(payload, step, release, runtime=runtime)
        cases.append(summary)
        issue = _case_issue(summary)
        if issue is not None:
            stopped_after = step.step_key
            break
    real_execution = bool(cases) and all(case.real_execution_performed for case in cases)
    result = aggregate_p1_suite(
        cases,
        steps=steps,
        real_execution_performed=real_execution,
        human_intervention=False,
        stopped_after=stopped_after,
    )
    _write_suite_evidence(result, _validate_output_dir(output_dir))
    return result


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="obsidianlink.env.integration.p1_suite",
        description=(
            "Authorized P1 E0-E12 validation suite entrypoint; --check is offline-safe."
        ),
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execution-mode")
    parser.add_argument("--authorized-live-run")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        if (
            any(value is not None for value in (args.execution_mode, args.authorized_live_run, args.output_dir))
            or args.preflight_only
        ):
            raise P1SuiteAuthorizationError("--check cannot be combined with live arguments")
        payload: Mapping[str, Any] = check_p1_suite()
    else:
        if args.output_dir is None:
            raise P1SuiteAuthorizationError("--output-dir is required for P1 suite live/preflight")
        result = run_authorized_p1_suite(
            execution_mode=args.execution_mode,
            authorized_live_run=args.authorized_live_run,
            output_dir=args.output_dir,
            preflight_only=args.preflight_only,
        )
        payload = result if isinstance(result, Mapping) else result.as_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

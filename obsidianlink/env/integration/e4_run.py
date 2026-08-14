"""Explicitly authorized P1 E4 real-MineRL camera validation entrypoint.

Imports and ``--check`` are offline-safe. The live path resolves the
production backend only after the exact gate, performs one bounded look, and
records evaluator-only before/after orientation evidence. It never runs
Gradle, a model, a solver, movement, placement, or later validation cases.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from obsidianlink.core.task_catalog import load_task_catalog
from obsidianlink.env.integration.e0_adapter import BACKEND_IDENTITY
from obsidianlink.env.integration.e0_cleanup import E0CleanupStatus
from obsidianlink.env.integration.e4_adapter import MineRLE4CameraAdapter
from obsidianlink.env.integration.e4_config import (
    E4_AGENT_ID,
    E4_COMPATIBILITY_INVENTORY,
    E4_PITCH_TOLERANCE,
    E4_REQUESTED_PITCH,
    E4_REQUESTED_YAW,
    E4_YAW_TOLERANCE,
    build_e4_compatibility_task,
)
from obsidianlink.env.validation import E4_CAMERA_CASE, EnvironmentValidationRunner
from obsidianlink.env.validation.contract import EnvironmentValidationId
from obsidianlink.env.validation.result import EnvironmentValidationResult


ROOT = Path(__file__).resolve().parents[3]
FORMAL_E4_RUNS_ROOT = (ROOT / "runs" / "p1_e4_camera_control").resolve()
EXECUTION_MODE_AUTHORIZED_LIVE_E4 = "authorized_live_e4"
AUTHORIZED_LIVE_E4_RUN_VALUE = "e4_camera_control"

_PROCESS_LIVE_RUN_STARTED = False
_PROCESS_LIVE_RUN_LOCK = threading.Lock()


class E4AuthorizationError(ValueError):
    """Raised when the exact E4 live gate or preflight is invalid."""


@dataclass(frozen=True)
class E4MineRLRunRecord:
    execution_mode: str
    authorized_live_run: str
    backend_identity: str
    opened: bool
    authorization_accepted: bool
    real_execution_performed: bool
    lifecycle: EnvironmentValidationResult
    cleanup: E0CleanupStatus

    def __post_init__(self) -> None:
        if self.execution_mode != EXECUTION_MODE_AUTHORIZED_LIVE_E4:
            raise ValueError("execution_mode must be authorized_live_e4")
        if self.authorized_live_run != AUTHORIZED_LIVE_E4_RUN_VALUE:
            raise ValueError("authorized_live_run must be e4_camera_control")
        if not isinstance(self.lifecycle, EnvironmentValidationResult) or self.lifecycle.check_id is not EnvironmentValidationId.E4:
            raise ValueError("lifecycle must be an E4 validation result")
        if not isinstance(self.cleanup, E0CleanupStatus):
            raise ValueError("cleanup must be E0CleanupStatus")
        if any(type(value) is not bool for value in (self.opened, self.authorization_accepted, self.real_execution_performed)):
            raise ValueError("record flags must be bool")

    @property
    def success(self) -> bool:
        return self.lifecycle.success and not self.cleanup.has_explicit_failure()

    @property
    def outcome(self) -> str:
        if self.lifecycle.success and self.cleanup.has_explicit_failure():
            return "cleanup_failed"
        return self.lifecycle.outcome

    def as_dict(self) -> dict[str, Any]:
        payload = self.lifecycle.as_dict()
        if self.lifecycle.success and self.cleanup.has_explicit_failure():
            payload["error"] = self.cleanup.failure_detail()
        payload.update(
            {
                "authorization_accepted": self.authorization_accepted,
                "authorized_live_run": self.authorized_live_run,
                "backend_identity": self.backend_identity,
                "cleanup": self.cleanup.as_dict(),
                "execution_mode": self.execution_mode,
                "opened": self.opened,
                "outcome": self.outcome,
                "real_execution_performed": self.real_execution_performed,
                "success": self.success,
            }
        )
        return payload


def reset_authorized_e4_process_guards_for_tests() -> None:
    global _PROCESS_LIVE_RUN_STARTED
    with _PROCESS_LIVE_RUN_LOCK:
        _PROCESS_LIVE_RUN_STARTED = False


def assert_e4_live_authorized(*, execution_mode: object, authorized_live_run: object, allow_gradle: object = False) -> None:
    if execution_mode != EXECUTION_MODE_AUTHORIZED_LIVE_E4:
        raise E4AuthorizationError("execution_mode must be exactly authorized_live_e4")
    if authorized_live_run != AUTHORIZED_LIVE_E4_RUN_VALUE:
        raise E4AuthorizationError("authorized_live_run must be exactly e4_camera_control")
    if allow_gradle is not False:
        raise E4AuthorizationError("Gradle is not authorized for E4")


def _validate_catalog_policy() -> None:
    catalog = load_task_catalog(ROOT / "benchmark/catalog/tasks.json")
    if catalog.active_phase != "P1-REAL-MINERL-ENVIRONMENT-VALIDATION":
        raise E4AuthorizationError("active catalog phase must remain P1")
    if any(entry.live_run_allowed for entry in catalog.entries):
        raise E4AuthorizationError("catalog live_run_allowed must remain false")


def _validate_configuration() -> None:
    if not (
        E4_CAMERA_CASE.check_id is EnvironmentValidationId.E4
        and E4_CAMERA_CASE.requires_server_truth
        and E4_CAMERA_CASE.calibration_only
        and E4_REQUESTED_YAW == 20.0
        and E4_REQUESTED_PITCH == 0.0
        and E4_YAW_TOLERANCE == 1.0
        and E4_PITCH_TOLERANCE == 1.0
    ):
        raise E4AuthorizationError("frozen E4 calibration differs")
    task = build_e4_compatibility_task("p1-e4-preflight")
    if dict(task.initial_inventories[E4_AGENT_ID]) != E4_COMPATIBILITY_INVENTORY:
        raise E4AuthorizationError("E4 compatibility inventory differs")
    adapter = MineRLE4CameraAdapter(episode_id="p1-e4-preflight")
    if adapter._backend is not None:
        raise E4AuthorizationError("E4 adapter construction created a backend")


def check_e4_live_runner() -> dict[str, Any]:
    _validate_catalog_policy()
    _validate_configuration()
    return {
        "authorized_live_run_required": AUTHORIZED_LIVE_E4_RUN_VALUE,
        "calibration_only": True,
        "check_id": "E4",
        "execution_mode_required": EXECUTION_MODE_AUTHORIZED_LIVE_E4,
        "gradle_authorized": False,
        "integration_verified": False,
        "name": "camera_control",
        "production_backend_constructed": False,
        "real_execution_performed": False,
        "requested_pitch": E4_REQUESTED_PITCH,
        "requested_yaw": E4_REQUESTED_YAW,
        "status": "ok",
        "verification_level": "unit_verified",
    }


def _validate_output_dir(output_dir: Path) -> Path:
    if not isinstance(output_dir, Path) or not output_dir.is_absolute():
        raise E4AuthorizationError("output_dir must be an absolute pathlib.Path")
    resolved = output_dir.resolve()
    if resolved.exists() or resolved.is_symlink():
        raise E4AuthorizationError(f"output_dir must not exist: {resolved}")
    try:
        resolved.relative_to(FORMAL_E4_RUNS_ROOT)
    except ValueError as exc:
        raise E4AuthorizationError(f"output_dir must be under {FORMAL_E4_RUNS_ROOT}") from exc
    if resolved == FORMAL_E4_RUNS_ROOT or resolved.parent != FORMAL_E4_RUNS_ROOT:
        raise E4AuthorizationError("output_dir must be a unique direct child")
    return resolved


def _production_backend_cls() -> type:
    from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
    return MineRLEnvironmentBackend


def preflight_authorized_e4(*, execution_mode: str, authorized_live_run: str, output_dir: Path | None = None, allow_gradle: bool = False) -> dict[str, Any]:
    assert_e4_live_authorized(execution_mode=execution_mode, authorized_live_run=authorized_live_run, allow_gradle=allow_gradle)
    _validate_catalog_policy()
    _validate_configuration()
    payload = check_e4_live_runner()
    payload.update({"execution_mode": execution_mode, "authorized_live_run": authorized_live_run, "requires_server_truth": True})
    if output_dir is not None:
        payload["output_dir"] = str(_validate_output_dir(output_dir))
    return payload


def _write_evidence(record: E4MineRLRunRecord, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "e4_camera_control.json"
    path.write_text(json.dumps(record.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "authorization.json").write_text(
        json.dumps(
            {
                "authorized_live_run": AUTHORIZED_LIVE_E4_RUN_VALUE,
                "catalog_live_run_allowed_remains_false": True,
                "execution_mode": EXECUTION_MODE_AUTHORIZED_LIVE_E4,
                "gradle_authorized": False,
                "integration_verified": False,
                "model_api_authorized": False,
                "tested_camera_action_count": 1,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return path


def run_authorized_e4_minerl(*, execution_mode: str, authorized_live_run: str, output_dir: Path, episode_id: str = "p1-e4-camera-control", allow_gradle: bool = False, preflight_only: bool = False) -> E4MineRLRunRecord | dict[str, Any]:
    preflight = preflight_authorized_e4(execution_mode=execution_mode, authorized_live_run=authorized_live_run, output_dir=output_dir, allow_gradle=allow_gradle)
    if preflight_only:
        return preflight
    global _PROCESS_LIVE_RUN_STARTED
    with _PROCESS_LIVE_RUN_LOCK:
        if _PROCESS_LIVE_RUN_STARTED:
            raise E4AuthorizationError("authorized E4 allows one real run per process")
        _PROCESS_LIVE_RUN_STARTED = True
    backend_cls = _production_backend_cls()
    factory = MineRLE4CameraAdapter.lifecycle_factory(episode_id=episode_id, backend_cls=backend_cls, backend_kwargs={"max_reset_attempts": 1})
    holder: dict[str, MineRLE4CameraAdapter | None] = {"adapter": None}

    def capture() -> MineRLE4CameraAdapter:
        adapter = factory()
        holder["adapter"] = adapter
        return adapter

    lifecycle = EnvironmentValidationRunner().run(
        E4_CAMERA_CASE,
        capture,
        episode_id=episode_id,
        requested_yaw=E4_REQUESTED_YAW,
        requested_pitch=E4_REQUESTED_PITCH,
        yaw_tolerance=E4_YAW_TOLERANCE,
        pitch_tolerance=E4_PITCH_TOLERANCE,
    )
    adapter = holder["adapter"]
    cleanup = adapter.cleanup_status() if adapter is not None else E0CleanupStatus(lifecycle.closed, lifecycle.closed, lifecycle.closed, lifecycle.closed)
    record = E4MineRLRunRecord(
        execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E4,
        authorized_live_run=AUTHORIZED_LIVE_E4_RUN_VALUE,
        backend_identity=getattr(adapter, "backend_identity", backend_cls.__name__),
        opened=False if adapter is None else adapter.open_succeeded,
        authorization_accepted=True,
        real_execution_performed=backend_cls.__name__ == BACKEND_IDENTITY,
        lifecycle=lifecycle,
        cleanup=cleanup,
    )
    _write_evidence(record, _validate_output_dir(output_dir))
    return record


def build_parser():
    import argparse
    parser = argparse.ArgumentParser(prog="obsidianlink.env.integration.e4_run", description="Authorized P1 E4 camera-control entrypoint; --check is offline-safe.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execution-mode")
    parser.add_argument("--authorized-live-run")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episode-id", default="p1-e4-camera-control")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        if any(value is not None for value in (args.execution_mode, args.authorized_live_run, args.output_dir)) or args.preflight_only:
            raise E4AuthorizationError("--check cannot be combined with live arguments")
        payload: Mapping[str, Any] = check_e4_live_runner()
    else:
        if args.output_dir is None:
            raise E4AuthorizationError("--output-dir is required for E4 live/preflight")
        result = run_authorized_e4_minerl(execution_mode=args.execution_mode, authorized_live_run=args.authorized_live_run, output_dir=args.output_dir, episode_id=args.episode_id, preflight_only=args.preflight_only)
        payload = result if isinstance(result, Mapping) else result.as_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

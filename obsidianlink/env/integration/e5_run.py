"""Explicitly authorized P1 E5 real-MineRL movement validation entrypoint.

Imports and ``--check`` are offline-safe. The live path resolves production
MineRL only after the exact gate and performs exactly one bounded move. It
never runs Gradle, models, solvers, placement, or later validation cases.
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
from obsidianlink.env.integration.e5_adapter import MineRLE5MovementAdapter
from obsidianlink.env.integration.e5_config import (
    E5_AGENT_ID,
    E5_COMPATIBILITY_INVENTORY,
    E5_DURATION_TICKS,
    E5_FORWARD,
    E5_JUMP,
    E5_MAX_HORIZONTAL_DISTANCE,
    E5_MAX_LATERAL_DRIFT,
    E5_MAX_VERTICAL_DRIFT,
    E5_MIN_FORWARD_PROJECTION,
    E5_MIN_HORIZONTAL_DISTANCE,
    E5_SPRINT,
    E5_STRAFE,
    build_e5_compatibility_task,
)
from obsidianlink.env.validation import E5_MOVEMENT_CASE, EnvironmentValidationRunner
from obsidianlink.env.validation.contract import EnvironmentValidationId
from obsidianlink.env.validation.result import EnvironmentValidationResult


ROOT = Path(__file__).resolve().parents[3]
FORMAL_E5_RUNS_ROOT = (ROOT / "runs" / "p1_e5_movement").resolve()
EXECUTION_MODE_AUTHORIZED_LIVE_E5 = "authorized_live_e5"
AUTHORIZED_LIVE_E5_RUN_VALUE = "e5_movement"

_PROCESS_LIVE_RUN_STARTED = False
_PROCESS_LIVE_RUN_LOCK = threading.Lock()


class E5AuthorizationError(ValueError):
    """Raised when the exact E5 live gate or preflight is invalid."""


@dataclass(frozen=True)
class E5MineRLRunRecord:
    execution_mode: str
    authorized_live_run: str
    backend_identity: str
    opened: bool
    authorization_accepted: bool
    real_execution_performed: bool
    lifecycle: EnvironmentValidationResult
    cleanup: E0CleanupStatus

    def __post_init__(self) -> None:
        if self.execution_mode != EXECUTION_MODE_AUTHORIZED_LIVE_E5:
            raise ValueError("execution_mode must be authorized_live_e5")
        if self.authorized_live_run != AUTHORIZED_LIVE_E5_RUN_VALUE:
            raise ValueError("authorized_live_run must be e5_movement")
        if not isinstance(self.lifecycle, EnvironmentValidationResult) or self.lifecycle.check_id is not EnvironmentValidationId.E5:
            raise ValueError("lifecycle must be an E5 validation result")
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


def reset_authorized_e5_process_guards_for_tests() -> None:
    global _PROCESS_LIVE_RUN_STARTED
    with _PROCESS_LIVE_RUN_LOCK:
        _PROCESS_LIVE_RUN_STARTED = False


def assert_e5_live_authorized(*, execution_mode: object, authorized_live_run: object, allow_gradle: object = False) -> None:
    if execution_mode != EXECUTION_MODE_AUTHORIZED_LIVE_E5:
        raise E5AuthorizationError("execution_mode must be exactly authorized_live_e5")
    if authorized_live_run != AUTHORIZED_LIVE_E5_RUN_VALUE:
        raise E5AuthorizationError("authorized_live_run must be exactly e5_movement")
    if allow_gradle is not False:
        raise E5AuthorizationError("Gradle is not authorized for E5")


def _validate_catalog_policy() -> None:
    catalog = load_task_catalog(ROOT / "benchmark/catalog/tasks.json")
    if catalog.active_phase != "P1-REAL-MINERL-ENVIRONMENT-VALIDATION":
        raise E5AuthorizationError("active catalog phase must remain P1")
    if any(entry.live_run_allowed for entry in catalog.entries):
        raise E5AuthorizationError("catalog live_run_allowed must remain false")


def _validate_configuration() -> None:
    if not (
        E5_MOVEMENT_CASE.check_id is EnvironmentValidationId.E5
        and E5_MOVEMENT_CASE.requires_server_truth
        and E5_MOVEMENT_CASE.calibration_only
        and (E5_FORWARD, E5_STRAFE, E5_SPRINT, E5_JUMP, E5_DURATION_TICKS) == (1.0, 0.0, False, False, 1)
        and (E5_MIN_HORIZONTAL_DISTANCE, E5_MIN_FORWARD_PROJECTION, E5_MAX_LATERAL_DRIFT, E5_MAX_HORIZONTAL_DISTANCE, E5_MAX_VERTICAL_DRIFT) == (0.02, 0.02, 0.02, 0.5, 0.25)
    ):
        raise E5AuthorizationError("frozen E5 calibration differs")
    task = build_e5_compatibility_task("p1-e5-preflight")
    if dict(task.initial_inventories[E5_AGENT_ID]) != E5_COMPATIBILITY_INVENTORY:
        raise E5AuthorizationError("E5 compatibility inventory differs")
    if task.spawn_positions[E5_AGENT_ID] != (0, 4, 0):
        raise E5AuthorizationError("E5 flat-ground spawn differs")
    if MineRLE5MovementAdapter(episode_id="p1-e5-preflight")._backend is not None:
        raise E5AuthorizationError("E5 adapter construction created a backend")


def check_e5_live_runner() -> dict[str, Any]:
    _validate_catalog_policy()
    _validate_configuration()
    return {
        "authorized_live_run_required": AUTHORIZED_LIVE_E5_RUN_VALUE,
        "calibration_only": True,
        "check_id": "E5",
        "execution_mode_required": EXECUTION_MODE_AUTHORIZED_LIVE_E5,
        "gradle_authorized": False,
        "integration_verified": False,
        "name": "movement",
        "production_backend_constructed": False,
        "real_execution_performed": False,
        "requested_duration_ticks": E5_DURATION_TICKS,
        "requested_forward": E5_FORWARD,
        "requested_jump": E5_JUMP,
        "requested_sprint": E5_SPRINT,
        "requested_strafe": E5_STRAFE,
        "status": "ok",
        "verification_level": "unit_verified",
    }


def _validate_output_dir(output_dir: Path) -> Path:
    if not isinstance(output_dir, Path) or not output_dir.is_absolute():
        raise E5AuthorizationError("output_dir must be an absolute pathlib.Path")
    resolved = output_dir.resolve()
    if resolved.exists() or resolved.is_symlink():
        raise E5AuthorizationError(f"output_dir must not exist: {resolved}")
    try:
        resolved.relative_to(FORMAL_E5_RUNS_ROOT)
    except ValueError as exc:
        raise E5AuthorizationError(f"output_dir must be under {FORMAL_E5_RUNS_ROOT}") from exc
    if resolved == FORMAL_E5_RUNS_ROOT or resolved.parent != FORMAL_E5_RUNS_ROOT:
        raise E5AuthorizationError("output_dir must be a unique direct child")
    return resolved


def _production_backend_cls() -> type:
    from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
    return MineRLEnvironmentBackend


def preflight_authorized_e5(*, execution_mode: str, authorized_live_run: str, output_dir: Path | None = None, allow_gradle: bool = False) -> dict[str, Any]:
    assert_e5_live_authorized(execution_mode=execution_mode, authorized_live_run=authorized_live_run, allow_gradle=allow_gradle)
    _validate_catalog_policy()
    _validate_configuration()
    payload = check_e5_live_runner()
    payload.update({"execution_mode": execution_mode, "authorized_live_run": authorized_live_run, "requires_server_truth": True})
    if output_dir is not None:
        payload["output_dir"] = str(_validate_output_dir(output_dir))
    return payload


def _write_evidence(record: E5MineRLRunRecord, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "e5_movement.json"
    path.write_text(json.dumps(record.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "authorization.json").write_text(
        json.dumps(
            {
                "authorized_live_run": AUTHORIZED_LIVE_E5_RUN_VALUE,
                "catalog_live_run_allowed_remains_false": True,
                "execution_mode": EXECUTION_MODE_AUTHORIZED_LIVE_E5,
                "gradle_authorized": False,
                "integration_verified": False,
                "model_api_authorized": False,
                "tested_movement_action_count": 1,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return path


def run_authorized_e5_minerl(*, execution_mode: str, authorized_live_run: str, output_dir: Path, episode_id: str = "p1-e5-movement", allow_gradle: bool = False, preflight_only: bool = False) -> E5MineRLRunRecord | dict[str, Any]:
    preflight = preflight_authorized_e5(execution_mode=execution_mode, authorized_live_run=authorized_live_run, output_dir=output_dir, allow_gradle=allow_gradle)
    if preflight_only:
        return preflight
    global _PROCESS_LIVE_RUN_STARTED
    with _PROCESS_LIVE_RUN_LOCK:
        if _PROCESS_LIVE_RUN_STARTED:
            raise E5AuthorizationError("authorized E5 allows one real run per process")
        _PROCESS_LIVE_RUN_STARTED = True
    backend_cls = _production_backend_cls()
    factory = MineRLE5MovementAdapter.lifecycle_factory(episode_id=episode_id, backend_cls=backend_cls, backend_kwargs={"max_reset_attempts": 1})
    holder: dict[str, MineRLE5MovementAdapter | None] = {"adapter": None}

    def capture() -> MineRLE5MovementAdapter:
        adapter = factory()
        holder["adapter"] = adapter
        return adapter

    lifecycle = EnvironmentValidationRunner().run(
        E5_MOVEMENT_CASE,
        capture,
        episode_id=episode_id,
        requested_forward=E5_FORWARD,
        requested_strafe=E5_STRAFE,
        requested_sprint=E5_SPRINT,
        requested_jump=E5_JUMP,
        requested_duration_ticks=E5_DURATION_TICKS,
        minimum_horizontal_distance=E5_MIN_HORIZONTAL_DISTANCE,
        minimum_forward_projection=E5_MIN_FORWARD_PROJECTION,
        maximum_lateral_drift=E5_MAX_LATERAL_DRIFT,
        maximum_horizontal_distance=E5_MAX_HORIZONTAL_DISTANCE,
        maximum_vertical_drift=E5_MAX_VERTICAL_DRIFT,
    )
    adapter = holder["adapter"]
    cleanup = adapter.cleanup_status() if adapter is not None else E0CleanupStatus(lifecycle.closed, lifecycle.closed, lifecycle.closed, lifecycle.closed)
    record = E5MineRLRunRecord(
        execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E5,
        authorized_live_run=AUTHORIZED_LIVE_E5_RUN_VALUE,
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
    parser = argparse.ArgumentParser(prog="obsidianlink.env.integration.e5_run", description="Authorized P1 E5 movement entrypoint; --check is offline-safe.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execution-mode")
    parser.add_argument("--authorized-live-run")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episode-id", default="p1-e5-movement")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        if any(value is not None for value in (args.execution_mode, args.authorized_live_run, args.output_dir)) or args.preflight_only:
            raise E5AuthorizationError("--check cannot be combined with live arguments")
        payload: Mapping[str, Any] = check_e5_live_runner()
    else:
        if args.output_dir is None:
            raise E5AuthorizationError("--output-dir is required for E5 live/preflight")
        result = run_authorized_e5_minerl(execution_mode=args.execution_mode, authorized_live_run=args.authorized_live_run, output_dir=args.output_dir, episode_id=args.episode_id, preflight_only=args.preflight_only)
        payload = result if isinstance(result, Mapping) else result.as_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

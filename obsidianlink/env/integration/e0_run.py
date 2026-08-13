"""Authorized P1 E0 real-MineRL entrypoint.

This module never starts MineRL on import, during ``python -m obsidianlink
--check``, or during ordinary unit tests. A later authorized command must
supply the explicit live flags before any backend is constructed.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from obsidianlink.core.task_catalog import load_task_catalog
from obsidianlink.env.integration.e0_adapter import (
    BACKEND_IDENTITY,
    MineRLE0LifecycleAdapter,
)
from obsidianlink.env.integration.e0_cleanup import E0CleanupStatus
from obsidianlink.env.validation import (
    E0_LIFECYCLE_CASE,
    EnvironmentValidationRunner,
)
from obsidianlink.env.validation.contract import EnvironmentValidationId
from obsidianlink.env.validation.result import UNIT_VERIFIED, EnvironmentValidationResult


ROOT = Path(__file__).resolve().parents[3]
FORMAL_E0_RUNS_ROOT = (ROOT / "runs" / "p1_e0_reset_close").resolve()
EXECUTION_MODE_AUTHORIZED_LIVE_E0 = "authorized_live_e0"
AUTHORIZED_LIVE_RUN_VALUE = "e0_reset_close"

_PROCESS_LIVE_RUN_STARTED = False
_PROCESS_LIVE_RUN_LOCK = threading.Lock()


class E0AuthorizationError(ValueError):
    """Raised when E0 live authorization is missing or invalid."""


def _require_bool(value: object, field_name: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be bool")


def _require_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class E0MineRLRunRecord:
    """Evidence for a prepared or authorized E0 MineRL lifecycle attempt.

    This type can record that a real backend path was selected. It cannot
    claim ``integration_verified``. Offline adapter tests must keep
    ``real_execution_performed=False``.
    """

    check_id: str
    name: str
    episode_id: str
    step_id: int
    backend_identity: str
    opened: bool
    created: bool
    reset_completed: bool
    initial_state_present: bool
    closed: bool
    success: bool
    outcome: str
    cleanup: E0CleanupStatus
    error: str | None = None
    close_error: str | None = None
    authorization_accepted: bool = False
    real_execution_performed: bool = False
    integration_verified: bool = False
    verification_level: str = UNIT_VERIFIED
    calibration_only: bool = True

    def __post_init__(self) -> None:
        if self.check_id != EnvironmentValidationId.E0.value:
            raise ValueError("check_id must be E0")
        _require_identifier(self.name, "name")
        _require_identifier(self.episode_id, "episode_id")
        _require_identifier(self.backend_identity, "backend_identity")
        if self.name != "reset_close":
            raise ValueError("name must be reset_close")
        if type(self.step_id) is not int or self.step_id < 0:
            raise ValueError("step_id must be a non-negative int")
        for field_name in (
            "opened",
            "created",
            "reset_completed",
            "initial_state_present",
            "closed",
            "success",
            "authorization_accepted",
            "real_execution_performed",
            "integration_verified",
            "calibration_only",
        ):
            _require_bool(getattr(self, field_name), field_name)
        if not isinstance(self.cleanup, E0CleanupStatus):
            raise ValueError("cleanup must be E0CleanupStatus")
        if self.verification_level != UNIT_VERIFIED:
            raise ValueError("this runtime may only emit unit_verified")
        if self.integration_verified:
            raise ValueError("this runtime cannot claim integration_verified")
        if not self.calibration_only:
            raise ValueError("E0 MineRL records must remain calibration-only")
        if self.success and not (
            self.opened
            and self.created
            and self.reset_completed
            and self.initial_state_present
            and self.closed
            and self.error is None
            and self.close_error is None
        ):
            raise ValueError("success requires a clean E0 MineRL lifecycle")

    def as_dict(self) -> dict[str, Any]:
        return {
            "authorization_accepted": self.authorization_accepted,
            "backend_identity": self.backend_identity,
            "calibration_only": True,
            "check_id": self.check_id,
            "cleanup": self.cleanup.as_dict(),
            "close_error": self.close_error,
            "closed": self.closed,
            "created": self.created,
            "episode_id": self.episode_id,
            "error": self.error,
            "initial_state_present": self.initial_state_present,
            "integration_verified": False,
            "name": self.name,
            "opened": self.opened,
            "outcome": self.outcome,
            "real_execution_performed": self.real_execution_performed,
            "reset_completed": self.reset_completed,
            "step_id": self.step_id,
            "success": self.success,
            "verification_level": UNIT_VERIFIED,
        }


def reset_authorized_e0_process_guards_for_tests() -> None:
    """Reset process-once guards. Intended for offline unit tests only."""

    global _PROCESS_LIVE_RUN_STARTED
    with _PROCESS_LIVE_RUN_LOCK:
        _PROCESS_LIVE_RUN_STARTED = False


def assert_e0_live_authorized(
    *,
    execution_mode: object,
    authorized_live_run: object,
    allow_gradle: object = False,
) -> None:
    if execution_mode != EXECUTION_MODE_AUTHORIZED_LIVE_E0:
        raise E0AuthorizationError(
            "execution_mode must be exactly authorized_live_e0"
        )
    if authorized_live_run != AUTHORIZED_LIVE_RUN_VALUE:
        raise E0AuthorizationError(
            "authorized_live_run must be exactly e0_reset_close"
        )
    if allow_gradle is not False:
        raise E0AuthorizationError(
            "Gradle is not authorized for E0; allow_gradle must be False"
        )


def _validate_catalog_policy() -> None:
    catalog = load_task_catalog(ROOT / "benchmark/catalog/tasks.json")
    if catalog.active_phase != "P1-REAL-MINERL-ENVIRONMENT-VALIDATION":
        raise E0AuthorizationError(
            "active catalog phase must remain P1 environment validation"
        )
    if any(entry.live_run_allowed for entry in catalog.entries):
        raise E0AuthorizationError(
            "catalog live_run_allowed must remain false; "
            "authorization is per-run only"
        )


def _validate_output_dir(output_dir: Path) -> Path:
    if not isinstance(output_dir, Path):
        raise E0AuthorizationError("output_dir must be a pathlib.Path")
    resolved = output_dir
    if not resolved.is_absolute():
        raise E0AuthorizationError("output_dir must be an absolute path")
    resolved = resolved.resolve()
    if resolved.exists() or resolved.is_symlink():
        raise E0AuthorizationError(f"output_dir must not already exist: {resolved}")
    try:
        resolved.relative_to(FORMAL_E0_RUNS_ROOT)
    except ValueError as error:
        raise E0AuthorizationError(
            f"output_dir must be under {FORMAL_E0_RUNS_ROOT}"
        ) from error
    if resolved == FORMAL_E0_RUNS_ROOT or resolved.parent != FORMAL_E0_RUNS_ROOT:
        raise E0AuthorizationError(
            "output_dir must be a unique direct child of runs/p1_e0_reset_close/"
        )
    return resolved


def _production_backend_cls() -> type:
    from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend

    return MineRLEnvironmentBackend


def _production_backend_kwargs() -> dict[str, Any]:
    return {"max_reset_attempts": 1}


def preflight_authorized_e0(
    *,
    execution_mode: str,
    authorized_live_run: str,
    allow_gradle: bool = False,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate authorization without constructing a MineRL backend."""

    assert_e0_live_authorized(
        execution_mode=execution_mode,
        authorized_live_run=authorized_live_run,
        allow_gradle=allow_gradle,
    )
    _validate_catalog_policy()
    payload: dict[str, Any] = {
        "authorized_live_run": AUTHORIZED_LIVE_RUN_VALUE,
        "execution_mode": EXECUTION_MODE_AUTHORIZED_LIVE_E0,
        "gradle_authorized": False,
        "integration_verified": False,
        "live_run_allowed_catalog": False,
        "real_execution_performed": False,
        "verification_level": UNIT_VERIFIED,
    }
    if output_dir is not None:
        payload["output_dir"] = str(_validate_output_dir(output_dir))
    return payload


def _write_evidence(record: E0MineRLRunRecord, output_dir: Path) -> Path:
    if record.integration_verified:
        raise E0AuthorizationError("refusing to persist integration_verified")
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "e0_lifecycle.json"
    path.write_text(
        json.dumps(record.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    authorization_path = output_dir / "authorization.json"
    authorization_path.write_text(
        json.dumps(
            {
                "authorized_live_run": AUTHORIZED_LIVE_RUN_VALUE,
                "catalog_live_run_allowed_remains_false": True,
                "execution_mode": EXECUTION_MODE_AUTHORIZED_LIVE_E0,
                "gradle_authorized": False,
                "integration_verified": False,
                "model_api_authorized": False,
                "note": (
                    "Per-run authorization only. Does not mark E0 "
                    "integration_verified."
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


def _record_from_lifecycle(
    *,
    adapter: MineRLE0LifecycleAdapter | None,
    lifecycle: EnvironmentValidationResult,
    authorization_accepted: bool,
    real_execution_performed: bool,
    backend_identity: str,
) -> E0MineRLRunRecord:
    cleanup = (
        adapter.cleanup_status()
        if adapter is not None
        else E0CleanupStatus(
            close_returned=lifecycle.closed,
            backend_marked_closed=lifecycle.closed,
            environment_reference_cleared=lifecycle.closed,
            owner_cleared=lifecycle.closed,
        )
    )
    opened = False if adapter is None else adapter.open_succeeded
    return E0MineRLRunRecord(
        check_id=EnvironmentValidationId.E0.value,
        name="reset_close",
        episode_id=lifecycle.episode_id,
        step_id=lifecycle.step_id,
        backend_identity=backend_identity,
        opened=bool(opened),
        created=lifecycle.created,
        reset_completed=lifecycle.reset_completed,
        initial_state_present=lifecycle.initial_state_present,
        closed=lifecycle.closed,
        success=lifecycle.success,
        outcome=lifecycle.outcome,
        cleanup=cleanup,
        error=lifecycle.error,
        close_error=lifecycle.close_error,
        authorization_accepted=authorization_accepted,
        real_execution_performed=real_execution_performed,
    )


def run_authorized_e0_minerl(
    *,
    execution_mode: str,
    authorized_live_run: str,
    output_dir: Path,
    episode_id: str = "p1-e0-reset-close",
    allow_gradle: bool = False,
    preflight_only: bool = False,
) -> E0MineRLRunRecord | dict[str, Any]:
    """Run authorized E0 MineRL lifecycle, or refuse without starting MineRL."""

    preflight = preflight_authorized_e0(
        execution_mode=execution_mode,
        authorized_live_run=authorized_live_run,
        allow_gradle=allow_gradle,
        output_dir=output_dir,
    )
    if preflight_only:
        return preflight

    global _PROCESS_LIVE_RUN_STARTED
    with _PROCESS_LIVE_RUN_LOCK:
        if _PROCESS_LIVE_RUN_STARTED:
            raise E0AuthorizationError(
                "authorized E0 allows only one real run attempt per process"
            )
        _PROCESS_LIVE_RUN_STARTED = True

    backend_cls = _production_backend_cls()
    real_execution = backend_cls.__name__ == BACKEND_IDENTITY
    factory = MineRLE0LifecycleAdapter.lifecycle_factory(
        episode_id=episode_id,
        backend_cls=backend_cls,
        backend_kwargs=_production_backend_kwargs(),
    )
    holder: dict[str, MineRLE0LifecycleAdapter | None] = {"adapter": None}

    def capturing_factory() -> MineRLE0LifecycleAdapter:
        adapter = factory()
        holder["adapter"] = adapter
        return adapter

    lifecycle = EnvironmentValidationRunner().run(
        E0_LIFECYCLE_CASE,
        capturing_factory,
        episode_id=episode_id,
    )
    record = _record_from_lifecycle(
        adapter=holder["adapter"],
        lifecycle=lifecycle,
        authorization_accepted=True,
        real_execution_performed=real_execution,
        backend_identity=getattr(
            holder["adapter"], "backend_identity", backend_cls.__name__
        ),
    )
    _write_evidence(record, _validate_output_dir(output_dir))
    return record


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="obsidianlink.env.integration.e0_run",
        description=(
            "Authorized P1 E0 MineRL lifecycle entrypoint. "
            "Does not start MineRL unless live flags are supplied."
        ),
    )
    parser.add_argument(
        "--execution-mode",
        required=True,
        help="must be authorized_live_e0",
    )
    parser.add_argument(
        "--authorized-live-run",
        required=True,
        help="must be e0_reset_close",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--episode-id",
        default="p1-e0-reset-close",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate authorization without starting MineRL",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_authorized_e0_minerl(
        execution_mode=args.execution_mode,
        authorized_live_run=args.authorized_live_run,
        output_dir=args.output_dir,
        episode_id=args.episode_id,
        preflight_only=args.preflight_only,
    )
    payload = result if isinstance(result, Mapping) else result.as_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

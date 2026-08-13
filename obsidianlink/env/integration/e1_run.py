"""Authorized P1 E1 real-MineRL RGB entrypoint.

This module never starts MineRL on import, during ``python -m obsidianlink
--check``, or during ordinary unit tests. A later authorized command must
supply the explicit live flags before any backend is constructed.

The live path is reset + public RGB validation + close. It does not move
the agent, call a model, or run Gradle.
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
from obsidianlink.env.integration.e1_adapter import MineRLE1RGBAdapter
from obsidianlink.env.validation import (
    E1_RGB_CASE,
    EnvironmentValidationRunner,
)
from obsidianlink.env.validation.contract import EnvironmentValidationId
from obsidianlink.env.validation.result import UNIT_VERIFIED, EnvironmentValidationResult


ROOT = Path(__file__).resolve().parents[3]
FORMAL_E1_RUNS_ROOT = (ROOT / "runs" / "p1_e1_rgb_observation").resolve()
EXECUTION_MODE_AUTHORIZED_LIVE_E1 = "authorized_live_e1"
AUTHORIZED_LIVE_E1_RUN_VALUE = "e1_rgb_observation"

_PROCESS_LIVE_RUN_STARTED = False
_PROCESS_LIVE_RUN_LOCK = threading.Lock()


class E1AuthorizationError(ValueError):
    """Raised when E1 live authorization is missing or invalid."""


def _require_bool(value: object, field_name: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be bool")


def _require_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_int(value: object, field_name: str) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"{field_name} must be a non-negative int or None")


@dataclass(frozen=True)
class E1MineRLRunRecord:
    """Evidence for a prepared or authorized E1 MineRL RGB attempt.

    This type can record that a real backend path was selected. It cannot
    claim ``integration_verified``. Offline adapter tests must keep
    ``real_execution_performed=False``. Image pixels are not stored.
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
    rgb_present: bool | None = None
    rgb_height: int | None = None
    rgb_width: int | None = None
    rgb_channels: int | None = None
    rgb_dtype: str | None = None
    error: str | None = None
    close_error: str | None = None
    authorization_accepted: bool = False
    real_execution_performed: bool = False
    integration_verified: bool = False
    verification_level: str = UNIT_VERIFIED
    calibration_only: bool = True

    def __post_init__(self) -> None:
        if self.check_id != EnvironmentValidationId.E1.value:
            raise ValueError("check_id must be E1")
        _require_identifier(self.name, "name")
        _require_identifier(self.episode_id, "episode_id")
        _require_identifier(self.backend_identity, "backend_identity")
        if self.name != "rgb_observation":
            raise ValueError("name must be rgb_observation")
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
        if self.rgb_present is not None:
            _require_bool(self.rgb_present, "rgb_present")
        for field_name in ("rgb_height", "rgb_width", "rgb_channels"):
            _optional_int(getattr(self, field_name), field_name)
        if self.rgb_dtype is not None:
            _require_identifier(self.rgb_dtype, "rgb_dtype")
        if not isinstance(self.cleanup, E0CleanupStatus):
            raise ValueError("cleanup must be E0CleanupStatus")
        if self.verification_level != UNIT_VERIFIED:
            raise ValueError("this runtime may only emit unit_verified")
        if self.integration_verified:
            raise ValueError("this runtime cannot claim integration_verified")
        if not self.calibration_only:
            raise ValueError("E1 MineRL records must remain calibration-only")
        if self.success:
            if self.outcome != "rgb_ok":
                raise ValueError("success requires outcome rgb_ok")
            if not (
                self.opened
                and self.created
                and self.reset_completed
                and self.initial_state_present
                and self.closed
                and self.rgb_present is True
                and type(self.rgb_height) is int
                and type(self.rgb_width) is int
                and self.rgb_channels == 3
                and self.rgb_dtype == "uint8"
                and self.rgb_height > 0
                and self.rgb_width > 0
                and self.error is None
                and self.close_error is None
            ):
                raise ValueError("success requires a clean E1 MineRL RGB observation")
            if self.cleanup.has_explicit_failure():
                raise ValueError(
                    "success requires observable cleanup without explicit failure"
                )

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
            "rgb_channels": self.rgb_channels,
            "rgb_dtype": self.rgb_dtype,
            "rgb_height": self.rgb_height,
            "rgb_present": self.rgb_present,
            "rgb_width": self.rgb_width,
            "step_id": self.step_id,
            "success": self.success,
            "verification_level": UNIT_VERIFIED,
        }


def reset_authorized_e1_process_guards_for_tests() -> None:
    """Reset process-once guards. Intended for offline unit tests only."""

    global _PROCESS_LIVE_RUN_STARTED
    with _PROCESS_LIVE_RUN_LOCK:
        _PROCESS_LIVE_RUN_STARTED = False


def assert_e1_live_authorized(
    *,
    execution_mode: object,
    authorized_live_run: object,
    allow_gradle: object = False,
) -> None:
    if execution_mode != EXECUTION_MODE_AUTHORIZED_LIVE_E1:
        raise E1AuthorizationError(
            "execution_mode must be exactly authorized_live_e1"
        )
    if authorized_live_run != AUTHORIZED_LIVE_E1_RUN_VALUE:
        raise E1AuthorizationError(
            "authorized_live_run must be exactly e1_rgb_observation"
        )
    if allow_gradle is not False:
        raise E1AuthorizationError(
            "Gradle is not authorized for E1; allow_gradle must be False"
        )


def _validate_catalog_policy() -> None:
    catalog = load_task_catalog(ROOT / "benchmark/catalog/tasks.json")
    if catalog.active_phase != "P1-REAL-MINERL-ENVIRONMENT-VALIDATION":
        raise E1AuthorizationError(
            "active catalog phase must remain P1 environment validation"
        )
    if any(entry.live_run_allowed for entry in catalog.entries):
        raise E1AuthorizationError(
            "catalog live_run_allowed must remain false; "
            "authorization is per-run only"
        )


def _validate_output_dir(output_dir: Path) -> Path:
    if not isinstance(output_dir, Path):
        raise E1AuthorizationError("output_dir must be a pathlib.Path")
    resolved = output_dir
    if not resolved.is_absolute():
        raise E1AuthorizationError("output_dir must be an absolute path")
    resolved = resolved.resolve()
    if resolved.exists() or resolved.is_symlink():
        raise E1AuthorizationError(f"output_dir must not already exist: {resolved}")
    try:
        resolved.relative_to(FORMAL_E1_RUNS_ROOT)
    except ValueError as error:
        raise E1AuthorizationError(
            f"output_dir must be under {FORMAL_E1_RUNS_ROOT}"
        ) from error
    if resolved == FORMAL_E1_RUNS_ROOT or resolved.parent != FORMAL_E1_RUNS_ROOT:
        raise E1AuthorizationError(
            "output_dir must be a unique direct child of runs/p1_e1_rgb_observation/"
        )
    return resolved


def _production_backend_cls() -> type:
    from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend

    return MineRLEnvironmentBackend


def _production_backend_kwargs() -> dict[str, Any]:
    return {"max_reset_attempts": 1}


def preflight_authorized_e1(
    *,
    execution_mode: str,
    authorized_live_run: str,
    allow_gradle: bool = False,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate authorization without constructing a MineRL backend."""

    assert_e1_live_authorized(
        execution_mode=execution_mode,
        authorized_live_run=authorized_live_run,
        allow_gradle=allow_gradle,
    )
    _validate_catalog_policy()
    payload: dict[str, Any] = {
        "authorized_live_run": AUTHORIZED_LIVE_E1_RUN_VALUE,
        "execution_mode": EXECUTION_MODE_AUTHORIZED_LIVE_E1,
        "gradle_authorized": False,
        "integration_verified": False,
        "live_run_allowed_catalog": False,
        "real_execution_performed": False,
        "verification_level": UNIT_VERIFIED,
    }
    if output_dir is not None:
        payload["output_dir"] = str(_validate_output_dir(output_dir))
    return payload


def _write_evidence(record: E1MineRLRunRecord, output_dir: Path) -> Path:
    if record.integration_verified:
        raise E1AuthorizationError("refusing to persist integration_verified")
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "e1_rgb.json"
    path.write_text(
        json.dumps(record.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    authorization_path = output_dir / "authorization.json"
    authorization_path.write_text(
        json.dumps(
            {
                "authorized_live_run": AUTHORIZED_LIVE_E1_RUN_VALUE,
                "catalog_live_run_allowed_remains_false": True,
                "execution_mode": EXECUTION_MODE_AUTHORIZED_LIVE_E1,
                "gradle_authorized": False,
                "integration_verified": False,
                "model_api_authorized": False,
                "note": (
                    "Per-run authorization only. Does not mark E1 "
                    "integration_verified. RGB pixels are not stored."
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
    adapter: MineRLE1RGBAdapter | None,
    lifecycle: EnvironmentValidationResult,
    authorization_accepted: bool,
    real_execution_performed: bool,
    backend_identity: str,
) -> E1MineRLRunRecord:
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
    success = lifecycle.success
    outcome = lifecycle.outcome
    error = lifecycle.error
    if success and cleanup.has_explicit_failure():
        success = False
        outcome = "cleanup_failed"
        error = cleanup.failure_detail()
    return E1MineRLRunRecord(
        check_id=EnvironmentValidationId.E1.value,
        name="rgb_observation",
        episode_id=lifecycle.episode_id,
        step_id=lifecycle.step_id,
        backend_identity=backend_identity,
        opened=bool(opened),
        created=lifecycle.created,
        reset_completed=lifecycle.reset_completed,
        initial_state_present=lifecycle.initial_state_present,
        closed=lifecycle.closed,
        success=success,
        outcome=outcome,
        cleanup=cleanup,
        rgb_present=lifecycle.rgb_present,
        rgb_height=lifecycle.rgb_height,
        rgb_width=lifecycle.rgb_width,
        rgb_channels=lifecycle.rgb_channels,
        rgb_dtype=lifecycle.rgb_dtype,
        error=error,
        close_error=lifecycle.close_error,
        authorization_accepted=authorization_accepted,
        real_execution_performed=real_execution_performed,
    )


def run_authorized_e1_minerl(
    *,
    execution_mode: str,
    authorized_live_run: str,
    output_dir: Path,
    episode_id: str = "p1-e1-rgb-observation",
    allow_gradle: bool = False,
    preflight_only: bool = False,
) -> E1MineRLRunRecord | dict[str, Any]:
    """Run authorized E1 MineRL RGB validation, or refuse without starting MineRL."""

    preflight = preflight_authorized_e1(
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
            raise E1AuthorizationError(
                "authorized E1 allows only one real run attempt per process"
            )
        _PROCESS_LIVE_RUN_STARTED = True

    backend_cls = _production_backend_cls()
    real_execution = backend_cls.__name__ == BACKEND_IDENTITY
    factory = MineRLE1RGBAdapter.lifecycle_factory(
        episode_id=episode_id,
        backend_cls=backend_cls,
        backend_kwargs=_production_backend_kwargs(),
    )
    holder: dict[str, MineRLE1RGBAdapter | None] = {"adapter": None}

    def capturing_factory() -> MineRLE1RGBAdapter:
        adapter = factory()
        holder["adapter"] = adapter
        return adapter

    lifecycle = EnvironmentValidationRunner().run(
        E1_RGB_CASE,
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
        prog="obsidianlink.env.integration.e1_run",
        description=(
            "Authorized P1 E1 MineRL RGB observation entrypoint. "
            "Does not start MineRL unless live flags are supplied."
        ),
    )
    parser.add_argument(
        "--execution-mode",
        required=True,
        help="must be authorized_live_e1",
    )
    parser.add_argument(
        "--authorized-live-run",
        required=True,
        help="must be e1_rgb_observation",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--episode-id",
        default="p1-e1-rgb-observation",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate authorization without starting MineRL",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_authorized_e1_minerl(
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

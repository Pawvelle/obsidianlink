"""Authorized P1 E2 real-MineRL inventory entrypoint.

Import and ``--check`` are offline-safe. The production backend is resolved
only after both explicit authorization values pass and the caller requests
execution rather than preflight. The live path performs reset, public
inventory validation, and close only; it never calls actions, models, Gradle,
or evaluator truth.
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
from obsidianlink.env.integration.e2_adapter import MineRLE2InventoryAdapter
from obsidianlink.env.integration.e2_config import (
    E2_AGENT_ID,
    E2_CALIBRATION_INVENTORY,
    build_e2_compatibility_task,
)
from obsidianlink.env.validation import (
    E2_INVENTORY_CASE,
    EnvironmentValidationRunner,
)
from obsidianlink.env.validation.contract import EnvironmentValidationId
from obsidianlink.env.validation.inventory import inspect_inventory
from obsidianlink.env.validation.result import (
    UNIT_VERIFIED,
    EnvironmentValidationResult,
)


ROOT = Path(__file__).resolve().parents[3]
FORMAL_E2_RUNS_ROOT = (
    ROOT / "runs" / "p1_e2_inventory_observation"
).resolve()
EXECUTION_MODE_AUTHORIZED_LIVE_E2 = "authorized_live_e2"
AUTHORIZED_LIVE_E2_RUN_VALUE = "e2_inventory_observation"

_PROCESS_LIVE_RUN_STARTED = False
_PROCESS_LIVE_RUN_LOCK = threading.Lock()


class E2AuthorizationError(ValueError):
    """Raised when E2 live authorization or preflight is invalid."""


def _require_bool(value: object, field_name: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be bool")


def _require_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _inventory_snapshot(value: object, field_name: str) -> dict[str, int]:
    inspection = inspect_inventory(value)
    if not inspection.valid or inspection.inventory is None:
        raise ValueError(
            f"{field_name} must be a valid inventory Mapping: "
            f"{inspection.error or 'invalid inventory'}"
        )
    return dict(inspection.inventory)


@dataclass(frozen=True)
class E2MineRLRunRecord:
    """Narrow evidence for a prepared or authorized E2 MineRL attempt."""

    execution_mode: str
    authorized_live_run: str
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
    inventory_present: bool | None = None
    observed_inventory: Mapping[str, int] | None = None
    expected_inventory: Mapping[str, int] | None = None
    inventory_matches_expected: bool | None = None
    error: str | None = None
    close_error: str | None = None
    authorization_accepted: bool = False
    real_execution_performed: bool = False
    integration_verified: bool = False
    verification_level: str = UNIT_VERIFIED
    calibration_only: bool = True

    def __post_init__(self) -> None:
        if self.execution_mode != EXECUTION_MODE_AUTHORIZED_LIVE_E2:
            raise ValueError("execution_mode must be authorized_live_e2")
        if self.authorized_live_run != AUTHORIZED_LIVE_E2_RUN_VALUE:
            raise ValueError(
                "authorized_live_run must be e2_inventory_observation"
            )
        if self.check_id != EnvironmentValidationId.E2.value:
            raise ValueError("check_id must be E2")
        if self.name != "inventory_observation":
            raise ValueError("name must be inventory_observation")
        _require_identifier(self.episode_id, "episode_id")
        _require_identifier(self.backend_identity, "backend_identity")
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
        if self.inventory_present is not None:
            _require_bool(self.inventory_present, "inventory_present")
        if self.inventory_matches_expected is not None:
            _require_bool(
                self.inventory_matches_expected,
                "inventory_matches_expected",
            )
        if self.observed_inventory is not None:
            object.__setattr__(
                self,
                "observed_inventory",
                _inventory_snapshot(
                    self.observed_inventory, "observed_inventory"
                ),
            )
        if self.expected_inventory is not None:
            object.__setattr__(
                self,
                "expected_inventory",
                _inventory_snapshot(
                    self.expected_inventory, "expected_inventory"
                ),
            )
        if not isinstance(self.cleanup, E0CleanupStatus):
            raise ValueError("cleanup must be E0CleanupStatus")
        if self.verification_level != UNIT_VERIFIED:
            raise ValueError("this runtime may only emit unit_verified")
        if self.integration_verified:
            raise ValueError("this runtime cannot claim integration_verified")
        if not self.calibration_only:
            raise ValueError("E2 MineRL records must remain calibration-only")
        if self.expected_inventory is not None and dict(
            self.expected_inventory
        ) != dict(E2_CALIBRATION_INVENTORY):
            raise ValueError(
                "expected_inventory must equal frozen E2 calibration inventory"
            )
        if (
            self.observed_inventory is not None
            and self.expected_inventory is not None
            and self.inventory_matches_expected is not None
            and self.inventory_matches_expected
            != (self.observed_inventory == self.expected_inventory)
        ):
            raise ValueError(
                "inventory_matches_expected contradicts inventory mappings"
            )
        if self.success:
            if self.outcome != "inventory_ok":
                raise ValueError("success requires outcome inventory_ok")
            if not (
                self.opened
                and self.created
                and self.reset_completed
                and self.initial_state_present
                and self.closed
                and self.inventory_present is True
                and self.observed_inventory is not None
                and self.expected_inventory is not None
                and bool(self.expected_inventory)
                and self.inventory_matches_expected is True
                and self.observed_inventory == self.expected_inventory
                and self.error is None
                and self.close_error is None
            ):
                raise ValueError(
                    "success requires a clean matching E2 MineRL inventory"
                )
            if self.cleanup.has_explicit_failure():
                raise ValueError(
                    "success requires observable cleanup without explicit failure"
                )

    def as_dict(self) -> dict[str, Any]:
        """Return detached, deterministic, JSON-serializable evidence."""

        return {
            "authorization_accepted": self.authorization_accepted,
            "authorized_live_run": self.authorized_live_run,
            "backend_identity": self.backend_identity,
            "calibration_only": True,
            "check_id": self.check_id,
            "cleanup": self.cleanup.as_dict(),
            "close_error": self.close_error,
            "closed": self.closed,
            "created": self.created,
            "episode_id": self.episode_id,
            "error": self.error,
            "execution_mode": self.execution_mode,
            "expected_inventory": (
                None
                if self.expected_inventory is None
                else dict(self.expected_inventory)
            ),
            "initial_state_present": self.initial_state_present,
            "integration_verified": False,
            "inventory_matches_expected": self.inventory_matches_expected,
            "inventory_present": self.inventory_present,
            "name": self.name,
            "observed_inventory": (
                None
                if self.observed_inventory is None
                else dict(self.observed_inventory)
            ),
            "opened": self.opened,
            "outcome": self.outcome,
            "real_execution_performed": self.real_execution_performed,
            "reset_completed": self.reset_completed,
            "step_id": self.step_id,
            "success": self.success,
            "verification_level": UNIT_VERIFIED,
        }


def reset_authorized_e2_process_guards_for_tests() -> None:
    """Reset process-once guards. Intended for offline unit tests only."""

    global _PROCESS_LIVE_RUN_STARTED
    with _PROCESS_LIVE_RUN_LOCK:
        _PROCESS_LIVE_RUN_STARTED = False


def assert_e2_live_authorized(
    *,
    execution_mode: object,
    authorized_live_run: object,
    allow_gradle: object = False,
) -> None:
    if execution_mode != EXECUTION_MODE_AUTHORIZED_LIVE_E2:
        raise E2AuthorizationError(
            "execution_mode must be exactly authorized_live_e2"
        )
    if authorized_live_run != AUTHORIZED_LIVE_E2_RUN_VALUE:
        raise E2AuthorizationError(
            "authorized_live_run must be exactly e2_inventory_observation"
        )
    if allow_gradle is not False:
        raise E2AuthorizationError(
            "Gradle is not authorized for E2; allow_gradle must be False"
        )


def _validate_catalog_policy() -> None:
    catalog = load_task_catalog(ROOT / "benchmark/catalog/tasks.json")
    if catalog.active_phase != "P1-REAL-MINERL-ENVIRONMENT-VALIDATION":
        raise E2AuthorizationError(
            "active catalog phase must remain P1 environment validation"
        )
    if any(entry.live_run_allowed for entry in catalog.entries):
        raise E2AuthorizationError(
            "catalog live_run_allowed must remain false; "
            "authorization is per-run only"
        )


def _validate_e2_configuration() -> dict[str, int]:
    inspection = inspect_inventory(E2_CALIBRATION_INVENTORY)
    if not inspection.valid or not inspection.inventory:
        raise E2AuthorizationError(
            "E2_CALIBRATION_INVENTORY must be valid and non-empty"
        )
    if E2_INVENTORY_CASE.check_id is not EnvironmentValidationId.E2:
        raise E2AuthorizationError("E2 validation case identity is invalid")
    if E2_INVENTORY_CASE.requires_server_truth:
        raise E2AuthorizationError("E2 must not require server truth")
    if not E2_INVENTORY_CASE.calibration_only:
        raise E2AuthorizationError("E2 must remain calibration-only")
    task = build_e2_compatibility_task("p1-e2-preflight")
    if dict(task.initial_inventories[E2_AGENT_ID]) != dict(
        inspection.inventory
    ):
        raise E2AuthorizationError(
            "E2 compatibility task inventory differs from calibration config"
        )
    adapter = MineRLE2InventoryAdapter(episode_id="p1-e2-preflight")
    if adapter._backend is not None:
        raise E2AuthorizationError(
            "E2 adapter construction must not create a production backend"
        )
    return dict(inspection.inventory)


def check_e2_live_runner() -> dict[str, Any]:
    """Check frozen E2 live-runner configuration without authorization."""

    expected = _validate_e2_configuration()
    _validate_catalog_policy()
    return {
        "authorized_live_run_required": AUTHORIZED_LIVE_E2_RUN_VALUE,
        "calibration_only": True,
        "check_id": "E2",
        "execution_mode_required": EXECUTION_MODE_AUTHORIZED_LIVE_E2,
        "expected_inventory": expected,
        "gradle_authorized": False,
        "integration_verified": False,
        "name": "inventory_observation",
        "production_backend_constructed": False,
        "real_execution_performed": False,
        "status": "ok",
        "verification_level": UNIT_VERIFIED,
    }


def _validate_output_dir(output_dir: Path) -> Path:
    if not isinstance(output_dir, Path):
        raise E2AuthorizationError("output_dir must be a pathlib.Path")
    if not output_dir.is_absolute():
        raise E2AuthorizationError("output_dir must be an absolute path")
    resolved = output_dir.resolve()
    if resolved.exists() or resolved.is_symlink():
        raise E2AuthorizationError(
            f"output_dir must not already exist: {resolved}"
        )
    try:
        resolved.relative_to(FORMAL_E2_RUNS_ROOT)
    except ValueError as error:
        raise E2AuthorizationError(
            f"output_dir must be under {FORMAL_E2_RUNS_ROOT}"
        ) from error
    if resolved == FORMAL_E2_RUNS_ROOT or resolved.parent != FORMAL_E2_RUNS_ROOT:
        raise E2AuthorizationError(
            "output_dir must be a unique direct child of "
            "runs/p1_e2_inventory_observation/"
        )
    return resolved


def _production_backend_cls() -> type:
    from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend

    return MineRLEnvironmentBackend


def _production_backend_kwargs() -> dict[str, Any]:
    return {"max_reset_attempts": 1}


def preflight_authorized_e2(
    *,
    execution_mode: str,
    authorized_live_run: str,
    allow_gradle: bool = False,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Validate E2 authorization/config without constructing a backend."""

    assert_e2_live_authorized(
        execution_mode=execution_mode,
        authorized_live_run=authorized_live_run,
        allow_gradle=allow_gradle,
    )
    _validate_catalog_policy()
    expected = _validate_e2_configuration()
    payload: dict[str, Any] = {
        "authorized_live_run": AUTHORIZED_LIVE_E2_RUN_VALUE,
        "calibration_only": True,
        "check_id": "E2",
        "execution_mode": EXECUTION_MODE_AUTHORIZED_LIVE_E2,
        "expected_inventory": expected,
        "gradle_authorized": False,
        "integration_verified": False,
        "live_run_allowed_catalog": False,
        "name": "inventory_observation",
        "production_backend_constructed": False,
        "real_execution_performed": False,
        "requires_server_truth": False,
        "verification_level": UNIT_VERIFIED,
    }
    if output_dir is not None:
        payload["output_dir"] = str(_validate_output_dir(output_dir))
    return payload


def _write_evidence(record: E2MineRLRunRecord, output_dir: Path) -> Path:
    if record.integration_verified:
        raise E2AuthorizationError("refusing to persist integration_verified")
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "e2_inventory.json"
    path.write_text(
        json.dumps(record.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    authorization_path = output_dir / "authorization.json"
    authorization_path.write_text(
        json.dumps(
            {
                "authorized_live_run": AUTHORIZED_LIVE_E2_RUN_VALUE,
                "catalog_live_run_allowed_remains_false": True,
                "execution_mode": EXECUTION_MODE_AUTHORIZED_LIVE_E2,
                "gradle_authorized": False,
                "integration_verified": False,
                "model_api_authorized": False,
                "note": (
                    "Per-run authorization only. Does not mark E2 "
                    "integration_verified. Evidence contains inventory and "
                    "lifecycle metadata only."
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
    adapter: MineRLE2InventoryAdapter | None,
    lifecycle: EnvironmentValidationResult,
    authorization_accepted: bool,
    real_execution_performed: bool,
    backend_identity: str,
) -> E2MineRLRunRecord:
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
    return E2MineRLRunRecord(
        execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E2,
        authorized_live_run=AUTHORIZED_LIVE_E2_RUN_VALUE,
        check_id=EnvironmentValidationId.E2.value,
        name="inventory_observation",
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
        inventory_present=lifecycle.inventory_present,
        observed_inventory=lifecycle.observed_inventory,
        expected_inventory=lifecycle.expected_inventory,
        inventory_matches_expected=lifecycle.inventory_matches_expected,
        error=error,
        close_error=lifecycle.close_error,
        authorization_accepted=authorization_accepted,
        real_execution_performed=real_execution_performed,
    )


def run_authorized_e2_minerl(
    *,
    execution_mode: str,
    authorized_live_run: str,
    output_dir: Path,
    episode_id: str = "p1-e2-inventory-observation",
    allow_gradle: bool = False,
    preflight_only: bool = False,
) -> E2MineRLRunRecord | dict[str, Any]:
    """Run authorized E2 inventory validation, or refuse before MineRL."""

    preflight = preflight_authorized_e2(
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
            raise E2AuthorizationError(
                "authorized E2 allows only one real run attempt per process"
            )
        _PROCESS_LIVE_RUN_STARTED = True

    backend_cls = _production_backend_cls()
    real_execution = backend_cls.__name__ == BACKEND_IDENTITY
    factory = MineRLE2InventoryAdapter.lifecycle_factory(
        episode_id=episode_id,
        backend_cls=backend_cls,
        backend_kwargs=_production_backend_kwargs(),
    )
    holder: dict[str, MineRLE2InventoryAdapter | None] = {"adapter": None}

    def capturing_factory() -> MineRLE2InventoryAdapter:
        adapter = factory()
        holder["adapter"] = adapter
        return adapter

    lifecycle = EnvironmentValidationRunner().run(
        E2_INVENTORY_CASE,
        capturing_factory,
        episode_id=episode_id,
        expected_inventory=E2_CALIBRATION_INVENTORY,
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
        prog="obsidianlink.env.integration.e2_run",
        description=(
            "Authorized P1 E2 MineRL inventory observation entrypoint. "
            "Use --check for an offline-safe configuration check."
        ),
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execution-mode")
    parser.add_argument("--authorized-live-run")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--episode-id",
        default="p1-e2-inventory-observation",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate authorization without starting MineRL",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        if any(
            value is not None
            for value in (
                args.execution_mode,
                args.authorized_live_run,
                args.output_dir,
            )
        ) or args.preflight_only:
            raise E2AuthorizationError(
                "--check cannot be combined with live or preflight arguments"
            )
        payload: Mapping[str, Any] = check_e2_live_runner()
    else:
        if args.output_dir is None:
            raise E2AuthorizationError("--output-dir is required for E2 live/preflight")
        result = run_authorized_e2_minerl(
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

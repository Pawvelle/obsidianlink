"""Authorized P1 E3 real-MineRL selected-item entrypoint.

Import and ``--check`` are offline-safe. The production backend is resolved
only after exact per-run authorization and only on the execution path. The
live path performs open/reset/public selected-item validation/close; it never
steps actions, reads evaluator truth, invokes models, or runs Gradle.
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
from obsidianlink.env.integration.e3_adapter import MineRLE3SelectedItemAdapter
from obsidianlink.env.integration.e3_config import (
    E3_AGENT_ID,
    E3_CALIBRATION_INVENTORY,
    E3_EXPECTED_SELECTED_ITEM,
    build_e3_compatibility_task,
)
from obsidianlink.env.validation import (
    E3_SELECTED_ITEM_CASE,
    EnvironmentValidationRunner,
)
from obsidianlink.env.validation.contract import EnvironmentValidationId
from obsidianlink.env.validation.result import UNIT_VERIFIED, EnvironmentValidationResult
from obsidianlink.env.validation.selected_item import validate_selected_item


ROOT = Path(__file__).resolve().parents[3]
FORMAL_E3_RUNS_ROOT = (ROOT / "runs" / "p1_e3_selected_item_observation").resolve()
EXECUTION_MODE_AUTHORIZED_LIVE_E3 = "authorized_live_e3"
AUTHORIZED_LIVE_E3_RUN_VALUE = "e3_selected_item"

_PROCESS_LIVE_RUN_STARTED = False
_PROCESS_LIVE_RUN_LOCK = threading.Lock()


class E3AuthorizationError(ValueError):
    """Raised when E3 live authorization or preflight is invalid."""


def _require_bool(value: object, field_name: str) -> None:
    if type(value) is not bool:
        raise ValueError(f"{field_name} must be bool")


def _require_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class E3MineRLRunRecord:
    """Narrow evidence for one prepared or authorized E3 attempt."""

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
    selected_item_present: bool | None = None
    observed_selected_item: str | None = None
    expected_selected_item: str | None = None
    selected_item_matches_expected: bool | None = None
    error: str | None = None
    close_error: str | None = None
    authorization_accepted: bool = False
    real_execution_performed: bool = False
    integration_verified: bool = False
    verification_level: str = UNIT_VERIFIED
    calibration_only: bool = True

    def __post_init__(self) -> None:
        if self.execution_mode != EXECUTION_MODE_AUTHORIZED_LIVE_E3:
            raise ValueError("execution_mode must be authorized_live_e3")
        if self.authorized_live_run != AUTHORIZED_LIVE_E3_RUN_VALUE:
            raise ValueError("authorized_live_run must be e3_selected_item")
        if self.check_id != EnvironmentValidationId.E3.value:
            raise ValueError("check_id must be E3")
        if self.name != "selected_item":
            raise ValueError("name must be selected_item")
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
        if self.selected_item_present is not None:
            _require_bool(self.selected_item_present, "selected_item_present")
        if self.selected_item_matches_expected is not None:
            _require_bool(
                self.selected_item_matches_expected,
                "selected_item_matches_expected",
            )
        if self.observed_selected_item is not None:
            object.__setattr__(
                self,
                "observed_selected_item",
                validate_selected_item(
                    self.observed_selected_item, "observed_selected_item"
                ),
            )
        if self.expected_selected_item is not None:
            object.__setattr__(
                self,
                "expected_selected_item",
                validate_selected_item(
                    self.expected_selected_item, "expected_selected_item"
                ),
            )
        if not isinstance(self.cleanup, E0CleanupStatus):
            raise ValueError("cleanup must be E0CleanupStatus")
        if self.verification_level != UNIT_VERIFIED:
            raise ValueError("this runtime may only emit unit_verified")
        if self.integration_verified:
            raise ValueError("this runtime cannot claim integration_verified")
        if not self.calibration_only:
            raise ValueError("E3 MineRL records must remain calibration-only")
        if (
            self.expected_selected_item is not None
            and self.expected_selected_item != E3_EXPECTED_SELECTED_ITEM
        ):
            raise ValueError("expected_selected_item must equal frozen E3 expectation")
        if (
            self.observed_selected_item is not None
            and self.expected_selected_item is not None
            and self.selected_item_matches_expected is not None
            and self.selected_item_matches_expected
            != (self.observed_selected_item == self.expected_selected_item)
        ):
            raise ValueError("selected_item_matches_expected contradicts item values")
        if self.success:
            if self.outcome != "selected_item_ok":
                raise ValueError("success requires outcome selected_item_ok")
            if not (
                self.opened
                and self.created
                and self.reset_completed
                and self.initial_state_present
                and self.closed
                and self.selected_item_present is True
                and self.observed_selected_item == E3_EXPECTED_SELECTED_ITEM
                and self.expected_selected_item == E3_EXPECTED_SELECTED_ITEM
                and self.selected_item_matches_expected is True
                and self.error is None
                and self.close_error is None
            ):
                raise ValueError("success requires a clean matching E3 selected item")
            if self.cleanup.has_explicit_failure():
                raise ValueError("success requires observable cleanup without failure")

    def as_dict(self) -> dict[str, Any]:
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
            "expected_selected_item": self.expected_selected_item,
            "initial_state_present": self.initial_state_present,
            "integration_verified": False,
            "name": self.name,
            "observed_selected_item": self.observed_selected_item,
            "opened": self.opened,
            "outcome": self.outcome,
            "real_execution_performed": self.real_execution_performed,
            "reset_completed": self.reset_completed,
            "selected_item_matches_expected": self.selected_item_matches_expected,
            "selected_item_present": self.selected_item_present,
            "step_id": self.step_id,
            "success": self.success,
            "verification_level": UNIT_VERIFIED,
        }


def reset_authorized_e3_process_guards_for_tests() -> None:
    global _PROCESS_LIVE_RUN_STARTED
    with _PROCESS_LIVE_RUN_LOCK:
        _PROCESS_LIVE_RUN_STARTED = False


def assert_e3_live_authorized(
    *, execution_mode: object, authorized_live_run: object, allow_gradle: object = False
) -> None:
    if execution_mode != EXECUTION_MODE_AUTHORIZED_LIVE_E3:
        raise E3AuthorizationError("execution_mode must be exactly authorized_live_e3")
    if authorized_live_run != AUTHORIZED_LIVE_E3_RUN_VALUE:
        raise E3AuthorizationError(
            "authorized_live_run must be exactly e3_selected_item"
        )
    if allow_gradle is not False:
        raise E3AuthorizationError("Gradle is not authorized for E3")


def _validate_catalog_policy() -> None:
    catalog = load_task_catalog(ROOT / "benchmark/catalog/tasks.json")
    if catalog.active_phase != "P1-REAL-MINERL-ENVIRONMENT-VALIDATION":
        raise E3AuthorizationError("active catalog phase must remain P1")
    if any(entry.live_run_allowed for entry in catalog.entries):
        raise E3AuthorizationError("catalog live_run_allowed must remain false")


def _validate_e3_configuration() -> str:
    expected = validate_selected_item(E3_EXPECTED_SELECTED_ITEM, "expected")
    if dict(E3_CALIBRATION_INVENTORY) != {expected: 1}:
        raise E3AuthorizationError(
            "E3 calibration inventory must contain only the expected item"
        )
    if (
        E3_SELECTED_ITEM_CASE.check_id is not EnvironmentValidationId.E3
        or E3_SELECTED_ITEM_CASE.requires_server_truth
        or not E3_SELECTED_ITEM_CASE.calibration_only
    ):
        raise E3AuthorizationError("E3 validation case identity/policy is invalid")
    task = build_e3_compatibility_task("p1-e3-preflight")
    if dict(task.initial_inventories[E3_AGENT_ID]) != dict(E3_CALIBRATION_INVENTORY):
        raise E3AuthorizationError("E3 compatibility task inventory differs")
    adapter = MineRLE3SelectedItemAdapter(episode_id="p1-e3-preflight")
    if adapter._backend is not None:
        raise E3AuthorizationError("E3 adapter construction created a backend")
    return expected


def check_e3_live_runner() -> dict[str, Any]:
    expected = _validate_e3_configuration()
    _validate_catalog_policy()
    return {
        "authorized_live_run_required": AUTHORIZED_LIVE_E3_RUN_VALUE,
        "calibration_only": True,
        "check_id": "E3",
        "execution_mode_required": EXECUTION_MODE_AUTHORIZED_LIVE_E3,
        "expected_selected_item": expected,
        "gradle_authorized": False,
        "integration_verified": False,
        "name": "selected_item",
        "production_backend_constructed": False,
        "real_execution_performed": False,
        "status": "ok",
        "verification_level": UNIT_VERIFIED,
    }


def _validate_output_dir(output_dir: Path) -> Path:
    if not isinstance(output_dir, Path) or not output_dir.is_absolute():
        raise E3AuthorizationError("output_dir must be an absolute pathlib.Path")
    resolved = output_dir.resolve()
    if resolved.exists() or resolved.is_symlink():
        raise E3AuthorizationError(f"output_dir must not exist: {resolved}")
    try:
        resolved.relative_to(FORMAL_E3_RUNS_ROOT)
    except ValueError as exc:
        raise E3AuthorizationError(
            f"output_dir must be under {FORMAL_E3_RUNS_ROOT}"
        ) from exc
    if resolved == FORMAL_E3_RUNS_ROOT or resolved.parent != FORMAL_E3_RUNS_ROOT:
        raise E3AuthorizationError("output_dir must be a unique direct child")
    return resolved


def _production_backend_cls() -> type:
    from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend

    return MineRLEnvironmentBackend


def _production_backend_kwargs() -> dict[str, Any]:
    return {"max_reset_attempts": 1}


def preflight_authorized_e3(
    *,
    execution_mode: str,
    authorized_live_run: str,
    allow_gradle: bool = False,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    assert_e3_live_authorized(
        execution_mode=execution_mode,
        authorized_live_run=authorized_live_run,
        allow_gradle=allow_gradle,
    )
    _validate_catalog_policy()
    expected = _validate_e3_configuration()
    payload: dict[str, Any] = {
        "authorized_live_run": AUTHORIZED_LIVE_E3_RUN_VALUE,
        "calibration_only": True,
        "check_id": "E3",
        "execution_mode": EXECUTION_MODE_AUTHORIZED_LIVE_E3,
        "expected_selected_item": expected,
        "gradle_authorized": False,
        "integration_verified": False,
        "live_run_allowed_catalog": False,
        "name": "selected_item",
        "production_backend_constructed": False,
        "real_execution_performed": False,
        "requires_server_truth": False,
        "verification_level": UNIT_VERIFIED,
    }
    if output_dir is not None:
        payload["output_dir"] = str(_validate_output_dir(output_dir))
    return payload


def _write_evidence(record: E3MineRLRunRecord, output_dir: Path) -> Path:
    if record.integration_verified:
        raise E3AuthorizationError("refusing to persist integration_verified")
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "e3_selected_item.json"
    path.write_text(
        json.dumps(record.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "authorization.json").write_text(
        json.dumps(
            {
                "authorized_live_run": AUTHORIZED_LIVE_E3_RUN_VALUE,
                "catalog_live_run_allowed_remains_false": True,
                "execution_mode": EXECUTION_MODE_AUTHORIZED_LIVE_E3,
                "gradle_authorized": False,
                "integration_verified": False,
                "model_api_authorized": False,
                "note": "Per-run authorization only; no capability promotion.",
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
    adapter: MineRLE3SelectedItemAdapter | None,
    lifecycle: EnvironmentValidationResult,
    authorization_accepted: bool,
    real_execution_performed: bool,
    backend_identity: str,
) -> E3MineRLRunRecord:
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
    success = lifecycle.success
    outcome = lifecycle.outcome
    error = lifecycle.error
    if success and cleanup.has_explicit_failure():
        success = False
        outcome = "cleanup_failed"
        error = cleanup.failure_detail()
    return E3MineRLRunRecord(
        execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E3,
        authorized_live_run=AUTHORIZED_LIVE_E3_RUN_VALUE,
        check_id="E3",
        name="selected_item",
        episode_id=lifecycle.episode_id,
        step_id=lifecycle.step_id,
        backend_identity=backend_identity,
        opened=False if adapter is None else adapter.open_succeeded,
        created=lifecycle.created,
        reset_completed=lifecycle.reset_completed,
        initial_state_present=lifecycle.initial_state_present,
        closed=lifecycle.closed,
        success=success,
        outcome=outcome,
        cleanup=cleanup,
        selected_item_present=lifecycle.selected_item_present,
        observed_selected_item=lifecycle.observed_selected_item,
        expected_selected_item=lifecycle.expected_selected_item,
        selected_item_matches_expected=lifecycle.selected_item_matches_expected,
        error=error,
        close_error=lifecycle.close_error,
        authorization_accepted=authorization_accepted,
        real_execution_performed=real_execution_performed,
    )


def run_authorized_e3_minerl(
    *,
    execution_mode: str,
    authorized_live_run: str,
    output_dir: Path,
    episode_id: str = "p1-e3-selected-item",
    allow_gradle: bool = False,
    preflight_only: bool = False,
) -> E3MineRLRunRecord | dict[str, Any]:
    """Run authorized E3 validation, or fail before resolving MineRL."""

    preflight = preflight_authorized_e3(
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
            raise E3AuthorizationError("authorized E3 allows one real run per process")
        _PROCESS_LIVE_RUN_STARTED = True

    backend_cls = _production_backend_cls()
    factory = MineRLE3SelectedItemAdapter.lifecycle_factory(
        episode_id=episode_id,
        backend_cls=backend_cls,
        backend_kwargs=_production_backend_kwargs(),
    )
    holder: dict[str, MineRLE3SelectedItemAdapter | None] = {"adapter": None}

    def capturing_factory() -> MineRLE3SelectedItemAdapter:
        adapter = factory()
        holder["adapter"] = adapter
        return adapter

    lifecycle = EnvironmentValidationRunner().run(
        E3_SELECTED_ITEM_CASE,
        capturing_factory,
        episode_id=episode_id,
        expected_selected_item=E3_EXPECTED_SELECTED_ITEM,
    )
    record = _record_from_lifecycle(
        adapter=holder["adapter"],
        lifecycle=lifecycle,
        authorization_accepted=True,
        real_execution_performed=backend_cls.__name__ == BACKEND_IDENTITY,
        backend_identity=getattr(holder["adapter"], "backend_identity", backend_cls.__name__),
    )
    _write_evidence(record, _validate_output_dir(output_dir))
    return record


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="obsidianlink.env.integration.e3_run",
        description=(
            "Authorized P1 E3 selected-item entrypoint. "
            "Use --check for an offline-safe configuration check."
        ),
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execution-mode")
    parser.add_argument("--authorized-live-run")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episode-id", default="p1-e3-selected-item")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        if any(
            value is not None
            for value in (args.execution_mode, args.authorized_live_run, args.output_dir)
        ) or args.preflight_only:
            raise E3AuthorizationError("--check cannot be combined with live arguments")
        payload: Mapping[str, Any] = check_e3_live_runner()
    else:
        if args.output_dir is None:
            raise E3AuthorizationError("--output-dir is required for E3 live/preflight")
        result = run_authorized_e3_minerl(
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

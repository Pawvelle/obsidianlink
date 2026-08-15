"""Explicitly authorized P1 E7 real-MineRL bucket-usage validation entrypoint.

Imports and ``--check`` are offline-safe. The live path resolves production
MineRL only after the exact per-variant gate and performs exactly one
bounded ``use_item``. One authorization token is one variant, one fresh
episode, and one bucket action. It never runs Gradle, models, solvers,
water+lava together, or later validation cases.
"""

from __future__ import annotations

import json
import hashlib
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from obsidianlink.core.task_catalog import load_task_catalog
from obsidianlink.core.types import TaskInstance
from obsidianlink.env.integration.e0_adapter import BACKEND_IDENTITY
from obsidianlink.env.integration.e0_cleanup import E0CleanupStatus
from obsidianlink.env.integration.e7_adapter import MineRLE7BucketAdapter
from obsidianlink.env.integration.e7_config import (
    E7_AGENT_ID,
    E7_DURATION_TICKS,
    E7_INITIAL_PITCH,
    E7_INITIAL_YAW,
    E7_SPAWN_WORLD,
    E7_TARGET_GRID_CELL,
    E7_TARGET_WORLD_CELL,
    E7_WATER_CALIBRATION,
    E7_LAVA_CALIBRATION,
    build_e7_compatibility_task,
    e7_calibration,
)
from obsidianlink.env.validation import E7_BUCKET_CASE, EnvironmentValidationRunner
from obsidianlink.env.validation.bucket import BucketCalibrationVariant, validate_bucket_variant
from obsidianlink.env.validation.contract import EnvironmentValidationId
from obsidianlink.env.validation.result import EnvironmentValidationResult


ROOT = Path(__file__).resolve().parents[3]
FORMAL_E7_RUNS_ROOT = (ROOT / "runs" / "p1_e7_bucket_usage").resolve()
RUNTIME_LOGS_ROOT = (ROOT / "logs").resolve()
EXECUTION_MODE_AUTHORIZED_LIVE_E7_WATER = "authorized_live_e7_water"
EXECUTION_MODE_AUTHORIZED_LIVE_E7_LAVA = "authorized_live_e7_lava"
AUTHORIZED_LIVE_E7_WATER_RUN_VALUE = "e7_water_bucket"
AUTHORIZED_LIVE_E7_LAVA_RUN_VALUE = "e7_lava_bucket"
_VARIANT_GATES = {
    BucketCalibrationVariant.WATER: (
        EXECUTION_MODE_AUTHORIZED_LIVE_E7_WATER,
        AUTHORIZED_LIVE_E7_WATER_RUN_VALUE,
    ),
    BucketCalibrationVariant.LAVA: (
        EXECUTION_MODE_AUTHORIZED_LIVE_E7_LAVA,
        AUTHORIZED_LIVE_E7_LAVA_RUN_VALUE,
    ),
}

_PROCESS_LIVE_RUN_STARTED = False
_PROCESS_LIVE_RUN_LOCK = threading.Lock()


class E7AuthorizationError(ValueError):
    """Raised when the exact E7 live gate or preflight is invalid."""


def resolve_e7_live_variant(*, execution_mode: object, authorized_live_run: object) -> BucketCalibrationVariant:
    for variant, (mode, token) in _VARIANT_GATES.items():
        if execution_mode == mode and authorized_live_run == token:
            return variant
    raise E7AuthorizationError(
        "execution_mode and authorized_live_run must be exactly one E7 variant gate"
    )


@dataclass(frozen=True)
class E7LogEvidence:
    kind: str
    path: str
    sha256: str
    summary: str

    def __post_init__(self) -> None:
        if self.kind not in {"minecraft", "jvm_crash", "process_watcher"}:
            raise ValueError("unknown E7 log evidence kind")
        if not Path(self.path).is_absolute():
            raise ValueError("E7 log evidence path must be absolute")
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError("E7 log evidence sha256 must be lowercase hex")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("E7 log evidence summary must be non-empty")

    def as_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class E7MineRLRunRecord:
    execution_mode: str
    authorized_live_run: str
    backend_identity: str
    opened: bool
    authorization_accepted: bool
    real_execution_performed: bool
    variant: str
    lifecycle: EnvironmentValidationResult
    cleanup: E0CleanupStatus
    failure_cause: str
    log_evidence: tuple[E7LogEvidence, ...]

    def __post_init__(self) -> None:
        variant = validate_bucket_variant(self.variant)
        mode, token = _VARIANT_GATES[variant]
        if self.execution_mode != mode:
            raise ValueError("execution_mode must match the frozen E7 variant gate")
        if self.authorized_live_run != token:
            raise ValueError("authorized_live_run must match the frozen E7 variant gate")
        if not isinstance(self.lifecycle, EnvironmentValidationResult) or self.lifecycle.check_id is not EnvironmentValidationId.E7:
            raise ValueError("lifecycle must be an E7 validation result")
        if not isinstance(self.cleanup, E0CleanupStatus):
            raise ValueError("cleanup must be E0CleanupStatus")
        if any(type(value) is not bool for value in (self.opened, self.authorization_accepted, self.real_execution_performed)):
            raise ValueError("record flags must be bool")
        if self.failure_cause not in {"unknown", "minecraft_native_crash"}:
            raise ValueError("unknown E7 failure cause")
        if not isinstance(self.log_evidence, tuple) or not all(
            isinstance(value, E7LogEvidence) for value in self.log_evidence
        ):
            raise ValueError("log_evidence must be a tuple of E7LogEvidence")
        if self.failure_cause == "minecraft_native_crash" and not (
            self.lifecycle.outcome == "reset_failed"
            and any(
                value.kind in {"minecraft", "jvm_crash"}
                and value.summary.startswith("JVM fatal SIGSEGV in liblwjgl")
                for value in self.log_evidence
            )
        ):
            raise ValueError(
                "native-crash cause requires reset failure and explicit JVM crash evidence"
            )

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
                "evidence_manifest": {
                    "logs": [value.as_dict() for value in self.log_evidence]
                },
                "failure_cause": self.failure_cause,
                "opened": self.opened,
                "outcome": self.outcome,
                "real_execution_performed": self.real_execution_performed,
                "success": self.success,
                "variant": self.variant,
            }
        )
        return payload


def _runtime_log_paths() -> tuple[Path, ...]:
    paths = list(RUNTIME_LOGS_ROOT.glob("mc_*.log"))
    paths.extend((RUNTIME_LOGS_ROOT / "minerl_watchers").glob("watcher_*.log"))
    return tuple(sorted(path.resolve() for path in paths if path.is_file()))


def _snapshot_runtime_logs() -> dict[str, tuple[int, int]]:
    return {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in _runtime_log_paths()
    }


def _log_summary(text: str, kind: str) -> str:
    if (
        "A fatal error has been detected by the Java Runtime Environment" in text
        and "SIGSEGV" in text
        and "liblwjgl" in text
    ):
        if "Sound engine" in text:
            return "JVM fatal SIGSEGV in liblwjgl native code on the Sound engine thread"
        return "JVM fatal SIGSEGV in liblwjgl native code"
    if kind == "process_watcher" and "Child is not running anymore" in text:
        return "process watcher reports that the child is no longer running"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return (lines[-1][:240] if lines else "log file is empty")


def _make_log_evidence(path: Path, kind: str) -> E7LogEvidence:
    data = path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    return E7LogEvidence(
        kind=kind,
        path=str(path.resolve()),
        sha256=hashlib.sha256(data).hexdigest(),
        summary=_log_summary(text, kind),
    )


def _collect_runtime_log_evidence(
    before: Mapping[str, tuple[int, int]],
) -> tuple[str, tuple[E7LogEvidence, ...]]:
    changed: list[Path] = []
    for path in _runtime_log_paths():
        state = (path.stat().st_size, path.stat().st_mtime_ns)
        if before.get(str(path)) != state:
            changed.append(path)
    evidence: list[E7LogEvidence] = []
    seen: set[Path] = set()
    native_crash = False
    for path in changed:
        kind = "process_watcher" if path.parent.name == "minerl_watchers" else "minecraft"
        item = _make_log_evidence(path, kind)
        evidence.append(item)
        seen.add(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        native_crash = native_crash or (
            "A fatal error has been detected by the Java Runtime Environment" in text
            and "SIGSEGV" in text
            and "liblwjgl" in text
        )
        for match in re.findall(r"(/[^\s]+/hs_err_pid\d+\.log)", text):
            crash_path = Path(match).resolve()
            if crash_path.is_file() and crash_path not in seen:
                crash_item = _make_log_evidence(crash_path, "jvm_crash")
                evidence.append(crash_item)
                seen.add(crash_path)
                crash_text = crash_path.read_text(encoding="utf-8", errors="replace")
                native_crash = native_crash or (
                    "A fatal error has been detected by the Java Runtime Environment" in crash_text
                    and "SIGSEGV" in crash_text
                    and "liblwjgl" in crash_text
                )
    return (
        "minecraft_native_crash" if native_crash else "unknown",
        tuple(sorted(evidence, key=lambda value: (value.kind, value.path))),
    )


def reset_authorized_e7_process_guards_for_tests() -> None:
    global _PROCESS_LIVE_RUN_STARTED
    with _PROCESS_LIVE_RUN_LOCK:
        _PROCESS_LIVE_RUN_STARTED = False


def assert_e7_live_authorized(*, execution_mode: object, authorized_live_run: object, allow_gradle: object = False) -> BucketCalibrationVariant:
    variant = resolve_e7_live_variant(
        execution_mode=execution_mode, authorized_live_run=authorized_live_run
    )
    if allow_gradle is not False:
        raise E7AuthorizationError("Gradle is not authorized for E7")
    return variant


def _validate_catalog_policy() -> None:
    catalog = load_task_catalog(ROOT / "benchmark/catalog/tasks.json")
    if catalog.active_phase != "P1-REAL-MINERL-ENVIRONMENT-VALIDATION":
        raise E7AuthorizationError("active catalog phase must remain P1")
    if any(entry.live_run_allowed for entry in catalog.entries):
        raise E7AuthorizationError("catalog live_run_allowed must remain false")


def _validate_configuration() -> None:
    if not (
        E7_BUCKET_CASE.check_id is EnvironmentValidationId.E7
        and E7_BUCKET_CASE.requires_server_truth
        and E7_BUCKET_CASE.calibration_only
        and E7_WATER_CALIBRATION.bucket_item == "water_bucket"
        and E7_LAVA_CALIBRATION.bucket_item == "lava_bucket"
        and E7_TARGET_WORLD_CELL == (0, 4, 1)
        and E7_TARGET_GRID_CELL == (0, 0, 1)
        and E7_SPAWN_WORLD == (0, 4, 0)
        and E7_DURATION_TICKS == 1
        and (E7_INITIAL_YAW, E7_INITIAL_PITCH) == (0.0, 60.0)
    ):
        raise E7AuthorizationError("frozen E7 calibration differs")
    for variant in (BucketCalibrationVariant.WATER, BucketCalibrationVariant.LAVA):
        task = build_e7_compatibility_task("p1-e7-preflight", variant)
        calibration = e7_calibration(variant)
        if dict(task.initial_inventories[E7_AGENT_ID]) != dict(calibration.initial_inventory):
            raise E7AuthorizationError("E7 compatibility inventory differs")
        if task.spawn_positions[E7_AGENT_ID] != E7_SPAWN_WORLD:
            raise E7AuthorizationError("E7 flat-ground spawn differs")
    if MineRLE7BucketAdapter(episode_id="p1-e7-preflight")._backend is not None:
        raise E7AuthorizationError("E7 adapter construction created a backend")


def check_e7_live_runner() -> dict[str, Any]:
    _validate_catalog_policy()
    _validate_configuration()
    return {
        "authorized_live_run_required": {
            "lava": AUTHORIZED_LIVE_E7_LAVA_RUN_VALUE,
            "water": AUTHORIZED_LIVE_E7_WATER_RUN_VALUE,
        },
        "calibration_only": True,
        "check_id": "E7",
        "execution_mode_required": {
            "lava": EXECUTION_MODE_AUTHORIZED_LIVE_E7_LAVA,
            "water": EXECUTION_MODE_AUTHORIZED_LIVE_E7_WATER,
        },
        "expected_before_fluid": "none",
        "gradle_authorized": False,
        "initial_pitch": E7_INITIAL_PITCH,
        "initial_yaw": E7_INITIAL_YAW,
        "integration_verified": False,
        "name": "bucket_usage",
        "one_token_one_variant": True,
        "production_backend_constructed": False,
        "real_execution_performed": False,
        "requested_duration_ticks": E7_DURATION_TICKS,
        "status": "ok",
        "target_grid_cell": list(E7_TARGET_GRID_CELL),
        "target_world_cell": list(E7_TARGET_WORLD_CELL),
        "variants": {
            "lava": {
                "bucket_item": E7_LAVA_CALIBRATION.bucket_item,
                "expected_fluid": E7_LAVA_CALIBRATION.expected_fluid,
                "initial_inventory": dict(E7_LAVA_CALIBRATION.initial_inventory),
            },
            "water": {
                "bucket_item": E7_WATER_CALIBRATION.bucket_item,
                "expected_fluid": E7_WATER_CALIBRATION.expected_fluid,
                "initial_inventory": dict(E7_WATER_CALIBRATION.initial_inventory),
            },
        },
        "verification_level": "unit_verified",
    }


def _validate_output_dir(output_dir: Path, variant: BucketCalibrationVariant) -> Path:
    if not isinstance(output_dir, Path) or not output_dir.is_absolute():
        raise E7AuthorizationError("output_dir must be an absolute pathlib.Path")
    resolved = output_dir.resolve()
    if resolved.exists() or resolved.is_symlink():
        raise E7AuthorizationError(f"output_dir must not exist: {resolved}")
    variant_root = (FORMAL_E7_RUNS_ROOT / variant.value).resolve()
    try:
        resolved.relative_to(variant_root)
    except ValueError as exc:
        raise E7AuthorizationError(f"output_dir must be under {variant_root}") from exc
    if resolved == variant_root or resolved.parent != variant_root:
        raise E7AuthorizationError("output_dir must be a unique direct child of the variant directory")
    return resolved


def _e7_env_factory(task: TaskInstance) -> Any:
    from obsidianlink.env.portal_spec import PortalA0EnvSpec

    initial_inventory = tuple(
        {"type": item, "quantity": quantity}
        for item, quantity in task.initial_inventories[E7_AGENT_ID].items()
        if quantity > 0
    )
    specification = PortalA0EnvSpec(
        max_episode_steps=task.limits["max_environment_steps"],
        max_game_time_seconds=task.limits["max_game_time_seconds"],
        initial_inventory=initial_inventory,
        initial_position=task.spawn_positions[E7_AGENT_ID],
        include_agent_start_placement=True,
        grid_at_spawn=True,
        initial_yaw=E7_INITIAL_YAW,
        initial_pitch=E7_INITIAL_PITCH,
    )
    return specification.make()


def _production_backend_cls() -> type:
    from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
    return MineRLEnvironmentBackend


def _production_backend_kwargs() -> dict[str, Any]:
    return {"max_reset_attempts": 1, "env_factory": _e7_env_factory}


def preflight_authorized_e7(*, execution_mode: str, authorized_live_run: str, output_dir: Path | None = None, allow_gradle: bool = False) -> dict[str, Any]:
    variant = assert_e7_live_authorized(execution_mode=execution_mode, authorized_live_run=authorized_live_run, allow_gradle=allow_gradle)
    _validate_catalog_policy()
    _validate_configuration()
    payload = check_e7_live_runner()
    calibration = e7_calibration(variant)
    payload.update(
        {
            "execution_mode": execution_mode,
            "authorized_live_run": authorized_live_run,
            "requires_server_truth": True,
            "variant": variant.value,
            "bucket_item": calibration.bucket_item,
            "expected_fluid": calibration.expected_fluid,
        }
    )
    if output_dir is not None:
        payload["output_dir"] = str(_validate_output_dir(output_dir, variant))
    return payload


def _write_evidence(record: E7MineRLRunRecord, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "e7_bucket_usage.json"
    path.write_text(json.dumps(record.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "authorization.json").write_text(
        json.dumps(
            {
                "authorized_live_run": record.authorized_live_run,
                "catalog_live_run_allowed_remains_false": True,
                "execution_mode": record.execution_mode,
                "gradle_authorized": False,
                "integration_verified": False,
                "model_api_authorized": False,
                "planned_tested_bucket_action_count": 1,
                "variant": record.variant,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return path


def run_authorized_e7_minerl(*, execution_mode: str, authorized_live_run: str, output_dir: Path, episode_id: str = "p1-e7-bucket-usage", allow_gradle: bool = False, preflight_only: bool = False) -> E7MineRLRunRecord | dict[str, Any]:
    preflight = preflight_authorized_e7(execution_mode=execution_mode, authorized_live_run=authorized_live_run, output_dir=output_dir, allow_gradle=allow_gradle)
    if preflight_only:
        return preflight
    variant = resolve_e7_live_variant(execution_mode=execution_mode, authorized_live_run=authorized_live_run)
    global _PROCESS_LIVE_RUN_STARTED
    with _PROCESS_LIVE_RUN_LOCK:
        if _PROCESS_LIVE_RUN_STARTED:
            raise E7AuthorizationError("authorized E7 allows one real run per process")
        _PROCESS_LIVE_RUN_STARTED = True
    log_snapshot = _snapshot_runtime_logs()
    backend_cls = _production_backend_cls()
    factory = MineRLE7BucketAdapter.lifecycle_factory(
        episode_id=episode_id,
        variant=variant,
        backend_cls=backend_cls,
        backend_kwargs=_production_backend_kwargs(),
    )
    holder: dict[str, MineRLE7BucketAdapter | None] = {"adapter": None}

    def capture() -> MineRLE7BucketAdapter:
        adapter = factory()
        holder["adapter"] = adapter
        return adapter

    lifecycle = EnvironmentValidationRunner().run(
        E7_BUCKET_CASE,
        capture,
        episode_id=episode_id,
        bucket_variant=variant.value,
        target_cell=E7_TARGET_WORLD_CELL,
        target_grid_cell=E7_TARGET_GRID_CELL,
        requested_duration_ticks=E7_DURATION_TICKS,
    )
    adapter = holder["adapter"]
    cleanup = adapter.cleanup_status() if adapter is not None else E0CleanupStatus(lifecycle.closed, lifecycle.closed, lifecycle.closed, lifecycle.closed)
    detected_cause, log_evidence = _collect_runtime_log_evidence(log_snapshot)
    failure_cause = (
        detected_cause if lifecycle.outcome == "reset_failed" else "unknown"
    )
    record = E7MineRLRunRecord(
        execution_mode=execution_mode,
        authorized_live_run=authorized_live_run,
        backend_identity=getattr(adapter, "backend_identity", backend_cls.__name__),
        opened=False if adapter is None else adapter.open_succeeded,
        authorization_accepted=True,
        real_execution_performed=backend_cls.__name__ == BACKEND_IDENTITY,
        variant=variant.value,
        lifecycle=lifecycle,
        cleanup=cleanup,
        failure_cause=failure_cause,
        log_evidence=log_evidence,
    )
    _write_evidence(record, _validate_output_dir(output_dir, variant))
    return record


def build_parser():
    import argparse
    parser = argparse.ArgumentParser(prog="obsidianlink.env.integration.e7_run", description="Authorized P1 E7 bucket-usage entrypoint; --check is offline-safe.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execution-mode")
    parser.add_argument("--authorized-live-run")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episode-id", default="p1-e7-bucket-usage")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        if any(value is not None for value in (args.execution_mode, args.authorized_live_run, args.output_dir)) or args.preflight_only:
            raise E7AuthorizationError("--check cannot be combined with live arguments")
        payload: Mapping[str, Any] = check_e7_live_runner()
    else:
        if args.output_dir is None:
            raise E7AuthorizationError("--output-dir is required for E7 live/preflight")
        result = run_authorized_e7_minerl(execution_mode=args.execution_mode, authorized_live_run=args.authorized_live_run, output_dir=args.output_dir, episode_id=args.episode_id, preflight_only=args.preflight_only)
        payload = result if isinstance(result, Mapping) else result.as_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

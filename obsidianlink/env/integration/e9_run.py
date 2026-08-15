"""Explicitly authorized P1 E9 real-MineRL server-side fluid-truth entrypoint.

Imports and ``--check`` are offline-safe. The live path resolves production
MineRL only after the exact per-variant gate and performs exactly one bounded
``use_item``. One authorization token is one variant, one fresh episode, and
one fluid stimulus. It never runs Gradle, models, solvers, water+lava
together, or later validation cases. E9 success is ``fluid_truth_ok``, not
bucket usage success.
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
from obsidianlink.env.integration.e9_adapter import MineRLE9FluidTruthAdapter
from obsidianlink.env.integration.e9_config import (
    E9_AGENT_ID,
    E9_DURATION_TICKS,
    E9_EXPECTED_BEFORE_FLUIDS,
    E9_INITIAL_PITCH,
    E9_INITIAL_YAW,
    E9_LAVA_CALIBRATION,
    E9_PROBE_GRID_CELLS,
    E9_PROBE_WORLD_CELLS,
    E9_SPAWN_WORLD,
    E9_TARGET_WORLD_CELL,
    E9_WATER_CALIBRATION,
    build_e9_compatibility_task,
    e9_calibration,
)
from obsidianlink.env.validation import E9_SERVER_FLUID_TRUTH_CASE, EnvironmentValidationRunner
from obsidianlink.env.validation.contract import EnvironmentValidationId
from obsidianlink.env.validation.result import EnvironmentValidationResult
from obsidianlink.env.validation.truth import FluidCalibrationVariant, validate_fluid_variant


ROOT = Path(__file__).resolve().parents[3]
FORMAL_E9_RUNS_ROOT = (ROOT / "runs" / "p1_e9_fluid_truth").resolve()
RUNTIME_LOGS_ROOT = (ROOT / "logs").resolve()
EXECUTION_MODE_AUTHORIZED_LIVE_E9_WATER = "authorized_live_e9_water"
EXECUTION_MODE_AUTHORIZED_LIVE_E9_LAVA = "authorized_live_e9_lava"
AUTHORIZED_LIVE_E9_WATER_RUN_VALUE = "e9_water_fluid_truth"
AUTHORIZED_LIVE_E9_LAVA_RUN_VALUE = "e9_lava_fluid_truth"
_VARIANT_GATES = {
    FluidCalibrationVariant.WATER: (
        EXECUTION_MODE_AUTHORIZED_LIVE_E9_WATER,
        AUTHORIZED_LIVE_E9_WATER_RUN_VALUE,
    ),
    FluidCalibrationVariant.LAVA: (
        EXECUTION_MODE_AUTHORIZED_LIVE_E9_LAVA,
        AUTHORIZED_LIVE_E9_LAVA_RUN_VALUE,
    ),
}

_PROCESS_LIVE_RUN_STARTED = False
_PROCESS_LIVE_RUN_LOCK = threading.Lock()


class E9AuthorizationError(ValueError):
    """Raised when the exact E9 live gate or preflight is invalid."""


def resolve_e9_live_variant(*, execution_mode: object, authorized_live_run: object) -> FluidCalibrationVariant:
    for variant, (mode, token) in _VARIANT_GATES.items():
        if execution_mode == mode and authorized_live_run == token:
            return variant
    raise E9AuthorizationError(
        "execution_mode and authorized_live_run must be exactly one E9 variant gate"
    )


@dataclass(frozen=True)
class E9LogEvidence:
    kind: str
    path: str
    sha256: str
    summary: str

    def __post_init__(self) -> None:
        if self.kind not in {"minecraft", "jvm_crash", "process_watcher"}:
            raise ValueError("unknown E9 log evidence kind")
        if not Path(self.path).is_absolute():
            raise ValueError("E9 log evidence path must be absolute")
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError("E9 log evidence sha256 must be lowercase hex")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("E9 log evidence summary must be non-empty")

    def as_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class E9MineRLRunRecord:
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
    log_evidence: tuple[E9LogEvidence, ...]

    def __post_init__(self) -> None:
        variant = validate_fluid_variant(self.variant)
        object.__setattr__(self, "variant", variant.value)
        expected_mode, expected_token = _VARIANT_GATES[variant]
        if self.execution_mode != expected_mode:
            raise ValueError("execution_mode must match the E9 variant gate")
        if self.authorized_live_run != expected_token:
            raise ValueError("authorized_live_run must match the E9 variant token")
        if not isinstance(self.lifecycle, EnvironmentValidationResult) or self.lifecycle.check_id is not EnvironmentValidationId.E9:
            raise ValueError("lifecycle must be an E9 validation result")
        if not isinstance(self.cleanup, E0CleanupStatus):
            raise ValueError("cleanup must be E0CleanupStatus")
        if any(type(value) is not bool for value in (self.opened, self.authorization_accepted, self.real_execution_performed)):
            raise ValueError("record flags must be bool")
        if self.failure_cause not in {"unknown", "minecraft_native_crash"}:
            raise ValueError("unknown E9 failure cause")
        if not isinstance(self.log_evidence, tuple) or not all(
            isinstance(value, E9LogEvidence) for value in self.log_evidence
        ):
            raise ValueError("log_evidence must be a tuple of E9LogEvidence")
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


def _make_log_evidence(path: Path, kind: str) -> E9LogEvidence:
    data = path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    return E9LogEvidence(
        kind=kind,
        path=str(path.resolve()),
        sha256=hashlib.sha256(data).hexdigest(),
        summary=_log_summary(text, kind),
    )


def _collect_runtime_log_evidence(
    before: Mapping[str, tuple[int, int]],
) -> tuple[str, tuple[E9LogEvidence, ...]]:
    changed: list[Path] = []
    for path in _runtime_log_paths():
        state = (path.stat().st_size, path.stat().st_mtime_ns)
        if before.get(str(path)) != state:
            changed.append(path)
    evidence: list[E9LogEvidence] = []
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


def reset_authorized_e9_process_guards_for_tests() -> None:
    global _PROCESS_LIVE_RUN_STARTED
    with _PROCESS_LIVE_RUN_LOCK:
        _PROCESS_LIVE_RUN_STARTED = False


def assert_e9_live_authorized(*, execution_mode: object, authorized_live_run: object, allow_gradle: object = False) -> FluidCalibrationVariant:
    variant = resolve_e9_live_variant(
        execution_mode=execution_mode, authorized_live_run=authorized_live_run
    )
    if allow_gradle is not False:
        raise E9AuthorizationError("Gradle is not authorized for E9")
    return variant


def _validate_catalog_policy() -> None:
    catalog = load_task_catalog(ROOT / "benchmark/catalog/tasks.json")
    if catalog.active_phase != "P1-REAL-MINERL-ENVIRONMENT-VALIDATION":
        raise E9AuthorizationError("active catalog phase must remain P1")
    if any(entry.live_run_allowed for entry in catalog.entries):
        raise E9AuthorizationError("catalog live_run_allowed must remain false")


def _validate_configuration() -> None:
    if not (
        E9_SERVER_FLUID_TRUTH_CASE.check_id is EnvironmentValidationId.E9
        and E9_SERVER_FLUID_TRUTH_CASE.name == "water_lava_fluid_truth"
        and E9_SERVER_FLUID_TRUTH_CASE.requires_server_truth
        and E9_SERVER_FLUID_TRUTH_CASE.calibration_only
        and E9_WATER_CALIBRATION.bucket_item == "water_bucket"
        and E9_LAVA_CALIBRATION.bucket_item == "lava_bucket"
        and E9_WATER_CALIBRATION.expected_flow_state == "source"
        and E9_LAVA_CALIBRATION.expected_flow_state == "source"
        and E9_PROBE_WORLD_CELLS == ((0, 4, 1), (0, 5, 1), (0, 5, 0))
        and E9_PROBE_GRID_CELLS == ((0, 0, 1), (0, 1, 1), (0, 1, 0))
        and E9_SPAWN_WORLD == (0, 4, 0)
        and E9_DURATION_TICKS == 1
        and (E9_INITIAL_YAW, E9_INITIAL_PITCH) == (0.0, 60.0)
        and E9_EXPECTED_BEFORE_FLUIDS[E9_TARGET_WORLD_CELL] == ("none", "none")
    ):
        raise E9AuthorizationError("frozen E9 calibration differs")
    for variant in (FluidCalibrationVariant.WATER, FluidCalibrationVariant.LAVA):
        task = build_e9_compatibility_task("p1-e9-preflight", variant)
        calibration = e9_calibration(variant)
        if dict(task.initial_inventories[E9_AGENT_ID]) != dict(calibration.initial_inventory):
            raise E9AuthorizationError("E9 compatibility inventory differs")
        if task.spawn_positions[E9_AGENT_ID] != E9_SPAWN_WORLD:
            raise E9AuthorizationError("E9 flat-ground spawn differs")
    if MineRLE9FluidTruthAdapter(episode_id="p1-e9-preflight")._backend is not None:
        raise E9AuthorizationError("E9 adapter construction created a backend")


def check_e9_live_runner() -> dict[str, Any]:
    _validate_catalog_policy()
    _validate_configuration()
    return {
        "authorized_live_run_required": {
            "lava": AUTHORIZED_LIVE_E9_LAVA_RUN_VALUE,
            "water": AUTHORIZED_LIVE_E9_WATER_RUN_VALUE,
        },
        "calibration_only": True,
        "check_id": "E9",
        "execution_mode_required": {
            "lava": EXECUTION_MODE_AUTHORIZED_LIVE_E9_LAVA,
            "water": EXECUTION_MODE_AUTHORIZED_LIVE_E9_WATER,
        },
        "expected_before_fluid": {
            "flow_state": "none",
            "fluid_type": "none",
        },
        "gradle_authorized": False,
        "integration_verified": False,
        "name": "water_lava_fluid_truth",
        "one_token_one_variant": True,
        "probe_count": len(E9_PROBE_WORLD_CELLS),
        "probe_grid_cells": [list(cell) for cell in E9_PROBE_GRID_CELLS],
        "probe_world_cells": [list(cell) for cell in E9_PROBE_WORLD_CELLS],
        "production_backend_constructed": False,
        "real_execution_performed": False,
        "status": "ok",
        "truth_missing_required": 0,
        "variants": {
            "lava": {
                "expected_flow_state": E9_LAVA_CALIBRATION.expected_flow_state,
                "expected_fluid_type": E9_LAVA_CALIBRATION.expected_fluid_type,
                "stimulus_target": E9_LAVA_CALIBRATION.bucket_item,
            },
            "water": {
                "expected_flow_state": E9_WATER_CALIBRATION.expected_flow_state,
                "expected_fluid_type": E9_WATER_CALIBRATION.expected_fluid_type,
                "stimulus_target": E9_WATER_CALIBRATION.bucket_item,
            },
        },
        "verification_level": "unit_verified",
    }


def _validate_output_dir(output_dir: Path, variant: FluidCalibrationVariant) -> Path:
    if not isinstance(output_dir, Path) or not output_dir.is_absolute():
        raise E9AuthorizationError("output_dir must be an absolute pathlib.Path")
    resolved = output_dir.resolve()
    if resolved.exists() or resolved.is_symlink():
        raise E9AuthorizationError(f"output_dir must not exist: {resolved}")
    variant_root = (FORMAL_E9_RUNS_ROOT / variant.value).resolve()
    try:
        resolved.relative_to(variant_root)
    except ValueError as exc:
        raise E9AuthorizationError(f"output_dir must be under {variant_root}") from exc
    if resolved == variant_root or resolved.parent != variant_root:
        raise E9AuthorizationError("output_dir must be a unique direct child of the variant directory")
    return resolved


def _e9_env_factory(task: TaskInstance) -> Any:
    from obsidianlink.env.portal_spec import PortalA0EnvSpec

    initial_inventory = tuple(
        {"type": item, "quantity": quantity}
        for item, quantity in task.initial_inventories[E9_AGENT_ID].items()
        if quantity > 0
    )
    specification = PortalA0EnvSpec(
        max_episode_steps=task.limits["max_environment_steps"],
        max_game_time_seconds=task.limits["max_game_time_seconds"],
        initial_inventory=initial_inventory,
        initial_position=task.spawn_positions[E9_AGENT_ID],
        include_agent_start_placement=True,
        grid_at_spawn=True,
        initial_yaw=E9_INITIAL_YAW,
        initial_pitch=E9_INITIAL_PITCH,
    )
    return specification.make()


def _production_backend_cls() -> type:
    from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
    return MineRLEnvironmentBackend


def _production_backend_kwargs() -> dict[str, Any]:
    return {"max_reset_attempts": 1, "env_factory": _e9_env_factory}


def preflight_authorized_e9(*, execution_mode: str, authorized_live_run: str, output_dir: Path | None = None, allow_gradle: bool = False) -> dict[str, Any]:
    variant = assert_e9_live_authorized(execution_mode=execution_mode, authorized_live_run=authorized_live_run, allow_gradle=allow_gradle)
    _validate_catalog_policy()
    _validate_configuration()
    payload = check_e9_live_runner()
    calibration = e9_calibration(variant)
    payload.update(
        {
            "execution_mode": execution_mode,
            "authorized_live_run": authorized_live_run,
            "requires_server_truth": True,
            "variant": variant.value,
            "stimulus_target": calibration.bucket_item,
            "expected_fluid_type": calibration.expected_fluid_type,
            "expected_flow_state": calibration.expected_flow_state,
        }
    )
    if output_dir is not None:
        payload["output_dir"] = str(_validate_output_dir(output_dir, variant))
    return payload


def _write_evidence(record: E9MineRLRunRecord, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "e9_fluid_truth.json"
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
                "planned_tested_stimulus_count": 1,
                "variant": record.variant,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return path


def run_authorized_e9_minerl(*, execution_mode: str, authorized_live_run: str, output_dir: Path, episode_id: str = "p1-e9-fluid-truth", allow_gradle: bool = False, preflight_only: bool = False) -> E9MineRLRunRecord | dict[str, Any]:
    preflight = preflight_authorized_e9(execution_mode=execution_mode, authorized_live_run=authorized_live_run, output_dir=output_dir, allow_gradle=allow_gradle)
    if preflight_only:
        return preflight
    variant = resolve_e9_live_variant(execution_mode=execution_mode, authorized_live_run=authorized_live_run)
    global _PROCESS_LIVE_RUN_STARTED
    with _PROCESS_LIVE_RUN_LOCK:
        if _PROCESS_LIVE_RUN_STARTED:
            raise E9AuthorizationError("authorized E9 allows one real run per process")
        _PROCESS_LIVE_RUN_STARTED = True
    log_snapshot = _snapshot_runtime_logs()
    backend_cls = _production_backend_cls()
    factory = MineRLE9FluidTruthAdapter.lifecycle_factory(
        episode_id=episode_id,
        variant=variant,
        backend_cls=backend_cls,
        backend_kwargs=_production_backend_kwargs(),
    )
    holder: dict[str, MineRLE9FluidTruthAdapter | None] = {"adapter": None}

    def capture() -> MineRLE9FluidTruthAdapter:
        adapter = factory()
        holder["adapter"] = adapter
        return adapter

    lifecycle = EnvironmentValidationRunner().run(
        E9_SERVER_FLUID_TRUTH_CASE,
        capture,
        episode_id=episode_id,
        fluid_variant=variant.value,
        requested_duration_ticks=E9_DURATION_TICKS,
    )
    adapter = holder["adapter"]
    cleanup = adapter.cleanup_status() if adapter is not None else E0CleanupStatus(lifecycle.closed, lifecycle.closed, lifecycle.closed, lifecycle.closed)
    detected_cause, log_evidence = _collect_runtime_log_evidence(log_snapshot)
    failure_cause = (
        detected_cause if lifecycle.outcome == "reset_failed" else "unknown"
    )
    record = E9MineRLRunRecord(
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
    parser = argparse.ArgumentParser(prog="obsidianlink.env.integration.e9_run", description="Authorized P1 E9 server-side fluid-truth entrypoint; --check is offline-safe.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execution-mode")
    parser.add_argument("--authorized-live-run")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episode-id", default="p1-e9-fluid-truth")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        if any(value is not None for value in (args.execution_mode, args.authorized_live_run, args.output_dir)) or args.preflight_only:
            raise E9AuthorizationError("--check cannot be combined with live arguments")
        payload: Mapping[str, Any] = check_e9_live_runner()
    else:
        if args.output_dir is None:
            raise E9AuthorizationError("--output-dir is required for E9 live/preflight")
        result = run_authorized_e9_minerl(execution_mode=args.execution_mode, authorized_live_run=args.authorized_live_run, output_dir=args.output_dir, episode_id=args.episode_id, preflight_only=args.preflight_only)
        payload = result if isinstance(result, Mapping) else result.as_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

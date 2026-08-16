"""Explicitly authorized P1 E10 real-MineRL obsidian-conversion entrypoint.

Imports and ``--check`` are offline-safe. The live path resolves production
MineRL only after the exact gate and performs exactly one bounded
``use_item(water_bucket)`` plus a bounded observation window. It never
runs Gradle, models, solvers, E11, or E12. E10 success is
``obsidian_conversion_ok`` from server truth, not bucket-use success.
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
from obsidianlink.env.integration.e10_adapter import MineRLE10ObsidianAdapter
from obsidianlink.env.integration.e10_config import (
    E10_AGENT_ID,
    E10_CALIBRATION,
    E10_COMPATIBILITY_INVENTORY,
    E10_CONTROL_WORLD_CELLS,
    E10_DURATION_TICKS,
    E10_EXPECTED_AFTER_BLOCK,
    E10_EXPECTED_BEFORE_BLOCK,
    E10_INITIAL_PITCH,
    E10_INITIAL_YAW,
    E10_OBSERVATION_WINDOW_TICKS,
    E10_PROBE_GRID_CELLS,
    E10_PROBE_WORLD_CELLS,
    E10_SPAWN_WORLD,
    E10_STIMULUS_ITEM_NAME,
    E10_TARGET_WORLD_CELL,
    E10_WATER_WORLD_CELL,
    build_e10_compatibility_task,
)
from obsidianlink.env.validation import E10_OBSIDIAN_CONVERSION_CASE, EnvironmentValidationRunner
from obsidianlink.env.validation.contract import EnvironmentValidationId
from obsidianlink.env.validation.result import EnvironmentValidationResult


ROOT = Path(__file__).resolve().parents[3]
FORMAL_E10_RUNS_ROOT = (ROOT / "runs" / "p1_e10_obsidian_conversion").resolve()
RUNTIME_LOGS_ROOT = (ROOT / "logs").resolve()
EXECUTION_MODE_AUTHORIZED_LIVE_E10 = "authorized_live_e10"
AUTHORIZED_LIVE_E10_RUN_VALUE = "e10_obsidian_conversion"

_PROCESS_LIVE_RUN_STARTED = False
_PROCESS_LIVE_RUN_LOCK = threading.Lock()


class E10AuthorizationError(ValueError):
    """Raised when the exact E10 live gate or preflight is invalid."""


@dataclass(frozen=True)
class E10LogEvidence:
    kind: str
    path: str
    sha256: str
    summary: str

    def __post_init__(self) -> None:
        if self.kind not in {"minecraft", "jvm_crash", "process_watcher"}:
            raise ValueError("unknown E10 log evidence kind")
        if not Path(self.path).is_absolute():
            raise ValueError("E10 log evidence path must be absolute")
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError("E10 log evidence sha256 must be lowercase hex")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("E10 log evidence summary must be non-empty")

    def as_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class E10MineRLRunRecord:
    execution_mode: str
    authorized_live_run: str
    backend_identity: str
    opened: bool
    authorization_accepted: bool
    real_execution_performed: bool
    lifecycle: EnvironmentValidationResult
    cleanup: E0CleanupStatus
    failure_cause: str
    log_evidence: tuple[E10LogEvidence, ...]

    def __post_init__(self) -> None:
        if self.execution_mode != EXECUTION_MODE_AUTHORIZED_LIVE_E10:
            raise ValueError("execution_mode must be authorized_live_e10")
        if self.authorized_live_run != AUTHORIZED_LIVE_E10_RUN_VALUE:
            raise ValueError("authorized_live_run must be e10_obsidian_conversion")
        if not isinstance(self.lifecycle, EnvironmentValidationResult) or self.lifecycle.check_id is not EnvironmentValidationId.E10:
            raise ValueError("lifecycle must be an E10 validation result")
        if not isinstance(self.cleanup, E0CleanupStatus):
            raise ValueError("cleanup must be E0CleanupStatus")
        if any(type(value) is not bool for value in (self.opened, self.authorization_accepted, self.real_execution_performed)):
            raise ValueError("record flags must be bool")
        if self.failure_cause not in {"unknown", "minecraft_native_crash"}:
            raise ValueError("unknown E10 failure cause")
        if not isinstance(self.log_evidence, tuple) or not all(
            isinstance(value, E10LogEvidence) for value in self.log_evidence
        ):
            raise ValueError("log_evidence must be a tuple of E10LogEvidence")
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


def _make_log_evidence(path: Path, kind: str) -> E10LogEvidence:
    data = path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    return E10LogEvidence(
        kind=kind,
        path=str(path.resolve()),
        sha256=hashlib.sha256(data).hexdigest(),
        summary=_log_summary(text, kind),
    )


def _collect_runtime_log_evidence(
    before: Mapping[str, tuple[int, int]],
) -> tuple[str, tuple[E10LogEvidence, ...]]:
    changed: list[Path] = []
    for path in _runtime_log_paths():
        state = (path.stat().st_size, path.stat().st_mtime_ns)
        if before.get(str(path)) != state:
            changed.append(path)
    evidence: list[E10LogEvidence] = []
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


def reset_authorized_e10_process_guards_for_tests() -> None:
    global _PROCESS_LIVE_RUN_STARTED
    with _PROCESS_LIVE_RUN_LOCK:
        _PROCESS_LIVE_RUN_STARTED = False


def assert_e10_live_authorized(*, execution_mode: object, authorized_live_run: object, allow_gradle: object = False) -> None:
    if execution_mode != EXECUTION_MODE_AUTHORIZED_LIVE_E10:
        raise E10AuthorizationError("execution_mode must be exactly authorized_live_e10")
    if authorized_live_run != AUTHORIZED_LIVE_E10_RUN_VALUE:
        raise E10AuthorizationError("authorized_live_run must be exactly e10_obsidian_conversion")
    if allow_gradle is not False:
        raise E10AuthorizationError("Gradle is not authorized for E10")


def _validate_catalog_policy() -> None:
    catalog = load_task_catalog(ROOT / "benchmark/catalog/tasks.json")
    if catalog.active_phase != "P1-REAL-MINERL-ENVIRONMENT-VALIDATION":
        raise E10AuthorizationError("active catalog phase must remain P1")
    if any(entry.live_run_allowed for entry in catalog.entries):
        raise E10AuthorizationError("catalog live_run_allowed must remain false")


def _validate_configuration() -> None:
    if not (
        E10_OBSIDIAN_CONVERSION_CASE.check_id is EnvironmentValidationId.E10
        and E10_OBSIDIAN_CONVERSION_CASE.name == "vanilla_water_lava_to_obsidian"
        and E10_OBSIDIAN_CONVERSION_CASE.requires_server_truth
        and E10_OBSIDIAN_CONVERSION_CASE.calibration_only
        and E10_STIMULUS_ITEM_NAME == "water_bucket"
        and E10_EXPECTED_BEFORE_BLOCK == "lava"
        and E10_EXPECTED_AFTER_BLOCK == "obsidian"
        and E10_PROBE_WORLD_CELLS == ((0, 4, 2), (0, 4, 1), (0, 5, 1), (0, 5, 2))
        and E10_PROBE_GRID_CELLS == ((0, 0, 2), (0, 0, 1), (0, 1, 1), (0, 1, 2))
        and E10_TARGET_WORLD_CELL == (0, 4, 2)
        and E10_WATER_WORLD_CELL == (0, 4, 1)
        and E10_CONTROL_WORLD_CELLS == ((0, 5, 1), (0, 5, 2))
        and E10_SPAWN_WORLD == (0, 4, 0)
        and E10_DURATION_TICKS == 1
        and E10_OBSERVATION_WINDOW_TICKS == 5
        and (E10_INITIAL_YAW, E10_INITIAL_PITCH) == (0.0, 60.0)
        and E10_CALIBRATION.expected_before_fluids[E10_TARGET_WORLD_CELL] == ("lava", "source")
    ):
        raise E10AuthorizationError("frozen E10 calibration differs")
    task = build_e10_compatibility_task("p1-e10-preflight")
    if dict(task.initial_inventories[E10_AGENT_ID]) != E10_COMPATIBILITY_INVENTORY:
        raise E10AuthorizationError("E10 compatibility inventory differs")
    if task.spawn_positions[E10_AGENT_ID] != E10_SPAWN_WORLD:
        raise E10AuthorizationError("E10 flat-ground spawn differs")
    if task.scenario_parameters.get("obsidian_preplaced") is not False:
        raise E10AuthorizationError("E10 must not pre-place target obsidian")
    if MineRLE10ObsidianAdapter(episode_id="p1-e10-preflight")._backend is not None:
        raise E10AuthorizationError("E10 adapter construction created a backend")


def check_e10_live_runner() -> dict[str, Any]:
    _validate_catalog_policy()
    _validate_configuration()
    return {
        "authorized_live_run_required": AUTHORIZED_LIVE_E10_RUN_VALUE,
        "calibration_only": True,
        "check_id": "E10",
        "execution_mode_required": EXECUTION_MODE_AUTHORIZED_LIVE_E10,
        "expected_after_block": E10_EXPECTED_AFTER_BLOCK,
        "expected_before_block": E10_EXPECTED_BEFORE_BLOCK,
        "expected_before_fluid": {
            "flow_state": "source",
            "fluid_type": "lava",
        },
        "gradle_authorized": False,
        "integration_verified": False,
        "lava_preplaced": True,
        "name": "vanilla_water_lava_to_obsidian",
        "observation_window_ticks": E10_OBSERVATION_WINDOW_TICKS,
        "obsidian_preplaced": False,
        "probe_count": len(E10_PROBE_WORLD_CELLS),
        "probe_grid_cells": [list(cell) for cell in E10_PROBE_GRID_CELLS],
        "probe_world_cells": [list(cell) for cell in E10_PROBE_WORLD_CELLS],
        "production_backend_constructed": False,
        "real_execution_performed": False,
        "status": "ok",
        "stimulus": {
            "action_type": "use_item",
            "duration_ticks": E10_DURATION_TICKS,
            "target": E10_STIMULUS_ITEM_NAME,
        },
        "target_world_cell": list(E10_TARGET_WORLD_CELL),
        "truth_missing_required": 0,
        "verification_level": "unit_verified",
        "water_world_cell": list(E10_WATER_WORLD_CELL),
    }


def _validate_output_dir(output_dir: Path) -> Path:
    if not isinstance(output_dir, Path) or not output_dir.is_absolute():
        raise E10AuthorizationError("output_dir must be an absolute pathlib.Path")
    resolved = output_dir.resolve()
    if resolved.exists() or resolved.is_symlink():
        raise E10AuthorizationError(f"output_dir must not exist: {resolved}")
    try:
        resolved.relative_to(FORMAL_E10_RUNS_ROOT)
    except ValueError as exc:
        raise E10AuthorizationError(f"output_dir must be under {FORMAL_E10_RUNS_ROOT}") from exc
    if resolved == FORMAL_E10_RUNS_ROOT or resolved.parent != FORMAL_E10_RUNS_ROOT:
        raise E10AuthorizationError("output_dir must be a unique direct child")
    return resolved


def _e10_env_factory(task: TaskInstance) -> Any:
    from obsidianlink.env.portal_spec import PortalA0EnvSpec

    initial_inventory = tuple(
        {"type": item, "quantity": quantity}
        for item, quantity in task.initial_inventories[E10_AGENT_ID].items()
        if quantity > 0
    )
    specification = PortalA0EnvSpec(
        max_episode_steps=task.limits["max_environment_steps"],
        max_game_time_seconds=task.limits["max_game_time_seconds"],
        initial_inventory=initial_inventory,
        initial_position=task.spawn_positions[E10_AGENT_ID],
        include_agent_start_placement=True,
        grid_at_spawn=True,
        initial_yaw=E10_INITIAL_YAW,
        initial_pitch=E10_INITIAL_PITCH,
    )
    return specification.make()


def _production_backend_cls() -> type:
    from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
    return MineRLEnvironmentBackend


def _production_backend_kwargs() -> dict[str, Any]:
    return {"max_reset_attempts": 1, "env_factory": _e10_env_factory}


def preflight_authorized_e10(*, execution_mode: str, authorized_live_run: str, output_dir: Path | None = None, allow_gradle: bool = False) -> dict[str, Any]:
    assert_e10_live_authorized(execution_mode=execution_mode, authorized_live_run=authorized_live_run, allow_gradle=allow_gradle)
    _validate_catalog_policy()
    _validate_configuration()
    payload = check_e10_live_runner()
    payload.update({"execution_mode": execution_mode, "authorized_live_run": authorized_live_run, "requires_server_truth": True})
    if output_dir is not None:
        payload["output_dir"] = str(_validate_output_dir(output_dir))
    return payload


def _write_evidence(record: E10MineRLRunRecord, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "result.json"
    path.write_text(json.dumps(record.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "control_world_cells": [list(cell) for cell in E10_CONTROL_WORLD_CELLS],
                "duration_ticks": E10_DURATION_TICKS,
                "expected_after_block": E10_EXPECTED_AFTER_BLOCK,
                "expected_before_block": E10_EXPECTED_BEFORE_BLOCK,
                "observation_window_ticks": E10_OBSERVATION_WINDOW_TICKS,
                "probe_grid_cells": [list(cell) for cell in E10_PROBE_GRID_CELLS],
                "probe_world_cells": [list(cell) for cell in E10_PROBE_WORLD_CELLS],
                "spawn_world": list(E10_SPAWN_WORLD),
                "stimulus_item": E10_STIMULUS_ITEM_NAME,
                "target_world_cell": list(E10_TARGET_WORLD_CELL),
                "water_world_cell": list(E10_WATER_WORLD_CELL),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    (output_dir / "authorization.json").write_text(
        json.dumps(
            {
                "authorized_live_run": AUTHORIZED_LIVE_E10_RUN_VALUE,
                "catalog_live_run_allowed_remains_false": True,
                "execution_mode": EXECUTION_MODE_AUTHORIZED_LIVE_E10,
                "gradle_authorized": False,
                "integration_verified": False,
                "model_api_authorized": False,
                "obsidian_preplaced": False,
                "planned_tested_stimulus_count": 1,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return path


def run_authorized_e10_minerl(*, execution_mode: str, authorized_live_run: str, output_dir: Path, episode_id: str = "p1-e10-obsidian-conversion", allow_gradle: bool = False, preflight_only: bool = False) -> E10MineRLRunRecord | dict[str, Any]:
    preflight = preflight_authorized_e10(execution_mode=execution_mode, authorized_live_run=authorized_live_run, output_dir=output_dir, allow_gradle=allow_gradle)
    if preflight_only:
        return preflight
    global _PROCESS_LIVE_RUN_STARTED
    with _PROCESS_LIVE_RUN_LOCK:
        if _PROCESS_LIVE_RUN_STARTED:
            raise E10AuthorizationError("authorized E10 allows one real run per process")
        _PROCESS_LIVE_RUN_STARTED = True
    log_snapshot = _snapshot_runtime_logs()
    backend_cls = _production_backend_cls()
    factory = MineRLE10ObsidianAdapter.lifecycle_factory(
        episode_id=episode_id,
        backend_cls=backend_cls,
        backend_kwargs=_production_backend_kwargs(),
    )
    holder: dict[str, MineRLE10ObsidianAdapter | None] = {"adapter": None}

    def capture() -> MineRLE10ObsidianAdapter:
        adapter = factory()
        holder["adapter"] = adapter
        return adapter

    lifecycle = EnvironmentValidationRunner().run(
        E10_OBSIDIAN_CONVERSION_CASE,
        capture,
        episode_id=episode_id,
        requested_duration_ticks=E10_DURATION_TICKS,
    )
    adapter = holder["adapter"]
    cleanup = adapter.cleanup_status() if adapter is not None else E0CleanupStatus(lifecycle.closed, lifecycle.closed, lifecycle.closed, lifecycle.closed)
    detected_cause, log_evidence = _collect_runtime_log_evidence(log_snapshot)
    failure_cause = (
        detected_cause if lifecycle.outcome == "reset_failed" else "unknown"
    )
    record = E10MineRLRunRecord(
        execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E10,
        authorized_live_run=AUTHORIZED_LIVE_E10_RUN_VALUE,
        backend_identity=getattr(adapter, "backend_identity", backend_cls.__name__),
        opened=False if adapter is None else adapter.open_succeeded,
        authorization_accepted=True,
        real_execution_performed=backend_cls.__name__ == BACKEND_IDENTITY,
        lifecycle=lifecycle,
        cleanup=cleanup,
        failure_cause=failure_cause,
        log_evidence=log_evidence,
    )
    _write_evidence(record, _validate_output_dir(output_dir))
    return record


def build_parser():
    import argparse
    parser = argparse.ArgumentParser(prog="obsidianlink.env.integration.e10_run", description="Authorized P1 E10 vanilla obsidian-conversion entrypoint; --check is offline-safe.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execution-mode")
    parser.add_argument("--authorized-live-run")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episode-id", default="p1-e10-obsidian-conversion")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        if any(value is not None for value in (args.execution_mode, args.authorized_live_run, args.output_dir)) or args.preflight_only:
            raise E10AuthorizationError("--check cannot be combined with live arguments")
        payload: Mapping[str, Any] = check_e10_live_runner()
    else:
        if args.output_dir is None:
            raise E10AuthorizationError("--output-dir is required for E10 live/preflight")
        result = run_authorized_e10_minerl(execution_mode=args.execution_mode, authorized_live_run=args.authorized_live_run, output_dir=args.output_dir, episode_id=args.episode_id, preflight_only=args.preflight_only)
        payload = result if isinstance(result, Mapping) else result.as_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Explicitly authorized P1 E8 real-MineRL server-side block-truth entrypoint.

Imports and ``--check`` are offline-safe. The live path resolves production
MineRL only after the exact gate and performs exactly one bounded
``place_block`` stimulus. It never runs Gradle, models, solvers, or later
validation cases. E8 success is ``block_truth_ok``, not placement success.
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
from obsidianlink.env.integration.e8_adapter import MineRLE8BlockTruthAdapter
from obsidianlink.env.integration.e8_config import (
    E8_AGENT_ID,
    E8_COMPATIBILITY_INVENTORY,
    E8_DURATION_TICKS,
    E8_EXPECTED_AFTER_BLOCKS,
    E8_EXPECTED_BEFORE_BLOCKS,
    E8_INITIAL_PITCH,
    E8_INITIAL_YAW,
    E8_PROBE_GRID_CELLS,
    E8_PROBE_WORLD_CELLS,
    E8_SPAWN_WORLD,
    E8_STIMULUS_BLOCK,
    build_e8_compatibility_task,
)
from obsidianlink.env.validation import E8_SERVER_BLOCK_TRUTH_CASE, EnvironmentValidationRunner
from obsidianlink.env.validation.contract import EnvironmentValidationId
from obsidianlink.env.validation.result import EnvironmentValidationResult


ROOT = Path(__file__).resolve().parents[3]
FORMAL_E8_RUNS_ROOT = (ROOT / "runs" / "p1_e8_block_truth").resolve()
RUNTIME_LOGS_ROOT = (ROOT / "logs").resolve()
EXECUTION_MODE_AUTHORIZED_LIVE_E8 = "authorized_live_e8"
AUTHORIZED_LIVE_E8_RUN_VALUE = "e8_block_truth"

_PROCESS_LIVE_RUN_STARTED = False
_PROCESS_LIVE_RUN_LOCK = threading.Lock()


class E8AuthorizationError(ValueError):
    """Raised when the exact E8 live gate or preflight is invalid."""


@dataclass(frozen=True)
class E8LogEvidence:
    kind: str
    path: str
    sha256: str
    summary: str

    def __post_init__(self) -> None:
        if self.kind not in {"minecraft", "jvm_crash", "process_watcher"}:
            raise ValueError("unknown E8 log evidence kind")
        if not Path(self.path).is_absolute():
            raise ValueError("E8 log evidence path must be absolute")
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError("E8 log evidence sha256 must be lowercase hex")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("E8 log evidence summary must be non-empty")

    def as_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class E8MineRLRunRecord:
    execution_mode: str
    authorized_live_run: str
    backend_identity: str
    opened: bool
    authorization_accepted: bool
    real_execution_performed: bool
    lifecycle: EnvironmentValidationResult
    cleanup: E0CleanupStatus
    failure_cause: str
    log_evidence: tuple[E8LogEvidence, ...]

    def __post_init__(self) -> None:
        if self.execution_mode != EXECUTION_MODE_AUTHORIZED_LIVE_E8:
            raise ValueError("execution_mode must be authorized_live_e8")
        if self.authorized_live_run != AUTHORIZED_LIVE_E8_RUN_VALUE:
            raise ValueError("authorized_live_run must be e8_block_truth")
        if not isinstance(self.lifecycle, EnvironmentValidationResult) or self.lifecycle.check_id is not EnvironmentValidationId.E8:
            raise ValueError("lifecycle must be an E8 validation result")
        if not isinstance(self.cleanup, E0CleanupStatus):
            raise ValueError("cleanup must be E0CleanupStatus")
        if any(type(value) is not bool for value in (self.opened, self.authorization_accepted, self.real_execution_performed)):
            raise ValueError("record flags must be bool")
        if self.failure_cause not in {"unknown", "minecraft_native_crash"}:
            raise ValueError("unknown E8 failure cause")
        if not isinstance(self.log_evidence, tuple) or not all(
            isinstance(value, E8LogEvidence) for value in self.log_evidence
        ):
            raise ValueError("log_evidence must be a tuple of E8LogEvidence")
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


def _make_log_evidence(path: Path, kind: str) -> E8LogEvidence:
    data = path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    return E8LogEvidence(
        kind=kind,
        path=str(path.resolve()),
        sha256=hashlib.sha256(data).hexdigest(),
        summary=_log_summary(text, kind),
    )


def _collect_runtime_log_evidence(
    before: Mapping[str, tuple[int, int]],
) -> tuple[str, tuple[E8LogEvidence, ...]]:
    changed: list[Path] = []
    for path in _runtime_log_paths():
        state = (path.stat().st_size, path.stat().st_mtime_ns)
        if before.get(str(path)) != state:
            changed.append(path)
    evidence: list[E8LogEvidence] = []
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


def reset_authorized_e8_process_guards_for_tests() -> None:
    global _PROCESS_LIVE_RUN_STARTED
    with _PROCESS_LIVE_RUN_LOCK:
        _PROCESS_LIVE_RUN_STARTED = False


def assert_e8_live_authorized(*, execution_mode: object, authorized_live_run: object, allow_gradle: object = False) -> None:
    if execution_mode != EXECUTION_MODE_AUTHORIZED_LIVE_E8:
        raise E8AuthorizationError("execution_mode must be exactly authorized_live_e8")
    if authorized_live_run != AUTHORIZED_LIVE_E8_RUN_VALUE:
        raise E8AuthorizationError("authorized_live_run must be exactly e8_block_truth")
    if allow_gradle is not False:
        raise E8AuthorizationError("Gradle is not authorized for E8")


def _validate_catalog_policy() -> None:
    catalog = load_task_catalog(ROOT / "benchmark/catalog/tasks.json")
    if catalog.active_phase != "P1-REAL-MINERL-ENVIRONMENT-VALIDATION":
        raise E8AuthorizationError("active catalog phase must remain P1")
    if any(entry.live_run_allowed for entry in catalog.entries):
        raise E8AuthorizationError("catalog live_run_allowed must remain false")


def _validate_configuration() -> None:
    if not (
        E8_SERVER_BLOCK_TRUTH_CASE.check_id is EnvironmentValidationId.E8
        and E8_SERVER_BLOCK_TRUTH_CASE.name == "server_side_block_truth"
        and E8_SERVER_BLOCK_TRUTH_CASE.requires_server_truth
        and E8_SERVER_BLOCK_TRUTH_CASE.calibration_only
        and E8_STIMULUS_BLOCK == "dirt"
        and E8_PROBE_WORLD_CELLS == ((0, 4, 1), (1, 4, 1), (-1, 4, 1))
        and E8_PROBE_GRID_CELLS == ((0, 0, 1), (1, 0, 1), (-1, 0, 1))
        and E8_SPAWN_WORLD == (0, 4, 0)
        and E8_DURATION_TICKS == 1
        and (E8_INITIAL_YAW, E8_INITIAL_PITCH) == (0.0, 60.0)
        and E8_EXPECTED_BEFORE_BLOCKS[(0, 4, 1)] == "air"
        and E8_EXPECTED_AFTER_BLOCKS[(0, 4, 1)] == "dirt"
    ):
        raise E8AuthorizationError("frozen E8 calibration differs")
    task = build_e8_compatibility_task("p1-e8-preflight")
    if dict(task.initial_inventories[E8_AGENT_ID]) != E8_COMPATIBILITY_INVENTORY:
        raise E8AuthorizationError("E8 compatibility inventory differs")
    if task.spawn_positions[E8_AGENT_ID] != E8_SPAWN_WORLD:
        raise E8AuthorizationError("E8 flat-ground spawn differs")
    if MineRLE8BlockTruthAdapter(episode_id="p1-e8-preflight")._backend is not None:
        raise E8AuthorizationError("E8 adapter construction created a backend")


def check_e8_live_runner() -> dict[str, Any]:
    _validate_catalog_policy()
    _validate_configuration()
    return {
        "authorized_live_run_required": AUTHORIZED_LIVE_E8_RUN_VALUE,
        "calibration_only": True,
        "check_id": "E8",
        "execution_mode_required": EXECUTION_MODE_AUTHORIZED_LIVE_E8,
        "expected_after": {
            ",".join(str(axis) for axis in cell): block
            for cell, block in E8_EXPECTED_AFTER_BLOCKS.items()
        },
        "expected_before": {
            ",".join(str(axis) for axis in cell): block
            for cell, block in E8_EXPECTED_BEFORE_BLOCKS.items()
        },
        "gradle_authorized": False,
        "integration_verified": False,
        "name": "server_side_block_truth",
        "probe_count": len(E8_PROBE_WORLD_CELLS),
        "probe_grid_cells": [list(cell) for cell in E8_PROBE_GRID_CELLS],
        "probe_world_cells": [list(cell) for cell in E8_PROBE_WORLD_CELLS],
        "production_backend_constructed": False,
        "real_execution_performed": False,
        "status": "ok",
        "stimulus": {
            "action_type": "place_block",
            "duration_ticks": E8_DURATION_TICKS,
            "target": E8_STIMULUS_BLOCK,
        },
        "truth_missing_required": 0,
        "verification_level": "unit_verified",
    }


def _validate_output_dir(output_dir: Path) -> Path:
    if not isinstance(output_dir, Path) or not output_dir.is_absolute():
        raise E8AuthorizationError("output_dir must be an absolute pathlib.Path")
    resolved = output_dir.resolve()
    if resolved.exists() or resolved.is_symlink():
        raise E8AuthorizationError(f"output_dir must not exist: {resolved}")
    try:
        resolved.relative_to(FORMAL_E8_RUNS_ROOT)
    except ValueError as exc:
        raise E8AuthorizationError(f"output_dir must be under {FORMAL_E8_RUNS_ROOT}") from exc
    if resolved == FORMAL_E8_RUNS_ROOT or resolved.parent != FORMAL_E8_RUNS_ROOT:
        raise E8AuthorizationError("output_dir must be a unique direct child")
    return resolved


def _e8_env_factory(task: TaskInstance) -> Any:
    from obsidianlink.env.portal_spec import PortalA0EnvSpec

    initial_inventory = tuple(
        {"type": item, "quantity": quantity}
        for item, quantity in task.initial_inventories[E8_AGENT_ID].items()
        if quantity > 0
    )
    specification = PortalA0EnvSpec(
        max_episode_steps=task.limits["max_environment_steps"],
        max_game_time_seconds=task.limits["max_game_time_seconds"],
        initial_inventory=initial_inventory,
        initial_position=task.spawn_positions[E8_AGENT_ID],
        include_agent_start_placement=True,
        grid_at_spawn=True,
        initial_yaw=E8_INITIAL_YAW,
        initial_pitch=E8_INITIAL_PITCH,
    )
    return specification.make()


def _production_backend_cls() -> type:
    from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
    return MineRLEnvironmentBackend


def _production_backend_kwargs() -> dict[str, Any]:
    return {"max_reset_attempts": 1, "env_factory": _e8_env_factory}


def preflight_authorized_e8(*, execution_mode: str, authorized_live_run: str, output_dir: Path | None = None, allow_gradle: bool = False) -> dict[str, Any]:
    assert_e8_live_authorized(execution_mode=execution_mode, authorized_live_run=authorized_live_run, allow_gradle=allow_gradle)
    _validate_catalog_policy()
    _validate_configuration()
    payload = check_e8_live_runner()
    payload.update({"execution_mode": execution_mode, "authorized_live_run": authorized_live_run, "requires_server_truth": True})
    if output_dir is not None:
        payload["output_dir"] = str(_validate_output_dir(output_dir))
    return payload


def _write_evidence(record: E8MineRLRunRecord, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "e8_block_truth.json"
    path.write_text(json.dumps(record.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "authorization.json").write_text(
        json.dumps(
            {
                "authorized_live_run": AUTHORIZED_LIVE_E8_RUN_VALUE,
                "catalog_live_run_allowed_remains_false": True,
                "execution_mode": EXECUTION_MODE_AUTHORIZED_LIVE_E8,
                "gradle_authorized": False,
                "integration_verified": False,
                "model_api_authorized": False,
                "planned_tested_stimulus_count": 1,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return path


def run_authorized_e8_minerl(*, execution_mode: str, authorized_live_run: str, output_dir: Path, episode_id: str = "p1-e8-block-truth", allow_gradle: bool = False, preflight_only: bool = False) -> E8MineRLRunRecord | dict[str, Any]:
    preflight = preflight_authorized_e8(execution_mode=execution_mode, authorized_live_run=authorized_live_run, output_dir=output_dir, allow_gradle=allow_gradle)
    if preflight_only:
        return preflight
    global _PROCESS_LIVE_RUN_STARTED
    with _PROCESS_LIVE_RUN_LOCK:
        if _PROCESS_LIVE_RUN_STARTED:
            raise E8AuthorizationError("authorized E8 allows one real run per process")
        _PROCESS_LIVE_RUN_STARTED = True
    log_snapshot = _snapshot_runtime_logs()
    backend_cls = _production_backend_cls()
    factory = MineRLE8BlockTruthAdapter.lifecycle_factory(
        episode_id=episode_id,
        backend_cls=backend_cls,
        backend_kwargs=_production_backend_kwargs(),
    )
    holder: dict[str, MineRLE8BlockTruthAdapter | None] = {"adapter": None}

    def capture() -> MineRLE8BlockTruthAdapter:
        adapter = factory()
        holder["adapter"] = adapter
        return adapter

    lifecycle = EnvironmentValidationRunner().run(
        E8_SERVER_BLOCK_TRUTH_CASE,
        capture,
        episode_id=episode_id,
        requested_duration_ticks=E8_DURATION_TICKS,
    )
    adapter = holder["adapter"]
    cleanup = adapter.cleanup_status() if adapter is not None else E0CleanupStatus(lifecycle.closed, lifecycle.closed, lifecycle.closed, lifecycle.closed)
    detected_cause, log_evidence = _collect_runtime_log_evidence(log_snapshot)
    failure_cause = (
        detected_cause if lifecycle.outcome == "reset_failed" else "unknown"
    )
    record = E8MineRLRunRecord(
        execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E8,
        authorized_live_run=AUTHORIZED_LIVE_E8_RUN_VALUE,
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
    parser = argparse.ArgumentParser(prog="obsidianlink.env.integration.e8_run", description="Authorized P1 E8 server-side block-truth entrypoint; --check is offline-safe.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execution-mode")
    parser.add_argument("--authorized-live-run")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episode-id", default="p1-e8-block-truth")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        if any(value is not None for value in (args.execution_mode, args.authorized_live_run, args.output_dir)) or args.preflight_only:
            raise E8AuthorizationError("--check cannot be combined with live arguments")
        payload: Mapping[str, Any] = check_e8_live_runner()
    else:
        if args.output_dir is None:
            raise E8AuthorizationError("--output-dir is required for E8 live/preflight")
        result = run_authorized_e8_minerl(execution_mode=args.execution_mode, authorized_live_run=args.authorized_live_run, output_dir=args.output_dir, episode_id=args.episode_id, preflight_only=args.preflight_only)
        payload = result if isinstance(result, Mapping) else result.as_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

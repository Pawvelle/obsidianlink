"""Explicitly authorized P1 E12 real-MineRL dimension-transition entrypoint.

Imports and ``--check`` are offline-safe. The canonical DrawingDecorator
allowlist still rejects portal DrawBlocks. This module never runs Gradle,
models, solvers, or E11 ignition. Real Overworld → Nether entry remains a
separately authorized live run after the E12 portal fixture JAR is built.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from obsidianlink.core.task_catalog import load_task_catalog
from obsidianlink.core.types import TaskInstance
from obsidianlink.env.integration.e0_adapter import BACKEND_IDENTITY
from obsidianlink.env.integration.e0_cleanup import E0CleanupStatus
from obsidianlink.env.integration.e12_adapter import MineRLE12DimensionTransitionAdapter
from obsidianlink.env.integration.e12_config import (
    E12_AGENT_ID,
    E12_CALIBRATION,
    E12_COMPATIBILITY_INVENTORY,
    E12_CONTROL_WORLD_CELLS,
    E12_DURATION_TICKS,
    E12_EXPECTED_AFTER_DIMENSION,
    E12_EXPECTED_BEFORE_DIMENSION,
    E12_FRAME_BLOCKS,
    E12_INITIAL_DRAW_BLOCKS,
    E12_INITIAL_PITCH,
    E12_INITIAL_YAW,
    E12_INTERIOR_CELLS,
    E12_OBSERVATION_WINDOW_TICKS,
    E12_PROBE_GRID_CELLS,
    E12_PROBE_WORLD_CELLS,
    E12_SPAWN_WORLD,
    E12_STIMULUS_ACTION_TYPE,
    build_e12_compatibility_task,
    e12_initial_blocks,
)
from obsidianlink.env.validation import E12_DIMENSION_TRANSITION_CASE, EnvironmentValidationRunner
from obsidianlink.env.validation.contract import EnvironmentValidationId
from obsidianlink.env.validation.result import EnvironmentValidationResult


ROOT = Path(__file__).resolve().parents[3]
FORMAL_E12_RUNS_ROOT = (ROOT / "runs" / "p1_e12_dimension_transition").resolve()
RUNTIME_LOGS_ROOT = (ROOT / "logs").resolve()
EXECUTION_MODE_AUTHORIZED_LIVE_E12 = "authorized_live_e12"
AUTHORIZED_LIVE_E12_RUN_VALUE = "e12_dimension_transition"
NEEDS_E12_RUNTIME_PORTAL_FIXTURE_AUTHORIZATION = True

_PROCESS_LIVE_RUN_STARTED = False
_PROCESS_LIVE_RUN_LOCK = threading.Lock()


class E12AuthorizationError(ValueError):
    """Raised when the exact E12 live gate or preflight is invalid."""


@dataclass(frozen=True)
class E12LogEvidence:
    kind: str
    path: str
    sha256: str
    summary: str

    def __post_init__(self) -> None:
        if self.kind not in {"minecraft", "jvm_crash", "process_watcher"}:
            raise ValueError("unknown E12 log evidence kind")
        if not Path(self.path).is_absolute():
            raise ValueError("E12 log evidence path must be absolute")
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError("E12 log evidence sha256 must be lowercase hex")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("E12 log evidence summary must be non-empty")

    def as_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class E12MineRLRunRecord:
    execution_mode: str
    authorized_live_run: str
    backend_identity: str
    opened: bool
    authorization_accepted: bool
    real_execution_performed: bool
    lifecycle: EnvironmentValidationResult
    cleanup: E0CleanupStatus
    failure_cause: str
    log_evidence: tuple[E12LogEvidence, ...]

    def __post_init__(self) -> None:
        if self.execution_mode != EXECUTION_MODE_AUTHORIZED_LIVE_E12:
            raise ValueError("execution_mode must be authorized_live_e12")
        if self.authorized_live_run != AUTHORIZED_LIVE_E12_RUN_VALUE:
            raise ValueError("authorized_live_run must be e12_dimension_transition")
        if not isinstance(self.lifecycle, EnvironmentValidationResult) or self.lifecycle.check_id is not EnvironmentValidationId.E12:
            raise ValueError("lifecycle must be an E12 validation result")
        if not isinstance(self.cleanup, E0CleanupStatus):
            raise ValueError("cleanup must be E0CleanupStatus")
        if any(type(value) is not bool for value in (self.opened, self.authorization_accepted, self.real_execution_performed)):
            raise ValueError("record flags must be bool")
        if self.failure_cause not in {"unknown", "minecraft_native_crash"}:
            raise ValueError("unknown E12 failure cause")
        if not isinstance(self.log_evidence, tuple) or not all(
            isinstance(value, E12LogEvidence) for value in self.log_evidence
        ):
            raise ValueError("log_evidence must be a tuple of E12LogEvidence")

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
            }
        )
        return payload


def reset_authorized_e12_process_guards_for_tests() -> None:
    global _PROCESS_LIVE_RUN_STARTED
    with _PROCESS_LIVE_RUN_LOCK:
        _PROCESS_LIVE_RUN_STARTED = False


def assert_e12_live_authorized(*, execution_mode: object, authorized_live_run: object, allow_gradle: object = False) -> None:
    if execution_mode != EXECUTION_MODE_AUTHORIZED_LIVE_E12:
        raise E12AuthorizationError("execution_mode must be exactly authorized_live_e12")
    if authorized_live_run != AUTHORIZED_LIVE_E12_RUN_VALUE:
        raise E12AuthorizationError("authorized_live_run must be exactly e12_dimension_transition")
    if allow_gradle is not False:
        raise E12AuthorizationError("Gradle is not authorized for E12")


def _validate_catalog_policy() -> None:
    catalog = load_task_catalog(ROOT / "benchmark/catalog/tasks.json")
    if catalog.active_phase != "P1-REAL-MINERL-ENVIRONMENT-VALIDATION":
        raise E12AuthorizationError("active catalog phase must remain P1")
    if any(entry.live_run_allowed for entry in catalog.entries):
        raise E12AuthorizationError("catalog live_run_allowed must remain false")


def _validate_configuration() -> None:
    if not (
        E12_DIMENSION_TRANSITION_CASE.check_id is EnvironmentValidationId.E12
        and E12_DIMENSION_TRANSITION_CASE.name == "dimension_transition"
        and E12_DIMENSION_TRANSITION_CASE.requires_server_truth
        and E12_DIMENSION_TRANSITION_CASE.calibration_only
        and E12_STIMULUS_ACTION_TYPE == "move"
        and E12_SPAWN_WORLD == (0, 4, 0)
        and len(E12_FRAME_BLOCKS) == 14
        and len(E12_INTERIOR_CELLS) == 6
        and E12_CONTROL_WORLD_CELLS == ((0, 8, 1), (0, 4, 3))
        and E12_DURATION_TICKS == 8
        and E12_OBSERVATION_WINDOW_TICKS == 100
        and (E12_INITIAL_YAW, E12_INITIAL_PITCH) == (0.0, 0.0)
        and E12_CALIBRATION.initial_draw_blocks == E12_INITIAL_DRAW_BLOCKS
        and E12_EXPECTED_BEFORE_DIMENSION == "minecraft:overworld"
        and E12_EXPECTED_AFTER_DIMENSION == "minecraft:the_nether"
    ):
        raise E12AuthorizationError("frozen E12 calibration differs")
    task = build_e12_compatibility_task("p1-e12-preflight")
    if dict(task.initial_inventories[E12_AGENT_ID]) != E12_COMPATIBILITY_INVENTORY:
        raise E12AuthorizationError("E12 compatibility inventory differs")
    if task.spawn_positions[E12_AGENT_ID] != E12_SPAWN_WORLD:
        raise E12AuthorizationError("E12 flat-ground spawn differs")
    if task.scenario_parameters.get("portal_preplaced") is not True:
        raise E12AuthorizationError("E12 must pre-place an active portal fixture")
    if task.scenario_parameters.get("fire_preplaced") is not False:
        raise E12AuthorizationError("E12 must not pre-place fire")
    if task.scenario_parameters.get("obsidian_frame_preplaced") is not True:
        raise E12AuthorizationError("E12 must request the obsidian frame fixture")
    from obsidianlink.env.portal_spec import PortalA0EnvSpec, parse_mission_draw_blocks

    xml = PortalA0EnvSpec(
        max_episode_steps=130,
        max_game_time_seconds=60,
        initial_inventory=({"type": "dirt", "quantity": 1},),
        initial_position=E12_SPAWN_WORLD,
        include_agent_start_placement=True,
        grid_at_spawn=True,
        initial_yaw=E12_INITIAL_YAW,
        initial_pitch=E12_INITIAL_PITCH,
        initial_blocks=e12_initial_blocks(),
        allow_active_portal_fixture=True,
    ).to_xml()
    draw_blocks = parse_mission_draw_blocks(xml)
    if any(block not in {"obsidian", "portal"} for _, _, _, block in draw_blocks):
        raise E12AuthorizationError("E12 Mission XML may pre-place only obsidian and portal")
    if any(block in {"nether_portal", "fire"} for _, _, _, block in draw_blocks):
        raise E12AuthorizationError("E12 Mission XML must not use nether_portal or fire names")
    if sum(block == "obsidian" for _, _, _, block in draw_blocks) != 14:
        raise E12AuthorizationError("E12 Mission XML must contain the complete 14-cell frame")
    if sum(block == "portal" for _, _, _, block in draw_blocks) != 6:
        raise E12AuthorizationError("E12 Mission XML must contain the 6-cell active portal interior")
    if MineRLE12DimensionTransitionAdapter(episode_id="p1-e12-preflight")._backend is not None:
        raise E12AuthorizationError("E12 adapter construction created a backend")


def check_e12_live_runner() -> dict[str, Any]:
    _validate_catalog_policy()
    _validate_configuration()
    return {
        "authorized_live_run_required": AUTHORIZED_LIVE_E12_RUN_VALUE,
        "calibration_only": True,
        "check_id": "E12",
        "execution_mode_required": EXECUTION_MODE_AUTHORIZED_LIVE_E12,
        "frame_cell_count": len(E12_FRAME_BLOCKS),
        "gradle_authorized": False,
        "integration_verified": False,
        "interior_cell_count": len(E12_INTERIOR_CELLS),
        "controlled_initial_geometry": True,
        "name": "dimension_transition",
        "observation_window_ticks": E12_OBSERVATION_WINDOW_TICKS,
        "obsidian_frame_preplaced": True,
        "portal_preplaced": True,
        "fire_preplaced": False,
        "prebuilt_active_portal_is_calibration_fixture": True,
        "probe_count": len(E12_PROBE_WORLD_CELLS),
        "probe_grid_cells": [list(cell) for cell in E12_PROBE_GRID_CELLS],
        "probe_world_cells": [list(cell) for cell in E12_PROBE_WORLD_CELLS],
        "production_backend_constructed": False,
        "real_execution_performed": False,
        "status": "ok",
        "stimulus": {
            "action_type": E12_STIMULUS_ACTION_TYPE,
            "duration_ticks": E12_DURATION_TICKS,
            "forward": 1.0,
            "jump": False,
            "sprint": False,
            "strafe": 0.0,
        },
        "expected_before_dimension": E12_EXPECTED_BEFORE_DIMENSION,
        "expected_after_dimension": E12_EXPECTED_AFTER_DIMENSION,
        "truth_missing_required": 0,
        "verification_level": "unit_verified",
        "runtime_applies_active_portal_draw_blocks": (
            not NEEDS_E12_RUNTIME_PORTAL_FIXTURE_AUTHORIZATION
        ),
        "needs_e12_runtime_portal_fixture_authorization": NEEDS_E12_RUNTIME_PORTAL_FIXTURE_AUTHORIZATION,
        "runtime_blocker": (
            "E12 live DrawBlock portal fixture is not in the canonical JAR"
            if NEEDS_E12_RUNTIME_PORTAL_FIXTURE_AUTHORIZATION
            else None
        ),
    }


def _validate_output_dir(output_dir: Path) -> Path:
    if not isinstance(output_dir, Path) or not output_dir.is_absolute():
        raise E12AuthorizationError("output_dir must be an absolute pathlib.Path")
    resolved = output_dir.resolve()
    if resolved.exists() or resolved.is_symlink():
        raise E12AuthorizationError(f"output_dir must not exist: {resolved}")
    try:
        resolved.relative_to(FORMAL_E12_RUNS_ROOT)
    except ValueError as exc:
        raise E12AuthorizationError(f"output_dir must be under {FORMAL_E12_RUNS_ROOT}") from exc
    if resolved == FORMAL_E12_RUNS_ROOT or resolved.parent != FORMAL_E12_RUNS_ROOT:
        raise E12AuthorizationError("output_dir must be a unique direct child")
    return resolved


def _e12_env_factory(task: TaskInstance) -> Any:
    from obsidianlink.env.portal_spec import PortalA0EnvSpec

    initial_inventory = tuple(
        {"type": item, "quantity": quantity}
        for item, quantity in task.initial_inventories[E12_AGENT_ID].items()
        if quantity > 0
    )
    specification = PortalA0EnvSpec(
        max_episode_steps=task.limits["max_environment_steps"],
        max_game_time_seconds=task.limits["max_game_time_seconds"],
        initial_inventory=initial_inventory,
        initial_position=task.spawn_positions[E12_AGENT_ID],
        include_agent_start_placement=True,
        grid_at_spawn=True,
        initial_yaw=E12_INITIAL_YAW,
        initial_pitch=E12_INITIAL_PITCH,
        initial_blocks=e12_initial_blocks(),
        allow_active_portal_fixture=True,
    )
    return specification.make()


def _production_backend_cls() -> type:
    from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
    return MineRLEnvironmentBackend


def _production_backend_kwargs() -> dict[str, Any]:
    return {"max_reset_attempts": 1, "env_factory": _e12_env_factory}


def preflight_authorized_e12(*, execution_mode: str, authorized_live_run: str, output_dir: Path | None = None, allow_gradle: bool = False) -> dict[str, Any]:
    assert_e12_live_authorized(execution_mode=execution_mode, authorized_live_run=authorized_live_run, allow_gradle=allow_gradle)
    _validate_catalog_policy()
    _validate_configuration()
    payload = check_e12_live_runner()
    payload.update({"execution_mode": execution_mode, "authorized_live_run": authorized_live_run, "requires_server_truth": True})
    if output_dir is not None:
        payload["output_dir"] = str(_validate_output_dir(output_dir))
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


def _make_log_evidence(path: Path, kind: str) -> E12LogEvidence:
    data = path.read_bytes()
    text = data.decode("utf-8", errors="replace")
    return E12LogEvidence(
        kind=kind,
        path=str(path.resolve()),
        sha256=hashlib.sha256(data).hexdigest(),
        summary=_log_summary(text, kind),
    )


def _collect_runtime_log_evidence(
    before: Mapping[str, tuple[int, int]],
) -> tuple[str, tuple[E12LogEvidence, ...]]:
    changed: list[Path] = []
    for path in _runtime_log_paths():
        state = (path.stat().st_size, path.stat().st_mtime_ns)
        if before.get(str(path)) != state:
            changed.append(path)
    evidence: list[E12LogEvidence] = []
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
        tuple(evidence),
    )


def _write_evidence(record: E12MineRLRunRecord, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "result.json"
    path.write_text(
        json.dumps(record.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "control_world_cells": [list(cell) for cell in E12_CONTROL_WORLD_CELLS],
                "duration_ticks": E12_DURATION_TICKS,
                "expected_after_dimension": E12_EXPECTED_AFTER_DIMENSION,
                "expected_before_dimension": E12_EXPECTED_BEFORE_DIMENSION,
                "frame_world_cells": [list(cell) for cell in E12_FRAME_BLOCKS],
                "interior_world_cells": [list(cell) for cell in E12_INTERIOR_CELLS],
                "controlled_initial_geometry": True,
                "obsidian_frame_preplaced": True,
                "portal_preplaced": True,
                "fire_preplaced": False,
                "prebuilt_active_portal_is_calibration_fixture": True,
                "observation_window_ticks": E12_OBSERVATION_WINDOW_TICKS,
                "probe_grid_cells": [list(cell) for cell in E12_PROBE_GRID_CELLS],
                "probe_world_cells": [list(cell) for cell in E12_PROBE_WORLD_CELLS],
                "spawn_world": list(E12_SPAWN_WORLD),
                "stimulus_action_type": E12_STIMULUS_ACTION_TYPE,
                "needs_e12_runtime_portal_fixture_authorization": NEEDS_E12_RUNTIME_PORTAL_FIXTURE_AUTHORIZATION,
                "runtime_applies_active_portal_draw_blocks": (
                    not NEEDS_E12_RUNTIME_PORTAL_FIXTURE_AUTHORIZATION
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "authorization.json").write_text(
        json.dumps(
            {
                "authorized_live_run": AUTHORIZED_LIVE_E12_RUN_VALUE,
                "catalog_live_run_allowed_remains_false": True,
                "execution_mode": EXECUTION_MODE_AUTHORIZED_LIVE_E12,
                "gradle_authorized": False,
                "integration_verified": False,
                "model_api_authorized": False,
                "portal_preplaced": True,
                "prebuilt_active_portal_is_calibration_fixture": True,
                "agent_built_portal": False,
                "needs_e12_runtime_portal_fixture_authorization": NEEDS_E12_RUNTIME_PORTAL_FIXTURE_AUTHORIZATION,
                "runtime_applies_active_portal_draw_blocks": (
                    not NEEDS_E12_RUNTIME_PORTAL_FIXTURE_AUTHORIZATION
                ),
                "planned_tested_stimulus_count": 1,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def run_authorized_e12_minerl(*, execution_mode: str, authorized_live_run: str, output_dir: Path, episode_id: str = "p1-e12-dimension-transition", allow_gradle: bool = False, preflight_only: bool = False) -> E12MineRLRunRecord | dict[str, Any]:
    preflight = preflight_authorized_e12(execution_mode=execution_mode, authorized_live_run=authorized_live_run, output_dir=output_dir, allow_gradle=allow_gradle)
    if preflight_only:
        return preflight
    if NEEDS_E12_RUNTIME_PORTAL_FIXTURE_AUTHORIZATION:
        raise E12AuthorizationError("NEEDS_E12_RUNTIME_PORTAL_FIXTURE_AUTHORIZATION")
    global _PROCESS_LIVE_RUN_STARTED
    with _PROCESS_LIVE_RUN_LOCK:
        if _PROCESS_LIVE_RUN_STARTED:
            raise E12AuthorizationError("authorized E12 allows one real run per process")
        _PROCESS_LIVE_RUN_STARTED = True
    log_snapshot = _snapshot_runtime_logs()
    backend_cls = _production_backend_cls()
    factory = MineRLE12DimensionTransitionAdapter.lifecycle_factory(
        episode_id=episode_id,
        backend_cls=backend_cls,
        backend_kwargs=_production_backend_kwargs(),
    )
    holder: dict[str, MineRLE12DimensionTransitionAdapter | None] = {"adapter": None}

    def capture() -> MineRLE12DimensionTransitionAdapter:
        adapter = factory()
        holder["adapter"] = adapter
        return adapter

    lifecycle = EnvironmentValidationRunner().run(
        E12_DIMENSION_TRANSITION_CASE,
        capture,
        episode_id=episode_id,
        requested_duration_ticks=E12_DURATION_TICKS,
    )
    adapter = holder["adapter"]
    cleanup = adapter.cleanup_status() if adapter is not None else E0CleanupStatus(lifecycle.closed, lifecycle.closed, lifecycle.closed, lifecycle.closed)
    detected_cause, log_evidence = _collect_runtime_log_evidence(log_snapshot)
    failure_cause = (
        detected_cause if lifecycle.outcome == "reset_failed" else "unknown"
    )
    record = E12MineRLRunRecord(
        execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E12,
        authorized_live_run=AUTHORIZED_LIVE_E12_RUN_VALUE,
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
    parser = argparse.ArgumentParser(prog="obsidianlink.env.integration.e12_run", description="Authorized P1 E12 vanilla dimension-transition entrypoint; --check is offline-safe.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execution-mode")
    parser.add_argument("--authorized-live-run")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episode-id", default="p1-e12-dimension-transition")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        if any(value is not None for value in (args.execution_mode, args.authorized_live_run, args.output_dir)) or args.preflight_only:
            raise E12AuthorizationError("--check cannot be combined with live arguments")
        payload: Mapping[str, Any] = check_e12_live_runner()
    else:
        if args.output_dir is None:
            raise E12AuthorizationError("--output-dir is required for E12 live/preflight")
        result = run_authorized_e12_minerl(execution_mode=args.execution_mode, authorized_live_run=args.authorized_live_run, output_dir=args.output_dir, episode_id=args.episode_id, preflight_only=args.preflight_only)
        payload = result if isinstance(result, Mapping) else result.as_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

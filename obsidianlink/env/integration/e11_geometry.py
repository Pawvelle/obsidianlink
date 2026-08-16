"""Authorized P1 E11 geometry-only MineRL smoke; no flint_and_steel ignition.

Imports and ``--check`` are offline-safe. The live path creates, resets,
reads evaluator-only before truth, and closes. It never calls
``use_item(flint_and_steel)``, never uses the E11 activation evaluator,
and never sets ``integration_verified`` or ``portal_activation_ok``.
"""

from __future__ import annotations

import json
import hashlib
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from obsidianlink.env.integration.e0_adapter import BACKEND_IDENTITY
from obsidianlink.env.integration.e0_cleanup import E0CleanupStatus
from obsidianlink.env.integration.e11_adapter import MineRLE11PortalActivationAdapter
from obsidianlink.env.integration.e11_config import (
    E11_CONTROL_WORLD_CELLS,
    E11_EXPECTED_DIMENSION,
    E11_FRAME_BLOCKS,
    E11_IGNITION_TARGET_CELL,
    E11_INTERIOR_CELLS,
    E11_PROBE_WORLD_CELLS,
    E11_SPAWN_WORLD,
    e11_initial_blocks,
)
from obsidianlink.env.integration.e11_run import (
    E11AuthorizationError,
    E11LogEvidence,
    _collect_runtime_log_evidence,
    _production_backend_cls,
    _production_backend_kwargs,
    _snapshot_runtime_logs,
    _validate_catalog_policy,
    _validate_configuration,
    _validate_output_dir,
)
from obsidianlink.env.validation.truth import ServerTruthSnapshot, is_portal_block


EXECUTION_MODE_AUTHORIZED_LIVE_E11_GEOMETRY = "authorized_live_e11_geometry"
AUTHORIZED_LIVE_E11_GEOMETRY_RUN_VALUE = "e11_geometry_smoke"
E11_GEOMETRY_READY = "e11_geometry_ready"
GEOMETRY_NOT_READY = "geometry_not_ready"
DEPLOYED_JAR_NAME = "mcprec-6.13.jar"

_PROCESS_GEOMETRY_STARTED = False
_PROCESS_GEOMETRY_LOCK = threading.Lock()

_GEOMETRY_OUTCOMES = frozenset(
    {
        E11_GEOMETRY_READY,
        GEOMETRY_NOT_READY,
        "reset_failed",
        "cleanup_failed",
        "runtime_error",
        "truth_snapshot_missing",
    }
)


def reset_authorized_e11_geometry_process_guards_for_tests() -> None:
    global _PROCESS_GEOMETRY_STARTED
    with _PROCESS_GEOMETRY_LOCK:
        _PROCESS_GEOMETRY_STARTED = False


def assert_e11_geometry_authorized(
    *,
    execution_mode: object,
    authorized_live_run: object,
    allow_gradle: object = False,
) -> None:
    if execution_mode != EXECUTION_MODE_AUTHORIZED_LIVE_E11_GEOMETRY:
        raise E11AuthorizationError(
            "execution_mode must be exactly authorized_live_e11_geometry"
        )
    if authorized_live_run != AUTHORIZED_LIVE_E11_GEOMETRY_RUN_VALUE:
        raise E11AuthorizationError(
            "authorized_live_run must be exactly e11_geometry_smoke"
        )
    if allow_gradle is not False:
        raise E11AuthorizationError("Gradle is not authorized for E11 geometry smoke")
    from obsidianlink.env.integration import e11_run as e11_run_module

    if e11_run_module.NEEDS_E11_RUNTIME_GEOMETRY_AUTHORIZATION:
        raise E11AuthorizationError("NEEDS_E11_RUNTIME_GEOMETRY_AUTHORIZATION")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deployed_runtime_paths() -> dict[str, str]:
    from obsidianlink.env.integration.native_runtime import discover_minerl_path

    mcp = discover_minerl_path() / "MCP-Reborn"
    jar = mcp / "build" / "libs" / DEPLOYED_JAR_NAME
    launcher = mcp / "launchClient.sh"
    return {
        "jar_path": str(jar.resolve()),
        "jar_sha256": _sha256_file(jar) if jar.is_file() else "",
        "launcher_path": str(launcher.resolve()),
        "launcher_sha256": _sha256_file(launcher) if launcher.is_file() else "",
    }


def _runtime_identity() -> dict[str, str]:
    return deployed_runtime_paths()


def e11_mission_xml() -> str:
    from obsidianlink.env.portal_spec import PortalA0EnvSpec

    return PortalA0EnvSpec(
        max_episode_steps=12,
        max_game_time_seconds=30,
        initial_inventory=({"type": "flint_and_steel", "quantity": 1},),
        initial_position=E11_SPAWN_WORLD,
        include_agent_start_placement=True,
        grid_at_spawn=True,
        initial_yaw=0.0,
        initial_pitch=60.0,
        initial_blocks=e11_initial_blocks(),
        allow_obsidian_frame_fixture=True,
    ).to_xml()


def inspect_e11_geometry(snapshot: ServerTruthSnapshot | None) -> dict[str, Any]:
    """Return a geometry-only verdict. Never reports portal activation."""

    empty_controls = {
        str(list(cell)): {"block": None, "expected": "air"}
        for cell in E11_CONTROL_WORLD_CELLS
    }
    if snapshot is None:
        return {
            "outcome": "truth_snapshot_missing",
            "ready": False,
            "dimension": None,
            "truth_missing_count": None,
            "frame_valid_before": False,
            "observed_frame_block_count": 0,
            "expected_frame_cells": 14,
            "interior_air_count": 0,
            "expected_interior_cells": 6,
            "before_portal_block_count": 0,
            "fire_block_count": 0,
            "ignition_cell": list(E11_IGNITION_TARGET_CELL),
            "ignition_block": None,
            "controls": empty_controls,
            "control_cells_expected": False,
            "detail": "before server truth snapshot is missing",
        }
    observed_frame: list[tuple[int, int, int]] = []
    interior_air = 0
    portal_count = 0
    fire_count = 0
    for cell in E11_FRAME_BLOCKS:
        block = snapshot.block_at(cell)
        if block == "obsidian":
            observed_frame.append(cell)
        if is_portal_block(block):
            portal_count += 1
        if block == "fire":
            fire_count += 1
    for cell in E11_INTERIOR_CELLS:
        block = snapshot.block_at(cell)
        if block == "air":
            interior_air += 1
        if is_portal_block(block):
            portal_count += 1
        if block == "fire":
            fire_count += 1
    controls: dict[str, dict[str, Any]] = {}
    control_ok = True
    for cell in E11_CONTROL_WORLD_CELLS:
        block = snapshot.block_at(cell)
        controls[str(list(cell))] = {"block": block, "expected": "air"}
        control_ok = control_ok and block == "air"
        if is_portal_block(block):
            portal_count += 1
        if block == "fire":
            fire_count += 1
    ignition_block = snapshot.block_at(E11_IGNITION_TARGET_CELL)
    frame_valid = len(observed_frame) == 14
    ignition_ok = ignition_block == "air"
    ready = (
        snapshot.dimension == E11_EXPECTED_DIMENSION
        and snapshot.truth_missing_count == 0
        and frame_valid
        and interior_air == 6
        and portal_count == 0
        and fire_count == 0
        and ignition_ok
        and control_ok
    )
    return {
        "outcome": E11_GEOMETRY_READY if ready else GEOMETRY_NOT_READY,
        "ready": ready,
        "dimension": snapshot.dimension,
        "truth_missing_count": snapshot.truth_missing_count,
        "frame_valid_before": frame_valid,
        "observed_frame_block_count": len(observed_frame),
        "expected_frame_cells": 14,
        "interior_air_count": interior_air,
        "expected_interior_cells": 6,
        "before_portal_block_count": portal_count,
        "fire_block_count": fire_count,
        "ignition_cell": list(E11_IGNITION_TARGET_CELL),
        "ignition_block": ignition_block,
        "controls": controls,
        "control_cells_expected": control_ok,
        "detail": (
            "reset geometry matches the frozen 14-cell obsidian frame with air interior"
            if ready
            else "reset geometry does not match required obsidian frame / air interior"
        ),
    }


@dataclass(frozen=True)
class E11GeometrySmokeRecord:
    execution_mode: str
    authorized_live_run: str
    backend_identity: str
    opened: bool
    authorization_accepted: bool
    real_execution_performed: bool
    reset_succeeded: bool
    closed: bool
    outcome: str
    success: bool
    integration_verified: bool
    tested_action_count: int
    observation_wait_count: int
    minerl_launched: bool
    minecraft_launched: bool
    reset_attempts: int
    environment_launch_count: int
    geometry: dict[str, Any]
    cleanup: E0CleanupStatus
    failure_cause: str
    log_evidence: tuple[E11LogEvidence, ...]
    runtime_jar_sha256: str
    traceback_text: str | None

    def __post_init__(self) -> None:
        if self.execution_mode != EXECUTION_MODE_AUTHORIZED_LIVE_E11_GEOMETRY:
            raise ValueError("execution_mode must be authorized_live_e11_geometry")
        if self.authorized_live_run != AUTHORIZED_LIVE_E11_GEOMETRY_RUN_VALUE:
            raise ValueError("authorized_live_run must be e11_geometry_smoke")
        if self.outcome not in _GEOMETRY_OUTCOMES:
            raise ValueError("unknown E11 geometry outcome")
        if self.outcome == E11_GEOMETRY_READY and not self.success:
            raise ValueError("e11_geometry_ready requires success")
        if self.success and self.outcome != E11_GEOMETRY_READY:
            raise ValueError("geometry success must use e11_geometry_ready")
        if self.integration_verified:
            raise ValueError("geometry smoke must not claim integration_verified")
        if self.tested_action_count != 0 or self.observation_wait_count != 0:
            raise ValueError("geometry smoke must not execute tested stimulus")
        if self.outcome == "portal_activation_ok":
            raise ValueError("geometry smoke must not report portal activation")
        if type(self.success) is not bool:
            raise ValueError("success must be bool")

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "authorization_accepted": self.authorization_accepted,
            "authorized_live_run": self.authorized_live_run,
            "backend_identity": self.backend_identity,
            "cleanup": self.cleanup.as_dict(),
            "closed": self.closed,
            "environment_launch_count": self.environment_launch_count,
            "execution_mode": self.execution_mode,
            "failure_cause": self.failure_cause,
            "geometry": dict(self.geometry),
            "integration_verified": False,
            "log_evidence": [item.as_dict() for item in self.log_evidence],
            "minecraft_launched": self.minecraft_launched,
            "minerl_launched": self.minerl_launched,
            "observation_wait_count": self.observation_wait_count,
            "opened": self.opened,
            "outcome": self.outcome,
            "real_execution_performed": self.real_execution_performed,
            "reset_attempts": self.reset_attempts,
            "reset_succeeded": self.reset_succeeded,
            "runtime_jar_sha256": self.runtime_jar_sha256,
            "success": self.success,
            "tested_action_count": 0,
            "traceback": self.traceback_text,
            "verification_level": "unit_verified",
        }
        payload.update(
            {
                "frame_valid_before": self.geometry.get("frame_valid_before"),
                "observed_frame_block_count": self.geometry.get("observed_frame_block_count"),
                "interior_air_count": self.geometry.get("interior_air_count"),
                "before_portal_block_count": self.geometry.get("before_portal_block_count"),
                "fire_block_count": self.geometry.get("fire_block_count"),
                "ignition_block": self.geometry.get("ignition_block"),
                "control_cells_expected": self.geometry.get("control_cells_expected"),
                "truth_missing_count": self.geometry.get("truth_missing_count"),
                "dimension": self.geometry.get("dimension"),
            }
        )
        return payload


def check_e11_geometry_runner() -> dict[str, Any]:
    from obsidianlink.env.integration import e11_run as e11_run_module

    _validate_catalog_policy()
    _validate_configuration()
    applies = not e11_run_module.NEEDS_E11_RUNTIME_GEOMETRY_AUTHORIZATION
    return {
        "authorized_live_run_required": AUTHORIZED_LIVE_E11_GEOMETRY_RUN_VALUE,
        "calibration_only": True,
        "check_id": "E11_GEOMETRY",
        "execution_mode_required": EXECUTION_MODE_AUTHORIZED_LIVE_E11_GEOMETRY,
        "expected_frame_cells": 14,
        "expected_interior_cells": 6,
        "gradle_authorized": False,
        "ignition_target_world_cell": list(E11_IGNITION_TARGET_CELL),
        "integration_verified": False,
        "name": "e11_reset_geometry_smoke",
        "needs_e11_runtime_geometry_authorization": (
            e11_run_module.NEEDS_E11_RUNTIME_GEOMETRY_AUTHORIZATION
        ),
        "planned_tested_stimulus_count": 0,
        "portal_preplaced": False,
        "fire_preplaced": False,
        "prebuilt_frame_is_calibration_fixture": True,
        "production_backend_constructed": False,
        "real_execution_performed": False,
        "runtime_applies_obsidian_draw_blocks": applies,
        "spawn_world": list(E11_SPAWN_WORLD),
        "status": "ok",
        "success_outcome": E11_GEOMETRY_READY,
        "verification_level": "unit_verified",
    }


def preflight_authorized_e11_geometry(
    *,
    execution_mode: str,
    authorized_live_run: str,
    output_dir: Path | None = None,
    allow_gradle: bool = False,
) -> dict[str, Any]:
    assert_e11_geometry_authorized(
        execution_mode=execution_mode,
        authorized_live_run=authorized_live_run,
        allow_gradle=allow_gradle,
    )
    _validate_catalog_policy()
    _validate_configuration()
    payload = check_e11_geometry_runner()
    payload.update(
        {
            "execution_mode": execution_mode,
            "authorized_live_run": authorized_live_run,
            "requires_server_truth": True,
        }
    )
    if output_dir is not None:
        payload["output_dir"] = str(_validate_output_dir(output_dir))
    return payload


def _write_geometry_evidence(
    record: E11GeometrySmokeRecord,
    output_dir: Path,
    *,
    snapshot: Mapping[str, Any] | None,
    mission_xml: str,
    reset_audit: Mapping[str, int],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    path = output_dir / "result.json"
    path.write_text(
        json.dumps(record.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "control_world_cells": [list(cell) for cell in E11_CONTROL_WORLD_CELLS],
                "frame_world_cells": [list(cell) for cell in E11_FRAME_BLOCKS],
                "ignition_target_world_cell": list(E11_IGNITION_TARGET_CELL),
                "interior_world_cells": [list(cell) for cell in E11_INTERIOR_CELLS],
                "planned_tested_stimulus_count": 0,
                "portal_preplaced": False,
                "fire_preplaced": False,
                "prebuilt_frame_is_calibration_fixture": True,
                "probe_world_cells": [list(cell) for cell in E11_PROBE_WORLD_CELLS],
                "spawn_world": list(E11_SPAWN_WORLD),
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
                "authorized_live_run": AUTHORIZED_LIVE_E11_GEOMETRY_RUN_VALUE,
                "catalog_live_run_allowed_remains_false": True,
                "execution_mode": EXECUTION_MODE_AUTHORIZED_LIVE_E11_GEOMETRY,
                "gradle_authorized": False,
                "integration_verified": False,
                "model_api_authorized": False,
                "planned_tested_stimulus_count": 0,
                "portal_activation_authorized": False,
                "runtime_applies_obsidian_draw_blocks": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "mission.xml").write_text(mission_xml, encoding="utf-8")
    (output_dir / "reset_audit.json").write_text(
        json.dumps(dict(reset_audit), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if snapshot is not None:
        (output_dir / "before_server_truth.json").write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if record.traceback_text:
        (output_dir / "traceback.txt").write_text(record.traceback_text, encoding="utf-8")
    return path


def run_authorized_e11_geometry_smoke(
    *,
    execution_mode: str,
    authorized_live_run: str,
    output_dir: Path,
    episode_id: str = "p1-e11-geometry-smoke",
    allow_gradle: bool = False,
    preflight_only: bool = False,
) -> E11GeometrySmokeRecord | dict[str, Any]:
    preflight = preflight_authorized_e11_geometry(
        execution_mode=execution_mode,
        authorized_live_run=authorized_live_run,
        output_dir=output_dir,
        allow_gradle=allow_gradle,
    )
    if preflight_only:
        return preflight
    global _PROCESS_GEOMETRY_STARTED
    with _PROCESS_GEOMETRY_LOCK:
        if _PROCESS_GEOMETRY_STARTED:
            raise E11AuthorizationError("authorized E11 geometry allows one real run per process")
        _PROCESS_GEOMETRY_STARTED = True
    log_snapshot = _snapshot_runtime_logs()
    backend_cls = _production_backend_cls()
    adapter = MineRLE11PortalActivationAdapter(
        episode_id=episode_id,
        backend_cls=backend_cls,
        backend_kwargs=_production_backend_kwargs(),
    )
    mission_xml = e11_mission_xml()
    runtime = _runtime_identity()
    snapshot_payload: dict[str, Any] | None = None
    typed_snapshot: ServerTruthSnapshot | None = None
    reset_succeeded = False
    outcome = "runtime_error"
    traceback_text: str | None = None
    geometry = inspect_e11_geometry(None)
    try:
        adapter.reset()
        reset_succeeded = True
        typed_snapshot = adapter.server_truth_snapshot()
        snapshot_payload = None if typed_snapshot is None else typed_snapshot.as_dict()
        geometry = inspect_e11_geometry(typed_snapshot)
        outcome = geometry["outcome"]
    except Exception:
        traceback_text = traceback.format_exc()
        diagnostics = adapter.reset_failure_diagnostics()
        if diagnostics.get("traceback"):
            traceback_text = str(diagnostics["traceback"])
        outcome = "reset_failed" if not reset_succeeded else "runtime_error"
        if not reset_succeeded:
            geometry = inspect_e11_geometry(None)
    closed = False
    try:
        adapter.close()
        closed = True
    except Exception:
        if traceback_text is None:
            traceback_text = traceback.format_exc()
        closed = False
    cleanup = adapter.cleanup_status()
    if cleanup.has_explicit_failure() or not closed:
        if outcome in {E11_GEOMETRY_READY, GEOMETRY_NOT_READY, "truth_snapshot_missing"}:
            outcome = "cleanup_failed"
    audit = {"reset_attempt_count": 0, "environment_launch_count": 0}
    try:
        audit = adapter.reset_failure_audit() if hasattr(adapter, "reset_failure_audit") else adapter.reset_audit()
    except Exception:
        pass
    detected_cause, log_evidence = _collect_runtime_log_evidence(log_snapshot)
    failure_cause = detected_cause if outcome == "reset_failed" else "unknown"
    success = (
        outcome == E11_GEOMETRY_READY
        and reset_succeeded
        and closed
        and not cleanup.has_explicit_failure()
    )
    if success:
        outcome = E11_GEOMETRY_READY
    elif outcome == E11_GEOMETRY_READY:
        outcome = "cleanup_failed"
    real = backend_cls.__name__ == BACKEND_IDENTITY
    record = E11GeometrySmokeRecord(
        execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_E11_GEOMETRY,
        authorized_live_run=AUTHORIZED_LIVE_E11_GEOMETRY_RUN_VALUE,
        backend_identity=getattr(adapter, "backend_identity", backend_cls.__name__),
        opened=adapter.open_succeeded,
        authorization_accepted=True,
        real_execution_performed=real,
        reset_succeeded=reset_succeeded,
        closed=closed,
        outcome=outcome,
        success=success,
        integration_verified=False,
        tested_action_count=getattr(adapter, "_tested_action_count", 0),
        observation_wait_count=getattr(adapter, "_observation_wait_count", 0),
        minerl_launched=real,
        minecraft_launched=real and (reset_succeeded or failure_cause == "minecraft_native_crash"),
        reset_attempts=int(audit.get("reset_attempt_count", 0)),
        environment_launch_count=int(audit.get("environment_launch_count", 0)),
        geometry=geometry,
        cleanup=cleanup,
        failure_cause=failure_cause,
        log_evidence=log_evidence,
        runtime_jar_sha256=runtime.get("jar_sha256", ""),
        traceback_text=traceback_text,
    )
    _write_geometry_evidence(
        record,
        _validate_output_dir(output_dir),
        snapshot=snapshot_payload,
        mission_xml=mission_xml,
        reset_audit=audit,
    )
    return record


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="obsidianlink.env.integration.e11_geometry",
        description="Authorized P1 E11 geometry-only smoke; --check is offline-safe.",
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execution-mode")
    parser.add_argument("--authorized-live-run")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episode-id", default="p1-e11-geometry-smoke")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check:
        if any(
            value is not None
            for value in (args.execution_mode, args.authorized_live_run, args.output_dir)
        ) or args.preflight_only:
            raise E11AuthorizationError("--check cannot be combined with live arguments")
        payload: Mapping[str, Any] = check_e11_geometry_runner()
    else:
        if args.output_dir is None:
            raise E11AuthorizationError("--output-dir is required for E11 geometry live/preflight")
        result = run_authorized_e11_geometry_smoke(
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

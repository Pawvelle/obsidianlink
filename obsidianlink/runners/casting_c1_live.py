"""Authorized one-shot live C1 MineRL smoke runner.

This module is the **only** supported path for a real ``casting_c1_fixed``
MineRL/Minecraft smoke. It does not weaken the offline stub runner: live
execution requires an explicit ``execution_mode="authorized_live_c1"`` and
``authorized_live_run="casting_c1_fixed"``. Catalog ``live_run_allowed`` must
remain ``false``; authorization is recorded per-run, not persisted globally.

Constraints (fail closed):

* at most one real env factory call per process;
* ``max_reset_attempts=1`` (no second Minecraft instance on reset failure);
* no caller-supplied backend / env_factory;
* no Gradle;
* no model API;
* output must be a fresh directory under ``runs/casting_c1_fixed/``.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from obsidianlink.core.task_catalog import load_task_catalog
from obsidianlink.core.types import TaskInstance
from obsidianlink.drivers.casting_c1 import (
    AGENT_ID,
    CastingPlanStep,
    build_casting_action_plan,
    run_casting_c1_driver,
)
from obsidianlink.env import minerl_backend as minerl_backend_module
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
from obsidianlink.evaluation.casting import CastingEvaluator
from obsidianlink.runners.casting_c1_live_smoke import (
    EXPERIMENT_PATH,
    FROZEN_CANONICAL_NAME,
    FROZEN_COMPATIBILITY_ID,
    FROZEN_TARGET_CELL,
    FROZEN_WORKFLOW,
    REQUIRED_EVIDENCE_FILES,
    ROOT,
    TASK_PATH,
    _ObservationCapturingBackend,
    _append_jsonl,
    _close_backend_with_retry,
    _code_version_snapshot,
    _contains_evaluator_token,
    _json_ready,
    _plans_equal,
    _sanitize_public_mapping,
    _validate_task_identity,
    _write_png,
    load_frozen_c1_task,
)


EXECUTION_MODE_AUTHORIZED_LIVE_C1 = "authorized_live_c1"
AUTHORIZED_LIVE_RUN_VALUE = "casting_c1_fixed"
FORMAL_C1_RUNS_ROOT = (ROOT / "runs" / "casting_c1_fixed").resolve()

# Process-wide once guard: a second real factory call in this process is
# always rejected, even if a caller somehow re-enters the runner.
_PROCESS_ENV_FACTORY_CALLS = 0
_PROCESS_ENV_FACTORY_LOCK = threading.Lock()
_PROCESS_LIVE_RUN_STARTED = False

DEFAULT_WALL_CLOCK_SECONDS = 900
MAX_CLOSE_ATTEMPTS = 2
TERMINATED_REASON = "driver_done"

REQUIRED_LIVE_EVIDENCE_FILES: tuple[str, ...] = REQUIRED_EVIDENCE_FILES + (
    "runtime_preflight.json",
    "authorization.json",
    "process_lifecycle.jsonl",
)

_FIXED_PRODUCTION_ENV_FACTORY = minerl_backend_module._default_env_factory
_MUTED_OPTIONS_TEXT = """version:2586
soundCategory_master:0.0
soundCategory_music:0.0
soundCategory_record:0.0
soundCategory_weather:0.0
soundCategory_block:0.0
soundCategory_hostile:0.0
soundCategory_neutral:0.0
soundCategory_player:0.0
soundCategory_ambient:0.0
soundCategory_voice:0.0
"""


class C1LivePreflightError(ValueError):
    """Raised when authorized live C1 preflight fails closed."""


class C1LiveAuthorizationError(C1LivePreflightError):
    """Raised when live authorization parameters are missing or wrong."""


@dataclass
class _IsolatedMinecraftRuntime:
    temporary_directory: tempfile.TemporaryDirectory[str]
    instance_manager: Any
    previous_minecraft_dir: str
    runtime_dir: Path

    def close(self) -> None:
        self.instance_manager.MINECRAFT_DIR = self.previous_minecraft_dir
        self.temporary_directory.cleanup()


def _copy_muted_minecraft_runtime(source: Path, destination: Path) -> None:
    """Copy the exact built runtime and add a sound-disabled options file."""
    source = source.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "build" / "libs").mkdir(parents=True)
    jar = source / "build" / "libs" / "mcprec-6.13.jar"
    launch = source / "launchClient.sh"
    if not jar.is_file() or not launch.is_file():
        raise C1LivePreflightError(
            "vendored Minecraft runtime is missing jar or launchClient.sh"
        )
    shutil.copy2(jar, destination / "build" / "libs" / jar.name)
    shutil.copy2(launch, destination / launch.name)
    (destination / "options.txt").write_text(
        _MUTED_OPTIONS_TEXT,
        encoding="utf-8",
    )


def _prepare_isolated_muted_minecraft_runtime() -> _IsolatedMinecraftRuntime:
    """Point MineRL at a disposable copy of the same vendored runtime.

    The native LWJGL STB Vorbis decoder has crashed intermittently on arm64.
    Muting all sound categories prevents audio decoding without changing the
    MineRL jar, Minecraft version, JDK, or the independent vendor repository.
    """
    from minerl.env.malmo import InstanceManager

    expected_source = (
        ROOT / "vendor" / "minerl" / "minerl" / "MCP-Reborn"
    ).resolve()
    current_source = Path(InstanceManager.MINECRAFT_DIR).resolve()
    if current_source != expected_source:
        raise C1LivePreflightError(
            "MineRL runtime source must be the pinned vendored MCP-Reborn"
        )
    temporary_directory = tempfile.TemporaryDirectory(
        prefix="obsidianlink-c1-muted-runtime-"
    )
    runtime_dir = Path(temporary_directory.name) / "MCP-Reborn"
    try:
        _copy_muted_minecraft_runtime(current_source, runtime_dir)
        previous = InstanceManager.MINECRAFT_DIR
        InstanceManager.MINECRAFT_DIR = str(runtime_dir)
        return _IsolatedMinecraftRuntime(
            temporary_directory=temporary_directory,
            instance_manager=InstanceManager,
            previous_minecraft_dir=previous,
            runtime_dir=runtime_dir,
        )
    except Exception:
        temporary_directory.cleanup()
        raise


def _require_no_gradle(allow_gradle: bool) -> None:
    if allow_gradle is not False:
        raise C1LivePreflightError(
            "Gradle is not authorized for this live C1 smoke; "
            "allow_gradle must be exactly False"
        )


def _validate_authorized_live_output_dir(output_dir: Path) -> Path:
    resolved = Path(output_dir)
    if not resolved.is_absolute():
        raise C1LivePreflightError("output_dir must be an absolute path")
    resolved = resolved.resolve()
    if resolved.exists() or resolved.is_symlink():
        raise C1LivePreflightError(
            f"output_dir must not already exist: {resolved}"
        )
    try:
        resolved.relative_to(FORMAL_C1_RUNS_ROOT)
    except ValueError as error:
        raise C1LivePreflightError(
            f"output_dir must be under {FORMAL_C1_RUNS_ROOT}"
        ) from error
    if resolved == FORMAL_C1_RUNS_ROOT:
        raise C1LivePreflightError(
            "output_dir must be a unique child of runs/casting_c1_fixed/"
        )
    if resolved.parent != FORMAL_C1_RUNS_ROOT:
        raise C1LivePreflightError(
            "output_dir must be a direct child of runs/casting_c1_fixed/"
        )
    return resolved


def _validate_catalog_policy_unchanged() -> None:
    catalog = load_task_catalog(ROOT / "benchmark/catalog/tasks.json")
    entry = next(
        (
            item
            for item in catalog.entries
            if item.compatibility_id == FROZEN_COMPATIBILITY_ID
        ),
        None,
    )
    if entry is None:
        raise C1LivePreflightError(
            f"catalog entry {FROZEN_COMPATIBILITY_ID!r} is missing"
        )
    if entry.live_run_allowed:
        raise C1LivePreflightError(
            "catalog live_run_allowed must remain false; "
            "authorization is per-run only"
        )
    if entry.canonical_name != FROZEN_CANONICAL_NAME:
        raise C1LivePreflightError("canonical task name mismatch")
    if catalog.active_phase != "P1-REAL-MINERL-ENVIRONMENT-VALIDATION":
        raise C1LivePreflightError(
            "active catalog phase must remain P1 environment validation"
        )
    if entry.kind != "legacy" or entry.benchmark_visible:
        raise C1LivePreflightError("C1 compatibility entry must remain quarantined legacy")


def collect_runtime_preflight(*, dry_run: bool = True) -> dict[str, Any]:
    """Read-only runtime preflight. Never starts MineRL or Gradle."""
    java_home = Path("/opt/anaconda3/envs/mc-agent")
    java_bin = java_home / "bin" / "java"
    java_version = None
    if java_bin.is_file():
        proc = subprocess.run(
            [str(java_bin), "-version"],
            check=False,
            capture_output=True,
            text=True,
        )
        java_version = (proc.stderr or proc.stdout or "").strip().splitlines()[:3]

    vendor_mcp = ROOT / "vendor" / "minerl" / "minerl" / "MCP-Reborn"
    site_mcp = Path(
        "/opt/anaconda3/envs/mc-agent/lib/python3.10/site-packages/minerl/MCP-Reborn"
    )
    jar_candidates = [
        vendor_mcp / "build" / "libs" / "mcprec-6.13.jar",
        site_mcp / "build" / "libs" / "mcprec-6.13.jar",
    ]
    existing_jars = [str(path) for path in jar_candidates if path.is_file()]
    launch_scripts = [
        str(path)
        for path in (
            vendor_mcp / "launchClient.sh",
            site_mcp / "launchClient.sh",
        )
        if path.is_file()
    ]

    import_versions: dict[str, str | None] = {}
    for name in ("gym", "numpy"):
        try:
            module = __import__(name)
            import_versions[name] = getattr(module, "__version__", None)
        except Exception as error:  # noqa: BLE001 - preflight must record
            import_versions[name] = f"import_failed:{type(error).__name__}"

    minerl_path = None
    try:
        import minerl

        minerl_path = str(Path(minerl.__file__).resolve())
        import_versions["minerl"] = getattr(minerl, "__version__", None)
    except Exception as error:  # noqa: BLE001
        import_versions["minerl"] = f"import_failed:{type(error).__name__}"

    caps = MineRLEnvironmentBackend.casting_c1_capabilities().as_dict()
    code_version = _code_version_snapshot()
    vendor_status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT / "vendor" / "minerl",
        check=False,
        capture_output=True,
        text=True,
    )

    gradle_needed = len(existing_jars) == 0 or len(launch_scripts) == 0
    return {
        "dry_run": bool(dry_run),
        "python_version_expected": "3.10.20",
        "java_bin": str(java_bin) if java_bin.is_file() else None,
        "java_version_lines": java_version,
        "java_home_pinned": str(java_home),
        "import_versions": import_versions,
        "minerl_module_path": minerl_path,
        "compiled_jars": existing_jars,
        "launch_scripts": launch_scripts,
        "launch_uses_java_jar_not_gradle": True,
        "gradle_needed": gradle_needed,
        "formal_c1_runs_root_exists": FORMAL_C1_RUNS_ROOT.exists(),
        "capability_manifest": caps,
        "catalog_live_run_allowed": False,
        "code_version": code_version,
        "vendor_minerl_status_short": (
            vendor_status.stdout.splitlines()
            if vendor_status.returncode == 0
            else []
        ),
        "note": (
            "read-only preflight; Minecraft was not started; "
            "Gradle was not invoked; live execution uses an ephemeral copy "
            "of the same vendored jar with master sound disabled"
        ),
    }


def preflight_authorized_c1_live(
    *,
    output_dir: Path | str,
    execution_mode: str,
    authorized_live_run: str,
    task: TaskInstance | None = None,
    plan: Sequence[CastingPlanStep] | None = None,
    allow_gradle: bool = False,
    request_model: bool = False,
    env_factory: Any = None,
    backend: Any = None,
    wall_clock_seconds: int = DEFAULT_WALL_CLOCK_SECONDS,
) -> dict[str, Any]:
    """Validate live inputs before any real environment is created."""
    if execution_mode != EXECUTION_MODE_AUTHORIZED_LIVE_C1:
        raise C1LiveAuthorizationError(
            f"execution_mode must be {EXECUTION_MODE_AUTHORIZED_LIVE_C1!r}"
        )
    if authorized_live_run != AUTHORIZED_LIVE_RUN_VALUE:
        raise C1LiveAuthorizationError(
            "authorized_live_run must be exactly "
            f"{AUTHORIZED_LIVE_RUN_VALUE!r}"
        )
    _require_no_gradle(allow_gradle)
    if request_model:
        raise C1LivePreflightError("model API is not authorized")
    if env_factory is not None:
        raise C1LivePreflightError(
            "caller-supplied env_factory is forbidden for live C1"
        )
    if backend is not None:
        raise C1LivePreflightError(
            "caller-supplied backend is forbidden for live C1"
        )
    if type(wall_clock_seconds) is bool or type(wall_clock_seconds) is not int:
        raise C1LivePreflightError(
            "wall_clock_seconds must be an int >= 60 (bool rejected)"
        )
    if wall_clock_seconds < 60:
        raise C1LivePreflightError(
            "wall_clock_seconds must be an int >= 60"
        )

    FORMAL_C1_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    resolved_output = _validate_authorized_live_output_dir(Path(output_dir))
    resolved_task = task if task is not None else load_frozen_c1_task()
    try:
        _validate_task_identity(resolved_task)
    except Exception as error:  # noqa: BLE001 - normalize to live preflight error
        raise C1LivePreflightError(str(error)) from error
    if resolved_task.workflow != FROZEN_WORKFLOW:
        raise C1LivePreflightError("workflow mismatch")
    _validate_catalog_policy_unchanged()

    caps = MineRLEnvironmentBackend.casting_c1_capabilities()
    if not (
        caps.can_select_water_bucket
        and caps.can_select_lava_bucket
        and caps.can_use_water_bucket
        and caps.can_use_lava_bucket
        and caps.exposes_public_inventory
        and caps.exposes_selected_item
        and caps.exposes_target_block_truth
        and caps.exposes_fluid_truth
    ):
        raise C1LivePreflightError(
            "production casting_c1_capabilities incomplete for live C1"
        )

    expected_plan = build_casting_action_plan()
    supplied_plan = tuple(plan) if plan is not None else expected_plan
    if not _plans_equal(supplied_plan, expected_plan):
        raise C1LivePreflightError(
            "plan must exactly match build_casting_action_plan()"
        )

    runtime = collect_runtime_preflight(dry_run=True)
    if runtime["gradle_needed"]:
        raise C1LivePreflightError(
            "compiled MineRL jar or launchClient.sh missing; "
            "Gradle would be required and is not authorized"
        )
    return {
        "output_dir": str(resolved_output),
        "runtime_preflight": runtime,
        "authorized_live_run": AUTHORIZED_LIVE_RUN_VALUE,
        "execution_mode": EXECUTION_MODE_AUTHORIZED_LIVE_C1,
    }


def reset_authorized_live_process_guards_for_tests() -> None:
    """Reset process-once guards. Intended for offline unit tests only."""
    global _PROCESS_ENV_FACTORY_CALLS, _PROCESS_LIVE_RUN_STARTED
    with _PROCESS_ENV_FACTORY_LOCK:
        _PROCESS_ENV_FACTORY_CALLS = 0
        _PROCESS_LIVE_RUN_STARTED = False


def _once_production_env_factory(task: TaskInstance) -> Any:
    """Call the fixed production factory at most once per process."""
    global _PROCESS_ENV_FACTORY_CALLS
    with _PROCESS_ENV_FACTORY_LOCK:
        if _PROCESS_ENV_FACTORY_CALLS >= 1:
            raise RuntimeError(
                "authorized live C1 allows only one real env factory call "
                "per process"
            )
        _PROCESS_ENV_FACTORY_CALLS += 1
    # Resolve through the module attribute so offline tests can patch
    # ``minerl_backend_module._default_env_factory`` without opening a
    # caller-supplied injection seam on the live runner API.
    return minerl_backend_module._default_env_factory(task)


def _authorization_record(
    *,
    output_dir: Path,
    wall_clock_seconds: int,
) -> dict[str, Any]:
    return {
        "authorized_live_run": AUTHORIZED_LIVE_RUN_VALUE,
        "execution_mode": EXECUTION_MODE_AUTHORIZED_LIVE_C1,
        "scope": {
            "task": AUTHORIZED_LIVE_RUN_VALUE,
            "workflow": FROZEN_WORKFLOW,
            "family": "casting",
            "mode": "single",
            "level": "C1",
            "layout": "fixed",
            "agent_id": AGENT_ID,
            "target_cell": list(FROZEN_TARGET_CELL),
            "max_real_environments": 1,
            "max_episodes": 1,
            "max_reset_attempts": 1,
            "gradle_authorized": False,
            "model_api_authorized": False,
            "c2_to_c5_live_authorized": False,
        },
        "catalog_live_run_allowed_remains_false": True,
        "output_dir": str(output_dir),
        "wall_clock_seconds": wall_clock_seconds,
        "note": (
            "Per-run authorization only. Does not permanently enable "
            "catalog live_run_allowed."
        ),
    }


def _write_manual_review(
    path: Path,
    *,
    driver_status: str,
    evaluator_outcome: str,
    evaluator_success: bool,
    evidence_complete: bool,
    real_env_started: bool,
    water_lava_obsidian_captured: bool | None,
) -> None:
    lines = [
        "# C1 Authorized Live MineRL Smoke — Manual Review",
        "",
        "This bundle is from an **authorized real** C1 MineRL/Minecraft smoke.",
        "It is a C1 capability slice (single target cell), **not** an end-to-end",
        "Nether-entry success.",
        "",
        f"- Real environment started: `{real_env_started}`",
        f"- Driver status: `{driver_status}`",
        f"- Evaluator outcome: `{evaluator_outcome}`",
        f"- Evaluator success: `{evaluator_success}`",
        f"- Evidence complete: `{evidence_complete}`",
        f"- Water/lava/obsidian transition captured (auto): "
        f"`{water_lava_obsidian_captured}`",
        "",
        "Success requires independent CastingEvaluator success on production",
        "typed truth, complete evidence, normal backend close, and no conflict",
        "between automatic evidence and this manual review.",
        "",
        "Driver completed alone is not success. Screenshots alone are not",
        "success. Stub/FakeBackend results are not live success.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _append_lifecycle(
    records: list[dict[str, Any]],
    *,
    event: str,
    clock: Callable[[], float],
    **extra: Any,
) -> None:
    payload = {
        "event": event,
        "timestamp": float(clock()),
        **extra,
    }
    records.append(_json_ready(payload))


@dataclass(frozen=True)
class CastingC1AuthorizedLiveResult:
    execution_mode: str
    output_dir: str
    driver_status: str
    driver_completed: bool
    evaluator_success: bool
    evaluator_outcome: str
    evidence_complete: bool
    close_status: str
    failure_reason: str | None
    close_errors: tuple[str, ...]
    real_env_factory_calls: int
    real_episode_count: int
    water_observed: bool | None
    lava_observed: bool | None
    obsidian_transition_observed: bool | None
    summary: Mapping[str, Any]

    @property
    def overall_success(self) -> bool:
        return (
            self.driver_completed
            and self.evaluator_success
            and self.evidence_complete
            and self.close_status == "closed"
            and self.real_env_factory_calls == 1
            and self.real_episode_count == 1
            and self.water_observed is True
            and self.lava_observed is True
            and self.obsidian_transition_observed is True
            and self.failure_reason is None
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_mode": self.execution_mode,
            "output_dir": self.output_dir,
            "driver_status": self.driver_status,
            "driver_completed": self.driver_completed,
            "evaluator_success": self.evaluator_success,
            "evaluator_outcome": self.evaluator_outcome,
            "evidence_complete": self.evidence_complete,
            "close_status": self.close_status,
            "failure_reason": self.failure_reason,
            "close_errors": list(self.close_errors),
            "real_env_factory_calls": self.real_env_factory_calls,
            "real_episode_count": self.real_episode_count,
            "water_observed": self.water_observed,
            "lava_observed": self.lava_observed,
            "obsidian_transition_observed": self.obsidian_transition_observed,
            "overall_success": self.overall_success,
            "summary": _json_ready(self.summary),
        }


def _truth_flags_from_eval_state(
    eval_state: Any,
) -> tuple[bool | None, bool | None, bool | None]:
    """Extract coarse transition flags without exposing full truth publicly."""
    try:
        water_truth = getattr(eval_state, "water_truth", None)
        lava_truth = getattr(eval_state, "lava_truth", None)
        water = None if water_truth is None else bool(water_truth.present is True)
        lava = None if lava_truth is None else bool(lava_truth.present is True)
        update = getattr(eval_state, "target_update_evidence", None)
        current = getattr(eval_state, "current_target_block", None)
        obsidian = bool(
            current == "obsidian"
            and update is not None
            and getattr(update, "after_block", None) == "obsidian"
        )
        return water, lava, obsidian
    except Exception:  # noqa: BLE001 - never crash evidence on optional flags
        return None, None, None


def run_casting_c1_authorized_live(
    *,
    output_dir: Path | str,
    execution_mode: str,
    authorized_live_run: str,
    task: TaskInstance | None = None,
    plan: tuple[CastingPlanStep, ...] | None = None,
    allow_gradle: bool = False,
    request_model: bool = False,
    wall_clock_seconds: int = DEFAULT_WALL_CLOCK_SECONDS,
    clock: Callable[[], float] | None = None,
    # Explicitly rejected injection seams (kept so misuse fails closed).
    env_factory: Any = None,
    backend: Any = None,
) -> CastingC1AuthorizedLiveResult:
    """Run the authorized one-shot live C1 smoke and write formal evidence."""
    global _PROCESS_LIVE_RUN_STARTED
    clock_fn = clock if clock is not None else time.time

    with _PROCESS_ENV_FACTORY_LOCK:
        if _PROCESS_LIVE_RUN_STARTED:
            raise C1LivePreflightError(
                "this process already started an authorized live C1 run"
            )
        _PROCESS_LIVE_RUN_STARTED = True

    preflight_info = preflight_authorized_c1_live(
        output_dir=output_dir,
        execution_mode=execution_mode,
        authorized_live_run=authorized_live_run,
        task=task,
        plan=plan,
        allow_gradle=allow_gradle,
        request_model=request_model,
        env_factory=env_factory,
        backend=backend,
        wall_clock_seconds=wall_clock_seconds,
    )
    resolved_output = Path(preflight_info["output_dir"])
    resolved_task = task if task is not None else load_frozen_c1_task()
    resolved_plan = plan if plan is not None else build_casting_action_plan()

    # Configure pinned Java before creating the production backend.
    pinned_java_home = Path("/opt/anaconda3/envs/mc-agent")
    os.environ["JAVA_HOME"] = str(pinned_java_home)
    os.environ["PATH"] = (
        f"{pinned_java_home / 'bin'}:{os.environ.get('PATH', '')}"
    )
    # Prefer vendored MineRL (existing project convention for live scripts).
    vendor_root = str((ROOT / "vendor" / "minerl").resolve())
    if vendor_root not in os.environ.get("PYTHONPATH", ""):
        existing = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = (
            f"{vendor_root}:{existing}" if existing else vendor_root
        )

    production_backend = MineRLEnvironmentBackend(
        env_factory=_once_production_env_factory,
        reset_warmup_steps=2,
        max_reset_attempts=1,
    )
    close_status = "not_opened"
    close_errors: tuple[str, ...] = ()
    driver_status = "not_started"
    driver_completed = False
    evaluator_success = False
    evaluator_outcome = "not_evaluated"
    failure_reason: str | None = None
    public_events: list[dict[str, Any]] = []
    evaluator_events: list[dict[str, Any]] = []
    lifecycle: list[dict[str, Any]] = []
    capture: _ObservationCapturingBackend | None = None
    staging_dir: Path | None = None
    staging_complete = False
    evidence_complete = False
    output_created_by_runner = False
    driver_steps_executed = 0
    water_observed: bool | None = None
    lava_observed: bool | None = None
    obsidian_transition_observed: bool | None = None
    real_episode_count = 0
    timed_out = False
    interrupt_received = False
    isolated_runtime: _IsolatedMinecraftRuntime | None = None

    def _on_alarm(_signum: int, _frame: Any) -> None:
        nonlocal timed_out
        timed_out = True
        raise TimeoutError(
            f"authorized live C1 exceeded wall_clock_seconds={wall_clock_seconds}"
        )

    previous_alarm_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(wall_clock_seconds)
    except Exception:  # noqa: BLE001 - platforms without SIGALRM still run
        previous_alarm_handler = None

    def _on_sigint(_signum: int, _frame: Any) -> None:
        nonlocal interrupt_received, failure_reason
        interrupt_received = True
        if failure_reason is None:
            failure_reason = "SIGINT received"
        raise KeyboardInterrupt("authorized live C1 interrupted")

    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _on_sigint)

    _append_lifecycle(
        lifecycle,
        event="preflight_passed",
        clock=clock_fn,
        output_dir=str(resolved_output),
    )

    try:
        if (
            minerl_backend_module._default_env_factory
            is _FIXED_PRODUCTION_ENV_FACTORY
        ):
            isolated_runtime = _prepare_isolated_muted_minecraft_runtime()
            _append_lifecycle(
                lifecycle,
                event="isolated_muted_runtime_prepared",
                clock=clock_fn,
                runtime_dir=str(isolated_runtime.runtime_dir),
                source="vendored_mcp_reborn_exact_jar_copy",
            )
        production_backend.open()
        _append_lifecycle(lifecycle, event="backend_opened", clock=clock_fn)
        capture = _ObservationCapturingBackend(production_backend)
        _append_lifecycle(
            lifecycle,
            event="env_factory_about_to_call_via_reset",
            clock=clock_fn,
            max_reset_attempts=1,
        )
        driver_result = run_casting_c1_driver(
            capture,
            resolved_task,
            plan=resolved_plan,
        )
        real_episode_count = 1 if _PROCESS_ENV_FACTORY_CALLS >= 1 else 0
        _append_lifecycle(
            lifecycle,
            event="driver_finished",
            clock=clock_fn,
            driver_status=driver_result.status,
            steps_executed=driver_result.steps_executed,
            env_factory_calls=_PROCESS_ENV_FACTORY_CALLS,
        )
        driver_status = driver_result.status
        driver_completed = driver_status == "completed"
        driver_steps_executed = driver_result.steps_executed
        public_events.extend(
            _sanitize_public_mapping(event) for event in driver_result.events
        )

        production_backend.mark_terminated(reason=TERMINATED_REASON)
        eval_state = production_backend.get_casting_evaluation_state(
            FROZEN_TARGET_CELL
        )
        water_observed, lava_observed, obsidian_transition_observed = (
            _truth_flags_from_eval_state(eval_state)
        )
        eval_result = CastingEvaluator().evaluate(eval_state)
        evaluator_outcome = eval_result.outcome
        evaluator_success = bool(eval_result.success)
        if not evaluator_success and failure_reason is None:
            failure_reason = f"evaluator outcome={eval_result.outcome}"

        evaluator_events.append(
            {
                "episode_id": resolved_task.task_id,
                "step_id": driver_result.steps_executed,
                "agent_id": AGENT_ID,
                "timestamp": float(clock_fn()),
                "outcome": eval_result.outcome,
                "success": eval_result.success,
                "failure_type": eval_result.failure_type,
            }
        )
        # Keep diagnostics on the evaluator event stream only when they are
        # already part of the public evaluator result surface.
        public_eval_diagnostics = {
            "outcome": eval_result.outcome,
            "success": bool(eval_result.success),
            "failure_type": eval_result.failure_type,
        }
        if hasattr(eval_result, "blocking_conditions"):
            public_eval_diagnostics["blocking_conditions"] = list(
                getattr(eval_result, "blocking_conditions") or ()
            )
        evaluator_events[-1].update(
            {
                key: value
                for key, value in public_eval_diagnostics.items()
                if key not in evaluator_events[-1]
            }
        )
        _append_lifecycle(
            lifecycle,
            event="evaluator_finished",
            clock=clock_fn,
            outcome=eval_result.outcome,
            success=bool(eval_result.success),
        )

        if capture.initial_observation is None or capture.final_observation is None:
            raise RuntimeError("observation capture failed")

        final_inventory = dict(
            capture.final_observation.visible_inventory or {}
        )
        initial_inventory = dict(
            capture.initial_observation.visible_inventory or {}
        )
        inventory_unchanged = final_inventory == initial_inventory
        final_selected_item = capture.final_observation.selected_item

        FORMAL_C1_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{resolved_output.name}.staging-",
                dir=str(FORMAL_C1_RUNS_ROOT),
            )
        )

        shutil.copyfile(TASK_PATH, staging_dir / "task_instance.json")
        shutil.copyfile(EXPERIMENT_PATH, staging_dir / "experiment_config.json")
        (staging_dir / "capability_manifest.json").write_text(
            json.dumps(
                production_backend.capabilities().as_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (staging_dir / "code_version.json").write_text(
            json.dumps(
                _code_version_snapshot(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (staging_dir / "runtime_preflight.json").write_text(
            json.dumps(
                preflight_info["runtime_preflight"],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (staging_dir / "authorization.json").write_text(
            json.dumps(
                _authorization_record(
                    output_dir=resolved_output,
                    wall_clock_seconds=wall_clock_seconds,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _write_png(staging_dir / "initial.png", capture.initial_observation)
        _write_png(staging_dir / "final.png", capture.final_observation)
        _append_jsonl(staging_dir / "events.jsonl", public_events)
        _append_jsonl(staging_dir / "evaluator_events.jsonl", evaluator_events)
        _append_jsonl(staging_dir / "process_lifecycle.jsonl", lifecycle)

        summary = {
            "execution_mode": EXECUTION_MODE_AUTHORIZED_LIVE_C1,
            "authorized_live_run": AUTHORIZED_LIVE_RUN_VALUE,
            "episode_id": resolved_task.task_id,
            "workflow": resolved_task.workflow,
            "agent_id": AGENT_ID,
            "target_cell": list(FROZEN_TARGET_CELL),
            "driver_status": driver_status,
            "driver_completed": driver_completed,
            "driver_steps_executed": driver_steps_executed,
            "evaluator_outcome": evaluator_outcome,
            "evaluator_success": evaluator_success,
            "close_status": close_status,
            "evidence_complete": False,
            "failure_reason": failure_reason,
            "real_env_factory_calls": _PROCESS_ENV_FACTORY_CALLS,
            "real_episode_count": real_episode_count,
            "water_observed": water_observed,
            "lava_observed": lava_observed,
            "obsidian_transition_observed": obsidian_transition_observed,
            "inventory_unchanged": inventory_unchanged,
            "final_selected_item": final_selected_item,
            "initial_visible_inventory": initial_inventory,
            "final_visible_inventory": final_inventory,
            "capability_slice": "casting_c1_fixed",
            "not_nether_entry_success": True,
            "timestamp": float(clock_fn()),
            "note": (
                "Authorized real C1 MineRL smoke. Catalog live_run_allowed "
                "remains false. C1 cell success is not Nether-entry success."
            ),
        }
        if _contains_evaluator_token(summary):
            raise RuntimeError("public summary leaked evaluator-only tokens")
        (staging_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        _write_manual_review(
            staging_dir / "manual_review.md",
            driver_status=driver_status,
            evaluator_outcome=evaluator_outcome,
            evaluator_success=evaluator_success,
            evidence_complete=False,
            real_env_started=_PROCESS_ENV_FACTORY_CALLS >= 1,
            water_lava_obsidian_captured=(
                water_observed is True
                and lava_observed is True
                and obsidian_transition_observed is True
            ),
        )

        staging_complete = all(
            (staging_dir / filename).is_file()
            and (staging_dir / filename).stat().st_size > 0
            for filename in REQUIRED_LIVE_EVIDENCE_FILES
        )
        if not staging_complete:
            raise RuntimeError("staged live evidence bundle is incomplete")
    except C1LivePreflightError:
        raise
    except Exception as error:  # noqa: BLE001 - structured failure required
        if failure_reason is None:
            failure_reason = f"{type(error).__name__}: {error}"
        if not driver_completed and driver_status == "not_started":
            driver_status = "failed"
        _append_lifecycle(
            lifecycle,
            event="run_exception",
            clock=clock_fn,
            error=failure_reason,
            traceback=traceback.format_exc()[-2000:],
        )
    finally:
        try:
            signal.alarm(0)
        except Exception:  # noqa: BLE001
            pass
        if previous_alarm_handler is not None:
            try:
                signal.signal(signal.SIGALRM, previous_alarm_handler)
            except Exception:  # noqa: BLE001
                pass
        try:
            signal.signal(signal.SIGINT, previous_sigint)
        except Exception:  # noqa: BLE001
            pass

        if getattr(production_backend, "_opened", False) and close_status != "closed":
            close_status, close_error_list = _close_backend_with_retry(
                production_backend
            )
            close_errors = tuple(close_error_list)
            _append_lifecycle(
                lifecycle,
                event="backend_close_attempted",
                clock=clock_fn,
                close_status=close_status,
                close_errors=list(close_errors),
            )
            if close_status != "closed" and failure_reason is None:
                failure_reason = "backend close failed"

        if isolated_runtime is not None:
            try:
                isolated_runtime.close()
                _append_lifecycle(
                    lifecycle,
                    event="isolated_muted_runtime_removed",
                    clock=clock_fn,
                )
            except Exception as error:  # noqa: BLE001
                if failure_reason is None:
                    failure_reason = (
                        "isolated runtime cleanup failed: "
                        f"{type(error).__name__}: {error}"
                    )

        if timed_out and failure_reason is None:
            failure_reason = "wall clock timeout"
        if interrupt_received and failure_reason is None:
            failure_reason = "SIGINT received"

        # Ensure formal root exists before finalize / fail-closed write.
        FORMAL_C1_RUNS_ROOT.mkdir(parents=True, exist_ok=True)

        if staging_complete and staging_dir is not None and staging_dir.exists():
            try:
                # Refresh lifecycle file with close events before publish.
                lifecycle_path = staging_dir / "process_lifecycle.jsonl"
                lifecycle_path.write_text("", encoding="utf-8")
                _append_jsonl(lifecycle_path, lifecycle)

                summary_path = staging_dir / "summary.json"
                final_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                final_summary["close_status"] = close_status
                final_summary["evidence_complete"] = True
                final_summary["failure_reason"] = failure_reason
                final_summary["close_errors"] = list(close_errors)
                final_summary["real_env_factory_calls"] = _PROCESS_ENV_FACTORY_CALLS
                final_summary["real_episode_count"] = real_episode_count
                final_summary["water_observed"] = water_observed
                final_summary["lava_observed"] = lava_observed
                final_summary["obsidian_transition_observed"] = (
                    obsidian_transition_observed
                )
                if _contains_evaluator_token(final_summary):
                    raise RuntimeError(
                        "public summary leaked evaluator-only tokens"
                    )
                summary_path.write_text(
                    json.dumps(
                        final_summary,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                _write_manual_review(
                    staging_dir / "manual_review.md",
                    driver_status=driver_status,
                    evaluator_outcome=evaluator_outcome,
                    evaluator_success=evaluator_success,
                    evidence_complete=True,
                    real_env_started=_PROCESS_ENV_FACTORY_CALLS >= 1,
                    water_lava_obsidian_captured=(
                        water_observed is True
                        and lava_observed is True
                        and obsidian_transition_observed is True
                    ),
                )
                if resolved_output.exists() or resolved_output.is_symlink():
                    raise RuntimeError(
                        "output_dir appeared after preflight; refusing overwrite"
                    )
                staging_dir.rename(resolved_output)
                output_created_by_runner = True
                evidence_complete = all(
                    (resolved_output / filename).is_file()
                    and (resolved_output / filename).stat().st_size > 0
                    for filename in REQUIRED_LIVE_EVIDENCE_FILES
                )
            except Exception as error:  # noqa: BLE001
                evidence_complete = False
                if failure_reason is None:
                    failure_reason = f"{type(error).__name__}: {error}"

        if not evidence_complete and not output_created_by_runner:
            # Fail-closed minimal summary at the reserved output path when
            # staging never completed. Never claim success.
            try:
                if not resolved_output.exists():
                    resolved_output.mkdir(parents=True, exist_ok=False)
                fail_summary = {
                    "execution_mode": EXECUTION_MODE_AUTHORIZED_LIVE_C1,
                    "authorized_live_run": AUTHORIZED_LIVE_RUN_VALUE,
                    "episode_id": resolved_task.task_id,
                    "workflow": resolved_task.workflow,
                    "agent_id": AGENT_ID,
                    "target_cell": list(FROZEN_TARGET_CELL),
                    "driver_status": driver_status,
                    "driver_completed": driver_completed,
                    "evaluator_outcome": evaluator_outcome,
                    "evaluator_success": evaluator_success,
                    "close_status": close_status,
                    "evidence_complete": False,
                    "failure_reason": failure_reason,
                    "real_env_factory_calls": _PROCESS_ENV_FACTORY_CALLS,
                    "real_episode_count": real_episode_count,
                    "note": "fail-closed live summary; evidence incomplete",
                }
                (resolved_output / "summary.json").write_text(
                    json.dumps(
                        fail_summary,
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                (resolved_output / "authorization.json").write_text(
                    json.dumps(
                        _authorization_record(
                            output_dir=resolved_output,
                            wall_clock_seconds=wall_clock_seconds,
                        ),
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                _append_jsonl(
                    resolved_output / "process_lifecycle.jsonl", lifecycle
                )
            except Exception:  # noqa: BLE001 - never mask primary failure
                pass

        if staging_dir is not None and staging_dir.exists():
            try:
                shutil.rmtree(staging_dir, ignore_errors=True)
            except Exception:  # noqa: BLE001
                pass

    summary_payload: dict[str, Any]
    if output_created_by_runner and (resolved_output / "summary.json").is_file():
        summary_payload = json.loads(
            (resolved_output / "summary.json").read_text(encoding="utf-8")
        )
    else:
        summary_payload = {
            "execution_mode": EXECUTION_MODE_AUTHORIZED_LIVE_C1,
            "driver_status": driver_status,
            "evaluator_outcome": evaluator_outcome,
            "evaluator_success": evaluator_success,
            "evidence_complete": evidence_complete,
            "close_status": close_status,
            "failure_reason": failure_reason,
        }

    return CastingC1AuthorizedLiveResult(
        execution_mode=EXECUTION_MODE_AUTHORIZED_LIVE_C1,
        output_dir=str(resolved_output),
        driver_status=driver_status,
        driver_completed=driver_completed,
        evaluator_success=evaluator_success,
        evaluator_outcome=evaluator_outcome,
        evidence_complete=evidence_complete,
        close_status=close_status,
        failure_reason=failure_reason,
        close_errors=close_errors,
        real_env_factory_calls=_PROCESS_ENV_FACTORY_CALLS,
        real_episode_count=real_episode_count,
        water_observed=water_observed,
        lava_observed=lava_observed,
        obsidian_transition_observed=obsidian_transition_observed,
        summary=summary_payload,
    )


def allocate_live_run_dir(*, clock: Callable[[], datetime] | None = None) -> Path:
    """Allocate a fresh absolute run directory under formal C1 runs root."""
    now = clock() if clock is not None else datetime.now(timezone.utc).astimezone()
    stamp = now.strftime("%Y%m%d-%H%M%S")
    candidate = FORMAL_C1_RUNS_ROOT / stamp
    if candidate.exists():
        candidate = FORMAL_C1_RUNS_ROOT / f"{stamp}-{os.getpid()}"
    return candidate.resolve()

"""Offline C1 live-smoke runner for ``casting_c1_fixed``.

The runner wires the deterministic C1 driver, production
:class:`~obsidianlink.env.minerl_backend.MineRLEnvironmentBackend`,
episode termination, independent :class:`~obsidianlink.evaluation.casting.CastingEvaluator`,
and a complete evidence bundle. Only ``execution_mode="offline_stub"`` is
supported; real MineRL/Minecraft is never started from this module.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from obsidianlink.core.task_catalog import load_task_catalog
from obsidianlink.core.types import BackendStep, MacroAction, Observation, TaskInstance
from obsidianlink.drivers.casting_c1 import (
    AGENT_ID,
    CastingPlanStep,
    build_casting_action_plan,
    run_casting_c1_driver,
)
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
from obsidianlink.env.portal_spec import (
    PORTAL_GRID_BLOCKS,
    PORTAL_GRID_MAX,
    PORTAL_GRID_MIN,
    PORTAL_GRID_MISSING_ID,
    PORTAL_GRID_SHAPE,
    PORTAL_GRID_SIZE,
    PORTAL_SELECTED_ITEM_NAME,
    PortalA0EnvSpec,
)
from obsidianlink.evaluation.casting import CastingEvaluator


ROOT = Path(__file__).resolve().parents[2]
TASK_PATH = ROOT / "benchmark/instances/active/casting_c1_fixed.json"
EXPERIMENT_PATH = ROOT / "configs/experiments/active/casting_c1_contract.json"
CATALOG_PATH = ROOT / "benchmark/catalog/tasks.json"
FORMAL_RUNS_DIR = (ROOT / "runs").resolve()

EXECUTION_MODE_OFFLINE_STUB = "offline_stub"
SUPPORTED_EXECUTION_MODES: frozenset[str] = frozenset({EXECUTION_MODE_OFFLINE_STUB})

FROZEN_FAMILY = "casting"
FROZEN_MODE = "single"
FROZEN_LEVEL = "C1"
FROZEN_LAYOUT = "fixed"
FROZEN_WORKFLOW = "casting_c1_fixed"
FROZEN_COMPATIBILITY_ID = "casting_c1_fixed"
FROZEN_CANONICAL_NAME = "casting_s_c1_fixed"
FROZEN_TARGET_CELL: tuple[int, int, int] = (2, 4, 3)
FROZEN_MECHANICS = "vanilla_water_lava_block_update"
FROZEN_INVENTORY: Mapping[str, int] = MappingProxyType(
    {"water_bucket": 1, "lava_bucket": 1, "cobblestone": 8}
)

C1_HOTBAR_SLOTS: Mapping[str, int] = MappingProxyType(
    {"water_bucket": 1, "lava_bucket": 2, "cobblestone": 3}
)

REQUIRED_EVIDENCE_FILES: tuple[str, ...] = (
    "task_instance.json",
    "experiment_config.json",
    "capability_manifest.json",
    "code_version.json",
    "initial.png",
    "final.png",
    "events.jsonl",
    "evaluator_events.jsonl",
    "summary.json",
    "manual_review.md",
)

EVALUATOR_ONLY_TOKENS: frozenset[str] = frozenset(
    {
        "target_block_truth",
        "fluid_truth",
        "portal_grid",
        "water_truth",
        "lava_truth",
        "target_update_evidence",
        "CastingEvaluationState",
        "initial_target_block",
        "current_target_block",
        "baseline_fluid_grid",
        "current_fluid_grid",
        "previous_truth_grid",
        "first_water_step_by_offset",
        "first_lava_step_by_offset",
        "first_obsidian_step_by_offset",
        "cast_credit_history",
    }
)

TERMINATED_REASON = "driver_done"


class C1SmokePreflightError(ValueError):
    """Raised when C1 live-smoke preflight checks fail closed."""


def load_frozen_c1_task() -> TaskInstance:
    """Load the frozen ``casting_c1_fixed`` task instance."""
    payload = json.loads(TASK_PATH.read_text(encoding="utf-8"))
    return TaskInstance.from_dict(payload)


def build_default_c1_plan() -> tuple[CastingPlanStep, ...]:
    """Return the canonical bounded C1 driver plan."""
    return build_casting_action_plan()


def _flat_offset(offset: tuple[int, int, int]) -> int:
    return (
        (offset[1] - PORTAL_GRID_MIN[1]) * PORTAL_GRID_SHAPE[0] * PORTAL_GRID_SHAPE[2]
        + (offset[2] - PORTAL_GRID_MIN[2]) * PORTAL_GRID_SHAPE[0]
        + (offset[0] - PORTAL_GRID_MIN[0])
    )


def _cell_index_in_flat_grid(cell: tuple[int, int, int]) -> int | None:
    x = cell[0] - PORTAL_GRID_MIN[0]
    y = cell[1] - PORTAL_GRID_MIN[1]
    z = cell[2] - PORTAL_GRID_MIN[2]
    x_size = PORTAL_GRID_MAX[0] - PORTAL_GRID_MIN[0] + 1
    y_size = PORTAL_GRID_MAX[1] - PORTAL_GRID_MIN[1] + 1
    z_size = PORTAL_GRID_MAX[2] - PORTAL_GRID_MIN[2] + 1
    if x < 0 or x >= x_size or y < 0 or y >= y_size or z < 0 or z >= z_size:
        return None
    return _flat_offset(cell)


def _grid_from_blocks(
    blocks: Mapping[tuple[int, int, int], str],
) -> np.ndarray:
    grid = np.full(PORTAL_GRID_SIZE, PORTAL_GRID_BLOCKS.index("air"), dtype=np.int32)
    for offset, block_name in blocks.items():
        if block_name not in PORTAL_GRID_BLOCKS:
            raise ValueError(f"unknown portal grid block: {block_name!r}")
        index = _cell_index_in_flat_grid(offset)
        if index is None:
            raise ValueError(f"cell {offset!r} is outside the portal grid")
        grid[index] = PORTAL_GRID_BLOCKS.index(block_name)
    return grid


def _block_at(grid: np.ndarray, cell: tuple[int, int, int]) -> str:
    index = _cell_index_in_flat_grid(cell)
    if index is None:
        return "missing"
    block_id = int(grid[index])
    if 0 <= block_id < len(PORTAL_GRID_BLOCKS):
        return PORTAL_GRID_BLOCKS[block_id]
    return "missing"


def _inventory_payload(inventory: Mapping[str, int]) -> dict[str, Any]:
    return {
        item: np.asarray(quantity, dtype=np.int64)
        for item, quantity in inventory.items()
    }


class C1ReactiveStubEnv:
    """MineRL-shaped reactive stub for offline C1 smoke validation.

    The stub simulates vanilla water/lava block updates at the frozen
    target cell by inspecting low-level hotbar + ``use`` actions:

    * ``use`` + lava bucket (hotbar slot 2) on ``air`` → ``lava``;
    * ``use`` + water bucket (hotbar slot 1) on ``lava`` → ``water``;
    * the step after water placement → ``obsidian``.
    """

    def __init__(
        self,
        task: TaskInstance,
        *,
        produce_obsidian: bool = True,
        target_cell: tuple[int, int, int] = FROZEN_TARGET_CELL,
    ) -> None:
        self.action_space = PortalA0EnvSpec().action_space
        self._task = task
        self._produce_obsidian = produce_obsidian
        self._target_cell = target_cell
        spawn = task.spawn_positions[AGENT_ID]
        self._grid_target_cell = tuple(
            target_cell[index] - spawn[index] for index in range(3)
        )
        self._grid = _grid_from_blocks({})
        self._selected_item = "lava_bucket"
        self._pending_obsidian = False
        self._seed_value: int | None = None
        self.closed = False
        self.step_count = 0
        self._inventory = {
            str(item): int(quantity)
            for item, quantity in dict(
                task.initial_inventories.get(AGENT_ID, {})
            ).items()
            if isinstance(quantity, int) and not isinstance(quantity, bool)
        }

    def seed(self, value: int) -> None:
        self._seed_value = value

    def _build_raw(self) -> dict[str, Any]:
        return {
            "pov": np.zeros((360, 640, 3), dtype=np.uint8),
            "inventory": _inventory_payload(self._inventory),
            "portal_grid": self._grid.copy(),
            "portal_grid_origin": np.asarray((0, 64, 0), dtype=np.int32),
            "portal_dimension": np.asarray("minecraft:overworld"),
            "location_stats": {
                "xpos": 0.5,
                "ypos": 64.0,
                "zpos": 0.5,
            },
            "use_item": {
                "obsidian": np.asarray(0, dtype=np.int64),
                "flint_and_steel": np.asarray(0, dtype=np.int64),
            },
            PORTAL_SELECTED_ITEM_NAME: {
                "mainhand": {"type": self._selected_item},
            },
        }

    def reset(self) -> dict[str, Any]:
        self._grid = _grid_from_blocks({})
        self._selected_item = "lava_bucket"
        self._pending_obsidian = False
        self.step_count = 0
        self._inventory = {
            str(item): int(quantity)
            for item, quantity in dict(
                self._task.initial_inventories.get(AGENT_ID, {})
            ).items()
            if isinstance(quantity, int) and not isinstance(quantity, bool)
        }
        return self._build_raw()

    @staticmethod
    def _slot_from_action(action: Mapping[str, Any]) -> int | None:
        for slot in range(1, 7):
            if int(action.get(f"hotbar.{slot}", 0)) == 1:
                return slot
        return None

    @staticmethod
    def _item_for_slot(slot: int) -> str | None:
        for item, mapped in C1_HOTBAR_SLOTS.items():
            if mapped == slot:
                return item
        return None

    def _consume(self, item: str) -> bool:
        quantity = self._inventory.get(item, 0)
        if quantity < 1:
            return False
        self._inventory[item] = quantity - 1
        return True

    def _apply_reactive_update(self, action: Mapping[str, Any]) -> None:
        slot = self._slot_from_action(action)
        if slot is None:
            return
        item = self._item_for_slot(slot)
        if item is not None:
            self._selected_item = item
        use = int(action.get("use", 0)) == 1
        if not use or item is None:
            return
        current = _block_at(self._grid, self._grid_target_cell)
        index = _cell_index_in_flat_grid(self._grid_target_cell)
        if index is None:
            return
        if item == "cobblestone":
            # Support placement consumes cobble even though the target cell
            # itself is reserved for fluid/obsidian truth.
            self._consume("cobblestone")
            return
        if item == "lava_bucket" and current == "air":
            if self._consume("lava_bucket"):
                self._grid[index] = PORTAL_GRID_BLOCKS.index("lava")
        elif item == "water_bucket" and current == "lava":
            if self._consume("water_bucket"):
                self._grid[index] = PORTAL_GRID_BLOCKS.index("water")
                self._pending_obsidian = self._produce_obsidian

    def step(self, action: Mapping[str, Any]):
        if self._pending_obsidian:
            index = _cell_index_in_flat_grid(self._grid_target_cell)
            if index is not None:
                self._grid[index] = PORTAL_GRID_BLOCKS.index("obsidian")
            self._pending_obsidian = False
        self._apply_reactive_update(action)
        self.step_count += 1
        return self._build_raw(), 0.0, False, {}

    def close(self) -> None:
        self.closed = True


class OfflineC1StubEnvFactory:
    """Typed, closed offline factory accepted by the smoke runner.

    Arbitrary callables are intentionally rejected: accepting a generic
    ``env_factory`` would let a caller smuggle the real MineRL factory into an
    API that promises never to start Minecraft.
    """

    __slots__ = ("_produce_obsidian", "_created_envs")

    def __init__(self, *, produce_obsidian: bool = True) -> None:
        if type(produce_obsidian) is not bool:
            raise ValueError("produce_obsidian must be a boolean")
        self._produce_obsidian = produce_obsidian
        self._created_envs: list[C1ReactiveStubEnv] = []

    @property
    def created_envs(self) -> tuple[C1ReactiveStubEnv, ...]:
        return tuple(self._created_envs)

    def __call__(self, task: TaskInstance) -> C1ReactiveStubEnv:
        env = C1ReactiveStubEnv(
            task,
            produce_obsidian=self._produce_obsidian,
        )
        self._created_envs.append(env)
        return env


def build_offline_stub_env_factory(
    *,
    produce_obsidian: bool = True,
) -> OfflineC1StubEnvFactory:
    """Return the only factory type accepted by the offline runner."""

    return OfflineC1StubEnvFactory(produce_obsidian=produce_obsidian)


def _plans_equal(
    left: Sequence[CastingPlanStep],
    right: Sequence[CastingPlanStep],
) -> bool:
    if len(left) != len(right):
        return False
    for left_step, right_step in zip(left, right):
        if (
            left_step.label != right_step.label
            or left_step.phase != right_step.phase
            or left_step.relevant_action != right_step.relevant_action
            or left_step.action != right_step.action
        ):
            return False
    return True


def _validate_output_dir(output_dir: Path) -> None:
    if not output_dir.is_absolute():
        raise C1SmokePreflightError("output_dir must be an absolute path")
    resolved = output_dir.resolve()
    try:
        resolved.relative_to(FORMAL_RUNS_DIR)
    except ValueError:
        return
    raise C1SmokePreflightError(
        "output_dir must not be under the formal runs/ directory"
    )


def _validate_new_output_dir(output_dir: Path) -> None:
    _validate_output_dir(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise C1SmokePreflightError("output_dir must not already exist")
    parent = output_dir.parent
    if not parent.exists() or not parent.is_dir():
        raise C1SmokePreflightError(
            "output_dir parent must be an existing directory"
        )


def _validate_task_identity(task: TaskInstance) -> None:
    if task.workflow != FROZEN_WORKFLOW:
        raise C1SmokePreflightError(
            f"workflow must be {FROZEN_WORKFLOW!r}, got {task.workflow!r}"
        )
    if task.agent_ids != (AGENT_ID,):
        raise C1SmokePreflightError(
            f"agent_ids must be ({AGENT_ID!r},), got {task.agent_ids!r}"
        )
    params = task.scenario_parameters
    if not isinstance(params, Mapping):
        raise C1SmokePreflightError("scenario_parameters must be a mapping")
    family = params.get("task_family")
    mode = params.get("agent_mode")
    level = params.get("task_level")
    layout = params.get("layout_type")
    if family != FROZEN_FAMILY:
        raise C1SmokePreflightError(
            f"task_family must be {FROZEN_FAMILY!r}, got {family!r}"
        )
    if mode != FROZEN_MODE:
        raise C1SmokePreflightError(
            f"agent_mode must be {FROZEN_MODE!r}, got {mode!r}"
        )
    if level != FROZEN_LEVEL:
        raise C1SmokePreflightError(
            f"task_level must be {FROZEN_LEVEL!r}, got {level!r}"
        )
    if layout != FROZEN_LAYOUT:
        raise C1SmokePreflightError(
            f"layout_type must be {FROZEN_LAYOUT!r}, got {layout!r}"
        )
    target = params.get("target_cell")
    if list(target) != list(FROZEN_TARGET_CELL):
        raise C1SmokePreflightError(
            f"target_cell must be {list(FROZEN_TARGET_CELL)}, got {target!r}"
        )
    mechanics = params.get("mechanics_required")
    if mechanics != FROZEN_MECHANICS:
        raise C1SmokePreflightError(
            f"mechanics_required must be {FROZEN_MECHANICS!r}, got {mechanics!r}"
        )
    inventory = dict(task.initial_inventories.get(AGENT_ID, {}))
    if inventory != dict(FROZEN_INVENTORY):
        raise C1SmokePreflightError(
            "initial inventory must exactly match the frozen C1 inventory"
        )
    if task != load_frozen_c1_task():
        raise C1SmokePreflightError(
            "task must exactly match the frozen casting_c1_fixed TaskInstance"
        )


def _validate_catalog_live_run_allowed() -> None:
    catalog = load_task_catalog(CATALOG_PATH)
    entry = next(
        (
            item
            for item in catalog.entries
            if item.compatibility_id == FROZEN_COMPATIBILITY_ID
        ),
        None,
    )
    if entry is None:
        raise C1SmokePreflightError(
            f"catalog entry {FROZEN_COMPATIBILITY_ID!r} is missing"
        )
    if entry.live_run_allowed:
        raise C1SmokePreflightError(
            f"{FROZEN_COMPATIBILITY_ID} live_run_allowed must remain false"
        )
    if entry.canonical_name != FROZEN_CANONICAL_NAME:
        raise C1SmokePreflightError(
            "canonical task name mismatch for casting_c1_fixed"
        )


def preflight_c1_live_smoke(
    *,
    output_dir: Path | str,
    execution_mode: str = EXECUTION_MODE_OFFLINE_STUB,
    task: TaskInstance | None = None,
    plan: Sequence[CastingPlanStep] | None = None,
    env_factory: OfflineC1StubEnvFactory | None = None,
    request_live: bool = False,
    allow_live_run_override: bool | None = None,
) -> None:
    """Validate runner inputs before any environment is created."""
    if execution_mode not in SUPPORTED_EXECUTION_MODES:
        raise C1SmokePreflightError(
            f"unsupported execution_mode {execution_mode!r}; "
            f"only {sorted(SUPPORTED_EXECUTION_MODES)} are allowed"
        )
    if request_live:
        raise C1SmokePreflightError("live MineRL execution is not supported")
    if allow_live_run_override:
        raise C1SmokePreflightError("live_run_allowed override is forbidden")
    if env_factory is None:
        raise C1SmokePreflightError("env_factory is required")
    if type(env_factory) is not OfflineC1StubEnvFactory:
        raise C1SmokePreflightError(
            "env_factory must be the controlled OfflineC1StubEnvFactory"
        )
    resolved_output = Path(output_dir)
    _validate_new_output_dir(resolved_output)
    resolved_task = task if task is not None else load_frozen_c1_task()
    _validate_task_identity(resolved_task)
    _validate_catalog_live_run_allowed()
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
        raise C1SmokePreflightError(
            "production casting_c1_capabilities incomplete for C1 smoke"
        )
    expected_plan = build_casting_action_plan()
    supplied_plan = tuple(plan) if plan is not None else expected_plan
    if not _plans_equal(supplied_plan, expected_plan):
        raise C1SmokePreflightError(
            "plan must exactly match build_casting_action_plan()"
        )


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _contains_evaluator_token(value: Any) -> bool:
    if isinstance(value, str):
        return any(token in value for token in EVALUATOR_ONLY_TOKENS)
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key in EVALUATOR_ONLY_TOKENS:
                return True
            if _contains_evaluator_token(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_evaluator_token(item) for item in value)
    return False


def _sanitize_public_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return _json_ready(
        {
            key: item
            for key, item in value.items()
            if key not in EVALUATOR_ONLY_TOKENS
        }
    )


def _code_version_snapshot() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def _write_png(path: Path, observation: Observation) -> None:
    frame = observation.frame
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("observation frame must be a non-empty ndarray")
    if frame.dtype != np.uint8 or frame.shape != (360, 640, 3):
        raise ValueError("observation frame must be uint8 with shape (360, 640, 3)")
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)
    if path.stat().st_size == 0:
        raise ValueError(f"refusing to keep empty PNG at {path}")


class _ObservationCapturingBackend:
    """Proxy backend that records the first and latest agent observations."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.initial_observation: Observation | None = None
        self.final_observation: Observation | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def reset(self, task: TaskInstance) -> Mapping[str, Observation]:
        observations = self._inner.reset(task)
        observation = observations[AGENT_ID]
        self.initial_observation = observation
        self.final_observation = observation
        return observations

    def step(self, actions: Mapping[str, MacroAction]) -> BackendStep:
        step = self._inner.step(actions)
        self.final_observation = step.observations[AGENT_ID]
        return step


def _append_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_json_ready(record), ensure_ascii=False))
            handle.write("\n")


def _write_manual_review(
    path: Path,
    *,
    driver_status: str,
    evaluator_outcome: str,
    evaluator_success: bool,
    evidence_complete: bool,
) -> None:
    lines = [
        "# C1 Live MineRL Smoke — Manual Review",
        "",
        "This bundle was produced by the offline stub runner. It does **not**",
        "constitute verification of real MineRL/Minecraft water, lava, or",
        "obsidian behavior.",
        "",
        f"- Driver status: `{driver_status}`",
        f"- Evaluator outcome: `{evaluator_outcome}`",
        f"- Evaluator success: `{evaluator_success}`",
        f"- Evidence complete: `{evidence_complete}`",
        "",
        "Review PNG frames, public events, and evaluator events before any",
        "authorized live MineRL run.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _close_backend_with_retry(backend: Any) -> tuple[str, list[str]]:
    errors: list[str] = []
    for attempt in range(1, 3):
        try:
            backend.close()
            return "closed", errors
        except Exception as error:  # noqa: BLE001 - evidence must record close failures
            errors.append(f"attempt {attempt}: {type(error).__name__}: {error}")
    return "failed", errors


@dataclass(frozen=True)
class CastingC1LiveSmokeResult:
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
    summary: Mapping[str, Any]

    @property
    def overall_success(self) -> bool:
        return (
            self.driver_completed
            and self.evaluator_success
            and self.evidence_complete
            and self.close_status == "closed"
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
            "overall_success": self.overall_success,
            "summary": _json_ready(self.summary),
        }


def run_casting_c1_live_smoke(
    *,
    output_dir: Path | str,
    env_factory: OfflineC1StubEnvFactory | None = None,
    execution_mode: str = EXECUTION_MODE_OFFLINE_STUB,
    task: TaskInstance | None = None,
    plan: tuple[CastingPlanStep, ...] | None = None,
    clock: Callable[[], float] | None = None,
    request_live: bool = False,
    allow_live_run_override: bool | None = None,
) -> CastingC1LiveSmokeResult:
    """Run the offline C1 live-smoke workflow and write evidence."""
    clock_fn = clock if clock is not None else time.time
    resolved_output = Path(output_dir)
    resolved_task = task if task is not None else load_frozen_c1_task()
    resolved_plan = plan if plan is not None else build_casting_action_plan()
    preflight_c1_live_smoke(
        output_dir=resolved_output,
        execution_mode=execution_mode,
        task=resolved_task,
        plan=resolved_plan,
        env_factory=env_factory,
        request_live=request_live,
        allow_live_run_override=allow_live_run_override,
    )

    assert env_factory is not None
    backend = MineRLEnvironmentBackend(
        env_factory=env_factory,
        reset_warmup_steps=0,
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
    capture: _ObservationCapturingBackend | None = None
    staging_dir: Path | None = None
    staging_complete = False
    evidence_complete = False
    output_created_by_runner = False
    driver_steps_executed = 0

    try:
        backend.open()
        capture = _ObservationCapturingBackend(backend)
        driver_result = run_casting_c1_driver(
            capture,
            resolved_task,
            plan=resolved_plan,
        )
        driver_status = driver_result.status
        driver_completed = driver_status == "completed"
        driver_steps_executed = driver_result.steps_executed
        public_events.extend(
            _sanitize_public_mapping(event) for event in driver_result.events
        )

        backend.mark_terminated(reason=TERMINATED_REASON)
        eval_state = backend.get_casting_evaluation_state(FROZEN_TARGET_CELL)
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

        if capture.initial_observation is None or capture.final_observation is None:
            raise RuntimeError("observation capture failed")

        # Stage beside the requested destination. Finalization uses one atomic
        # rename and never overwrites an existing path.
        staging_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{resolved_output.name}.staging-",
                dir=str(resolved_output.parent),
            )
        )
        shutil.copyfile(TASK_PATH, staging_dir / "task_instance.json")
        shutil.copyfile(EXPERIMENT_PATH, staging_dir / "experiment_config.json")
        capability_manifest = backend.capabilities().as_dict()
        (staging_dir / "capability_manifest.json").write_text(
            json.dumps(capability_manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        (staging_dir / "code_version.json").write_text(
            json.dumps(_code_version_snapshot(), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        _write_png(staging_dir / "initial.png", capture.initial_observation)
        _write_png(staging_dir / "final.png", capture.final_observation)
        _append_jsonl(staging_dir / "events.jsonl", public_events)
        _append_jsonl(staging_dir / "evaluator_events.jsonl", evaluator_events)

        summary = {
            "execution_mode": execution_mode,
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
            "timestamp": float(clock_fn()),
            "note": (
                "offline stub runner wiring only; not a live MineRL validation"
            ),
        }
        if _contains_evaluator_token(summary):
            raise RuntimeError("public summary leaked evaluator-only tokens")
        (staging_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_manual_review(
            staging_dir / "manual_review.md",
            driver_status=driver_status,
            evaluator_outcome=evaluator_outcome,
            evaluator_success=evaluator_success,
            evidence_complete=False,
        )

        staging_complete = all(
            (staging_dir / filename).is_file()
            and (staging_dir / filename).stat().st_size > 0
            for filename in REQUIRED_EVIDENCE_FILES
        )
        if not staging_complete:
            raise RuntimeError("staged evidence bundle is incomplete")
    except C1SmokePreflightError:
        raise
    except Exception as error:  # noqa: BLE001 - runner must return structured failure
        if failure_reason is None:
            failure_reason = f"{type(error).__name__}: {error}"
        if not driver_completed and driver_status == "not_started":
            driver_status = "failed"
    finally:
        if getattr(backend, "_opened", False) and close_status != "closed":
            close_status, close_error_list = _close_backend_with_retry(backend)
            close_errors = tuple(close_error_list)
            if close_status != "closed" and failure_reason is None:
                failure_reason = "backend close failed"

        # Finalize a complete staged bundle only after bounded backend close.
        if staging_complete and staging_dir is not None and staging_dir.exists():
            try:
                summary_path = staging_dir / "summary.json"
                final_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                final_summary["close_status"] = close_status
                final_summary["evidence_complete"] = True
                final_summary["failure_reason"] = failure_reason
                final_summary["close_errors"] = list(close_errors)
                summary_path.write_text(
                    json.dumps(
                        final_summary, ensure_ascii=False, indent=2, sort_keys=True
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
                )
                if resolved_output.exists() or resolved_output.is_symlink():
                    raise RuntimeError(
                        "output_dir appeared after preflight; refusing to overwrite"
                    )
                staging_dir.rename(resolved_output)
                output_created_by_runner = True
                evidence_complete = all(
                    (resolved_output / filename).is_file()
                    and (resolved_output / filename).stat().st_size > 0
                    for filename in REQUIRED_EVIDENCE_FILES
                )
            except Exception as error:  # noqa: BLE001
                evidence_complete = False
                if failure_reason is None:
                    failure_reason = f"{type(error).__name__}: {error}"

        if not evidence_complete and not output_created_by_runner:
            # A failed run gets only a fail-closed summary. Never touch a path
            # that appeared after preflight or belonged to the caller already.
            try:
                resolved_output.mkdir(parents=False, exist_ok=False)
                output_created_by_runner = True
                fail_summary = {
                    "execution_mode": execution_mode,
                    "episode_id": resolved_task.task_id,
                    "agent_id": AGENT_ID,
                    "driver_status": driver_status,
                    "driver_completed": driver_completed,
                    "evaluator_outcome": evaluator_outcome,
                    "evaluator_success": evaluator_success,
                    "evidence_complete": False,
                    "close_status": close_status,
                    "close_errors": list(close_errors),
                    "failure_reason": failure_reason,
                    "timestamp": float(clock_fn()),
                }
                (resolved_output / "summary.json").write_text(
                    json.dumps(
                        fail_summary, ensure_ascii=False, indent=2, sort_keys=True
                    )
                    + "\n",
                    encoding="utf-8",
                )
            except Exception as error:  # noqa: BLE001
                if failure_reason is None:
                    failure_reason = f"{type(error).__name__}: {error}"

        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    public_summary = {
        "execution_mode": execution_mode,
        "episode_id": resolved_task.task_id,
        "workflow": resolved_task.workflow,
        "agent_id": AGENT_ID,
        "target_cell": list(FROZEN_TARGET_CELL),
        "driver_status": driver_status,
        "driver_completed": driver_completed,
        "evaluator_outcome": evaluator_outcome,
        "evaluator_success": evaluator_success,
        "evidence_complete": evidence_complete,
        "close_status": close_status,
        "failure_reason": failure_reason,
        "close_errors": list(close_errors),
    }
    return CastingC1LiveSmokeResult(
        execution_mode=execution_mode,
        output_dir=str(resolved_output),
        driver_status=driver_status,
        driver_completed=driver_completed,
        evaluator_success=evaluator_success,
        evaluator_outcome=evaluator_outcome,
        evidence_complete=evidence_complete,
        close_status=close_status,
        failure_reason=failure_reason,
        close_errors=close_errors,
        summary=MappingProxyType(public_summary),
    )


__all__ = [
    "C1ReactiveStubEnv",
    "C1SmokePreflightError",
    "CastingC1LiveSmokeResult",
    "EXECUTION_MODE_OFFLINE_STUB",
    "OfflineC1StubEnvFactory",
    "build_default_c1_plan",
    "build_offline_stub_env_factory",
    "load_frozen_c1_task",
    "preflight_c1_live_smoke",
    "run_casting_c1_live_smoke",
]

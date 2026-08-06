"""R6 Casting-S-C3 frozen-frame evaluator for ``casting_s_c3_fixed``.

R6-C3 frame evaluator + task-origin / truth-grid 坐标锚定。 本模块只
处理 Casting-S-C3 合同：使用原版水、熔岩和方块更新浇筑公开 4×5
full-ring 14 个 cell，不点火、不进入 Nether。

Design contract
---------------

:class:`FrozenFrameEvaluator` 是一个 *纯* 确定性函数，输入是
:class:`FrozenFrameEvaluationState`，输出是
:class:`FrozenFrameEvaluationResult`。 Evaluator ：

* 从不读取 Agent 文本、prompt 或图像；
* 从不 import planner、driver、workflows、agents 或 model-adapter 表面；
* 不调用 backend，不读 wall-clock 时间；
* 对相同输入返回完全相同结果；
* 从不读取 :class:`Observation` 字段或 Agent-visible 公开动作值。

Stability contract
------------------

* :class:`FrozenFrameCellTruth`、 :class:`FrozenFrameInteriorCellTruth`、
  :class:`FrozenFrameEvaluationState` 和
  :class:`FrozenFrameEvaluationResult` 都是 frozen dataclass；
  所有字段在 ``__post_init__`` 中校验，非法 state 在
  evaluator 看到之前就已经 ``ValueError``/``TypeError`` 失败。
* :data:`FRAME_OUTCOMES` 是闭集的 outcome id 集合。
* 优先级由 :func:`_classify_outcome` 编码并由测试套件中的
  ``test_priority_is_stable_for_same_input`` 锁定。
* :meth:`FrozenFrameEvaluationResult.as_dict` 返回 detached、
  JSON-serializable 快照；state 侧的 evidence 不会重新导出。

Outcome set
-----------

闭集 outcome id：

* :data:`OUTCOME_SUCCESS` — 14 个 target cell 全部成为 obsidian，每个
  cell 都有完整 water / lava / transition / relevant-action 因果证据；
  6 个 interior cell 全部在 ``INTERIOR_ALLOWED`` 集合中；
  episode 在 step / time 预算内正常终止。
* :data:`OUTCOME_IN_PROGRESS` — episode 还没有被 terminate。
* :data:`OUTCOME_PARTIAL_COMPLEMENT` — episode 正常 terminate；成功
  的 target cell 构成任意非空真子集，剩余 cell 都保持在可继续浇筑
  的 air / water / lava 状态。完成顺序不影响 partial_completion。
* :data:`OUTCOME_WRONG_BLOCK` — episode 正常 terminate；没有 cell 成功，
  或至少一个未完成 cell 落在 cobblestone / stone 等错误方块上。
* :data:`OUTCOME_TRUTH_MISSING` — 至少一个 target cell 缺少必需证据
  （initial / current block、water / lava truth、transition、relevant
  action steps），或 cells 长度错误，或 14 个 target cell / 6 个
  interior cell 的目标列表与冻结 order 不一致。
* :data:`OUTCOME_STEP_BUDGET_EXCEEDED` —
  ``max(step_id, terminated_step) > max_environment_steps``。
* :data:`OUTCOME_TIME_BUDGET_EXCEEDED` —
  ``current_time_seconds > max_game_time_seconds``。
* :data:`OUTCOME_INVALID_INITIAL_STATE` — 至少一个 target cell 在
  baseline 已经是 obsidian（说明门框不是由本 episode 建造）。
* :data:`OUTCOME_CAUSALITY_MISSING` — 至少一个 target cell 已成为
  obsidian，但 block update 不在 relevant action 之后的有限因果窗口
  内，或 water / lava truth 不为 ``True``，或 transition 的
  after_block 不是 obsidian。
* :data:`OUTCOME_ABNORMAL_TERMINATION` — episode 被一个不在
  :data:`NORMAL_TERMINATION_REASONS` 集合中的 reason 终止。
* :data:`OUTCOME_INTERIOR_BLOCKED` — 至少一个 interior cell 出现
  ``dirt`` / ``bedrock`` / ``grass`` / ``grass_block`` / ``obsidian`` /
  ``other`` / ``missing`` 等阻挡块。``outranks`` 成功/partial/wrong-block
  ，因为门框不合法就不能算 C3 success。

Failure classification 与 R3 / R5 一致：``failure_type == outcome`` 对
每个 terminal failure；``success``、``in_progress``、``truth_missing``、
``interior_blocked``（仍是失败） 使用对应映射。

Coordinate anchor
-----------------

:class:`FrozenFrameOriginAnchor` 是一个 pure 不可变锚定器，把
task-origin-relative 坐标（``scenario_parameters.public_task_spec`` 公开
的 frame plan）转成 truth-grid 相对坐标（与
``obsidianlink.env.portal_spec.PORTAL_GRID_MIN``/``MAX`` 对齐）。

C3 固定实例的 anchor 是 ``default_c3_anchor()``：task-origin 标记对应
grid 原点 (0, 0, 0)，所以公开 14 个 cell（``min_corner=[0,0,1]`` 加上
``width=4``/``height=5``）在数值上落在 ``x=0..3``、``y=0..4``、``z=1``，
被 grid 范围 ``(-3,-1,0)–(3,5,6)`` 完全覆盖。 越界、缺失 origin、
类型错误全部 fail closed。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from obsidianlink.evaluation.casting import (
    CastingFluidTruth,
    CastingTransitionEvidence,
    DEFAULT_CAUSALITY_WINDOW_STEPS,
    MAX_CAUSALITY_WINDOW_STEPS,
    NORMAL_TERMINATION_REASONS,
)


# ----------------------------------------------------------------------
# Frozen public frame plan.
# ----------------------------------------------------------------------

#: 公开 14 个 full-ring cell。 顺序与 C3 合同
#: ``public_task_spec.frame_plan.fixed_offsets`` 严格一致。
CASTING_S_C3_FRAME_CELLS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 1),  # bottom-left corner
    (1, 0, 1),
    (2, 0, 1),
    (3, 0, 1),  # bottom-right corner
    (0, 4, 1),  # top-left corner
    (1, 4, 1),
    (2, 4, 1),
    (3, 4, 1),  # top-right corner
    (0, 1, 1),
    (0, 2, 1),
    (0, 3, 1),
    (3, 1, 1),
    (3, 2, 1),
    (3, 3, 1),
)

#: Four corner cells (subset of :data:`CASTING_S_C3_FRAME_CELLS`).
CASTING_S_C3_CORNER_CELLS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 1),
    (3, 0, 1),
    (0, 4, 1),
    (3, 4, 1),
)

#: Ten non-corner frame cells (Minecraft minimum legal count).
CASTING_S_C3_REQUIRED_CELLS: tuple[tuple[int, int, int], ...] = tuple(
    cell for cell in CASTING_S_C3_FRAME_CELLS
    if cell not in CASTING_S_C3_CORNER_CELLS
)

#: Six interior cells (must be air / nether_portal / fire).
CASTING_S_C3_INTERIOR_CELLS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 1),
    (2, 1, 1),
    (1, 2, 1),
    (2, 2, 1),
    (1, 3, 1),
    (2, 3, 1),
)

#: Total target cells the C3 evaluator requires.
CASTING_S_C3_TARGET_CELL_COUNT: int = 14
CASTING_S_C3_CORNER_CELL_COUNT: int = 4
CASTING_S_C3_REQUIRED_CELL_COUNT: int = 10
CASTING_S_C3_INTERIOR_CELL_COUNT: int = 6
CASTING_S_C3_AGENT_ID: str = "agent_1"
CASTING_S_C3_ACTION_TYPE: str = "use_item"
CASTING_S_C3_ACTION_ITEMS: frozenset[str] = frozenset(
    {"water_bucket", "lava_bucket"}
)

#: Per-cell allowed block for interior cells. Mirrors the
#: ``INTERIOR_ALLOWED`` list in ``frame_geometry`` so the C3 evaluator
#: agrees with the public ``public_task_spec.frame_plan.interior_allowlist``.
INTERIOR_ALLOWED: frozenset[str] = frozenset({"air", "nether_portal", "fire"})

#: Per-cell interior blocked token list (anything that fails the
#: interior allowlist triggers a fail-closed verdict).
INTERIOR_BLOCKERS: frozenset[str] = frozenset(
    {
        "dirt",
        "bedrock",
        "grass",
        "grass_block",
        "obsidian",
        "other",
        "missing",
    }
)


# ----------------------------------------------------------------------
# Outcome constants
# ----------------------------------------------------------------------

OUTCOME_SUCCESS: str = "success"
OUTCOME_IN_PROGRESS: str = "in_progress"
OUTCOME_WRONG_BLOCK: str = "wrong_block"
OUTCOME_TRUTH_MISSING: str = "truth_missing"
OUTCOME_STEP_BUDGET_EXCEEDED: str = "step_budget_exceeded"
OUTCOME_TIME_BUDGET_EXCEEDED: str = "time_budget_exceeded"
OUTCOME_INVALID_INITIAL_STATE: str = "invalid_initial_state"
OUTCOME_CAUSALITY_MISSING: str = "causality_missing"
OUTCOME_ABNORMAL_TERMINATION: str = "abnormal_termination"
OUTCOME_PARTIAL_COMPLEMENT: str = "partial_completion"
OUTCOME_INTERIOR_BLOCKED: str = "interior_blocked"

#: Closed set of outcome ids the evaluator may emit.
FRAME_OUTCOMES: frozenset[str] = frozenset(
    {
        OUTCOME_SUCCESS,
        OUTCOME_IN_PROGRESS,
        OUTCOME_PARTIAL_COMPLEMENT,
        OUTCOME_WRONG_BLOCK,
        OUTCOME_TRUTH_MISSING,
        OUTCOME_STEP_BUDGET_EXCEEDED,
        OUTCOME_TIME_BUDGET_EXCEEDED,
        OUTCOME_INVALID_INITIAL_STATE,
        OUTCOME_CAUSALITY_MISSING,
        OUTCOME_ABNORMAL_TERMINATION,
        OUTCOME_INTERIOR_BLOCKED,
    }
)

#: Terminal failure outcomes (drive ``failure_type``).
_TERMINAL_FAILURE_OUTCOMES: frozenset[str] = frozenset(
    {
        OUTCOME_PARTIAL_COMPLEMENT,
        OUTCOME_WRONG_BLOCK,
        OUTCOME_STEP_BUDGET_EXCEEDED,
        OUTCOME_TIME_BUDGET_EXCEEDED,
        OUTCOME_INVALID_INITIAL_STATE,
        OUTCOME_CAUSALITY_MISSING,
        OUTCOME_ABNORMAL_TERMINATION,
        OUTCOME_INTERIOR_BLOCKED,
    }
)


# ----------------------------------------------------------------------
# Per-cell verdict sentinels (target cells + interior cells)
# ----------------------------------------------------------------------

PER_CELL_SUCCESS: str = "cell_success"
PER_CELL_NOT_EVALUATED: str = "cell_not_evaluated"
PER_CELL_TRUTH_MISSING: str = "cell_truth_missing"
PER_CELL_CAUSALITY_MISSING: str = "cell_causality_missing"
PER_CELL_WRONG_BLOCK: str = "cell_wrong_block"
PER_CELL_INCOMPLETE: str = "cell_incomplete"

PER_CELL_TARGET_VERDICTS: frozenset[str] = frozenset(
    {
        PER_CELL_SUCCESS,
        PER_CELL_NOT_EVALUATED,
        PER_CELL_TRUTH_MISSING,
        PER_CELL_CAUSALITY_MISSING,
        PER_CELL_WRONG_BLOCK,
        PER_CELL_INCOMPLETE,
    }
)

PER_INTERIOR_CELL_ALLOWED: str = "interior_cell_allowed"
PER_INTERIOR_CELL_NOT_EVALUATED: str = "interior_cell_not_evaluated"
PER_INTERIOR_CELL_TRUTH_MISSING: str = "interior_cell_truth_missing"
PER_INTERIOR_CELL_BLOCKED: str = "interior_cell_blocked"

PER_CELL_INTERIOR_VERDICTS: frozenset[str] = frozenset(
    {
        PER_INTERIOR_CELL_ALLOWED,
        PER_INTERIOR_CELL_NOT_EVALUATED,
        PER_INTERIOR_CELL_TRUTH_MISSING,
        PER_INTERIOR_CELL_BLOCKED,
    }
)


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------


def _require_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_non_negative_int(value: int, field_name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_positive_int(value: int, field_name: str) -> int:
    if type(value) is not int or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _require_finite_number(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    return value


def _require_non_negative_number(value: float, field_name: str) -> float:
    value = _require_finite_number(value, field_name)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


def _require_positive_number(value: float, field_name: str) -> float:
    value = _require_finite_number(value, field_name)
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


def _freeze_json_value(value: Any, field_name: str) -> Any:
    """Validate and recursively freeze a JSON-compatible value tree."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must contain only finite numbers")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field_name} keys must be strings")
            frozen[key] = _freeze_json_value(item, f"{field_name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json_value(item, f"{field_name}[]") for item in value
        )
    raise ValueError(
        f"{field_name} must contain only JSON-compatible values, "
        f"got {type(value).__name__}"
    )


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _require_xyz(value: Any, field_name: str) -> tuple[int, int, int]:
    """Validate an ``(x, y, z)`` tuple of strict ints (no bools)."""
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(
            type(coordinate) is not int or isinstance(coordinate, bool)
            for coordinate in value
        )
    ):
        raise ValueError(
            f"{field_name} must be a (x, y, z) tuple of strict integers"
        )
    return value


def _require_interior_block_id(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string when set")
    return value


# ----------------------------------------------------------------------
# Per-cell truth types
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenFrameActionEvidence:
    """One audited, task-scoped action relevant to a target cell."""

    episode_id: str
    step_id: int
    agent_id: str
    action_type: str
    item: str
    target_cell: tuple[int, int, int]

    def __post_init__(self) -> None:
        _require_identifier(self.episode_id, "episode_id")
        _require_non_negative_int(self.step_id, "step_id")
        _require_identifier(self.agent_id, "agent_id")
        if self.action_type != CASTING_S_C3_ACTION_TYPE:
            raise ValueError(
                "action_type must be 'use_item' for casting evidence"
            )
        if self.item not in CASTING_S_C3_ACTION_ITEMS:
            raise ValueError(
                "item must be water_bucket or lava_bucket for casting evidence"
            )
        object.__setattr__(
            self, "target_cell", _require_xyz(self.target_cell, "target_cell")
        )


@dataclass(frozen=True)
class FrozenFrameCellTruth:
    """Typed, per-cell evaluator truth for one C3 target cell.

    Each of the 14 target cells has its own independent evidence. The
    evaluator never lifts another cell's evidence into a different
    cell's causality check. ``relevant_action_steps`` is the per-cell
    tuple of step ids at which ``agent_id`` performed a legal
    ``use_item`` action with ``water_bucket`` or ``lava_bucket``.
    Completed obsidian cells require non-empty typed action evidence
    and an explicit ``transition_action_step`` attribution; unfinished
    cells may carry an empty tuple.
    """

    target_cell: tuple[int, int, int]
    initial_block: str | None
    current_block: str | None
    water_truth: CastingFluidTruth | None
    lava_truth: CastingFluidTruth | None
    transition_evidence: CastingTransitionEvidence | None
    relevant_action_steps: tuple[int, ...]
    action_evidence: tuple[FrozenFrameActionEvidence, ...] = ()
    transition_action_step: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "target_cell", _require_xyz(self.target_cell, "target_cell")
        )
        if self.initial_block is not None:
            if (
                not isinstance(self.initial_block, str)
                or not self.initial_block.strip()
            ):
                raise ValueError(
                    "initial_block must be a non-empty string when set"
                )
        if self.current_block is not None:
            if (
                not isinstance(self.current_block, str)
                or not self.current_block.strip()
            ):
                raise ValueError(
                    "current_block must be a non-empty string when set"
                )
        if self.water_truth is not None and not isinstance(
            self.water_truth, CastingFluidTruth
        ):
            raise ValueError("water_truth must be a CastingFluidTruth or None")
        if self.lava_truth is not None and not isinstance(
            self.lava_truth, CastingFluidTruth
        ):
            raise ValueError("lava_truth must be a CastingFluidTruth or None")
        if (
            self.transition_evidence is not None
            and not isinstance(self.transition_evidence, CastingTransitionEvidence)
        ):
            raise ValueError(
                "transition_evidence must be a CastingTransitionEvidence or None"
            )
        try:
            steps = tuple(self.relevant_action_steps)
        except TypeError as exc:
            raise ValueError(
                "relevant_action_steps must be iterable"
            ) from exc
        for step in steps:
            _require_non_negative_int(step, "relevant_action_steps")
        if len(set(steps)) != len(steps):
            raise ValueError("relevant_action_steps must not contain duplicates")
        object.__setattr__(self, "relevant_action_steps", steps)
        try:
            actions = tuple(self.action_evidence)
        except TypeError as exc:
            raise ValueError("action_evidence must be iterable") from exc
        for index, action in enumerate(actions):
            if not isinstance(action, FrozenFrameActionEvidence):
                raise ValueError(
                    f"action_evidence[{index}] must be FrozenFrameActionEvidence"
                )
            if action.target_cell != self.target_cell:
                raise ValueError(
                    "action_evidence target_cell must match the owning cell"
                )
        action_steps = tuple(action.step_id for action in actions)
        if action_steps != steps:
            raise ValueError(
                "relevant_action_steps must exactly match action_evidence steps"
            )
        object.__setattr__(self, "action_evidence", actions)
        if self.transition_action_step is not None:
            _require_non_negative_int(
                self.transition_action_step, "transition_action_step"
            )


@dataclass(frozen=True)
class FrozenFrameInteriorCellTruth:
    """Typed per-interior-cell truth.

    Interior cells must be ``air`` / ``nether_portal`` / ``fire``; any
    other value (including ``None``) is a fail-closed blocker. The
    interior has no causality requirement; the frame evaluator only
    checks the final block.
    """

    target_cell: tuple[int, int, int]
    current_block: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "target_cell", _require_xyz(self.target_cell, "target_cell")
        )
        _require_interior_block_id(self.current_block, "current_block")


# ----------------------------------------------------------------------
# State
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenFrameEvaluationState:
    """Evaluator-only truth for ``casting_s_c3_fixed``.

    The state is the *only* input the evaluator accepts. All fields
    are validated at construction. ``cells`` must be exactly 14
    :class:`FrozenFrameCellTruth` entries in
    :data:`CASTING_S_C3_FRAME_CELLS` order. ``interior_cells`` must
    be exactly 6 :class:`FrozenFrameInteriorCellTruth` entries in
    :data:`CASTING_S_C3_INTERIOR_CELLS` order. The designated agent
    is captured as ``agent_id`` and may be a tuple of one (the
    current contract only allows ``agent_1``) — the FakeBackend
    additionally cross-checks it against the task's ``agent_ids``.
    """

    episode_id: str
    step_id: int
    cells: tuple[FrozenFrameCellTruth, ...]
    interior_cells: tuple[FrozenFrameInteriorCellTruth, ...]
    agent_id: str = CASTING_S_C3_AGENT_ID
    causality_window_steps: int = DEFAULT_CAUSALITY_WINDOW_STEPS
    episode_terminated: bool = False
    terminated_step: int | None = None
    terminated_reason: str | None = None
    current_time_seconds: float = 0.0
    max_environment_steps: int = 1
    max_game_time_seconds: float = 1.0
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.episode_id, "episode_id")
        _require_non_negative_int(self.step_id, "step_id")
        _require_identifier(self.agent_id, "agent_id")
        if self.agent_id != CASTING_S_C3_AGENT_ID:
            raise ValueError(
                f"agent_id must be {CASTING_S_C3_AGENT_ID!r} for casting_s_c3_fixed"
            )

        # --- target cells -------------------------------------------------
        try:
            cells = tuple(self.cells)
        except TypeError as exc:
            raise ValueError("cells must be iterable") from exc
        if len(cells) != CASTING_S_C3_TARGET_CELL_COUNT:
            raise ValueError(
                "casting_s_c3_fixed requires exactly "
                f"{CASTING_S_C3_TARGET_CELL_COUNT} target cells, got "
                f"{len(cells)}"
            )
        for index, cell in enumerate(cells):
            if not isinstance(cell, FrozenFrameCellTruth):
                raise ValueError(
                    f"cells[{index}] must be a FrozenFrameCellTruth"
                )
            if cell.target_cell != CASTING_S_C3_FRAME_CELLS[index]:
                raise ValueError(
                    "cells must match the frozen casting_s_c3_fixed frame "
                    f"order: index {index} expected "
                    f"{CASTING_S_C3_FRAME_CELLS[index]!r}, got "
                    f"{cell.target_cell!r}"
                )
        # Per-cell disjointness for relevant_action_steps.
        claimed_steps: dict[int, int] = {}
        for cell_index, cell in enumerate(cells):
            if tuple(action.step_id for action in cell.action_evidence) != (
                cell.relevant_action_steps
            ):
                raise ValueError(
                    "relevant_action_steps must exactly match action_evidence steps"
                )
            for action in cell.action_evidence:
                if action.episode_id != self.episode_id:
                    raise ValueError(
                        "action_evidence episode_id must match evaluation state"
                    )
                if action.agent_id != self.agent_id:
                    raise ValueError(
                        "action_evidence agent_id must match evaluation state"
                    )
                if action.target_cell != cell.target_cell:
                    raise ValueError(
                        "action_evidence target_cell must match the owning cell"
                    )
            for step in cell.relevant_action_steps:
                if step > self.step_id:
                    raise ValueError(
                        "cell relevant_action_steps cannot contain a future "
                        f"step (cell {cell_index}, step {step})"
                    )
                previous_owner = claimed_steps.get(step)
                if previous_owner is not None:
                    raise ValueError(
                        "relevant_action_steps must be disjoint across "
                        f"target cells: step {step} is claimed by cells "
                        f"{previous_owner} and {cell_index}"
                    )
                claimed_steps[step] = cell_index
            if (
                cell.transition_action_step is not None
                and cell.transition_action_step > self.step_id
            ):
                raise ValueError(
                    "cell transition_action_step cannot be in the future"
                )
            if (
                cell.water_truth is not None
                and cell.water_truth.evidence_step is not None
                and cell.water_truth.evidence_step > self.step_id
            ):
                raise ValueError(
                    "cell water_truth.evidence_step cannot be in the future"
                )
            if (
                cell.lava_truth is not None
                and cell.lava_truth.evidence_step is not None
                and cell.lava_truth.evidence_step > self.step_id
            ):
                raise ValueError(
                    "cell lava_truth.evidence_step cannot be in the future"
                )
            if (
                cell.transition_evidence is not None
                and cell.transition_evidence.update_step is not None
                and cell.transition_evidence.update_step > self.step_id
            ):
                raise ValueError(
                    "cell transition_evidence.update_step cannot be in the "
                    "future"
                )

        # --- interior cells ----------------------------------------------
        try:
            interior_cells = tuple(self.interior_cells)
        except TypeError as exc:
            raise ValueError("interior_cells must be iterable") from exc
        if len(interior_cells) != CASTING_S_C3_INTERIOR_CELL_COUNT:
            raise ValueError(
                "casting_s_c3_fixed requires exactly "
                f"{CASTING_S_C3_INTERIOR_CELL_COUNT} interior cells, got "
                f"{len(interior_cells)}"
            )
        for index, interior in enumerate(interior_cells):
            if not isinstance(interior, FrozenFrameInteriorCellTruth):
                raise ValueError(
                    f"interior_cells[{index}] must be a "
                    "FrozenFrameInteriorCellTruth"
                )
            if interior.target_cell != CASTING_S_C3_INTERIOR_CELLS[index]:
                raise ValueError(
                    "interior_cells must match the frozen casting_s_c3_fixed "
                    f"interior order: index {index} expected "
                    f"{CASTING_S_C3_INTERIOR_CELLS[index]!r}, got "
                    f"{interior.target_cell!r}"
                )

        # --- causality window --------------------------------------------
        if (
            type(self.causality_window_steps) is not int
            or isinstance(self.causality_window_steps, bool)
            or self.causality_window_steps < 1
            or self.causality_window_steps > MAX_CAUSALITY_WINDOW_STEPS
        ):
            raise ValueError(
                "causality_window_steps must be a positive int "
                f"<= {MAX_CAUSALITY_WINDOW_STEPS}"
            )

        # --- termination -------------------------------------------------
        if type(self.episode_terminated) is not bool:
            raise ValueError("episode_terminated must be a boolean")
        if self.episode_terminated:
            if self.terminated_step is None:
                raise ValueError(
                    "episode_terminated=True requires terminated_step"
                )
            _require_non_negative_int(self.terminated_step, "terminated_step")
            if self.terminated_step > self.step_id:
                raise ValueError("terminated_step cannot be in the future")
            if self.terminated_reason is not None:
                _require_identifier(
                    self.terminated_reason, "terminated_reason"
                )
        elif (
            self.terminated_step is not None
            or self.terminated_reason is not None
        ):
            raise ValueError(
                "terminated_step/terminated_reason require "
                "episode_terminated=True"
            )

        # --- budgets -----------------------------------------------------
        _require_non_negative_number(
            self.current_time_seconds, "current_time_seconds"
        )
        _require_positive_int(
            self.max_environment_steps, "max_environment_steps"
        )
        _require_positive_number(
            self.max_game_time_seconds, "max_game_time_seconds"
        )

        if not isinstance(self.evidence, Mapping):
            raise ValueError("evidence must be a mapping")

        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "interior_cells", interior_cells)
        object.__setattr__(
            self, "evidence", _freeze_json_value(self.evidence, "evidence")
        )


# ----------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenFrameEvaluationResult:
    """Typed, frozen, JSON-serializable C3 evaluator result.

    ``outcome`` ∈ :data:`FRAME_OUTCOMES`. ``success`` is derived
    (``outcome == OUTCOME_SUCCESS``). ``completed_cells`` is the number
    of target cells that ended on obsidian with full causal evidence;
    ``completed_corner_cells`` is the same count restricted to the four
    corner cells. ``interior_blocker_cells`` lists the interior offsets
    that are not in :data:`INTERIOR_ALLOWED`.

    ``per_cell_outcomes`` and ``per_interior_cell_outcomes`` carry the
    per-cell verdict tuples (14 and 6 entries respectively). They use
    the closed per-cell verdict sets (:data:`PER_CELL_TARGET_VERDICTS`
    and :data:`PER_CELL_INTERIOR_VERDICTS`).
    """

    episode_id: str
    step_id: int
    success: bool
    outcome: str
    completed_cells: int
    total_cells: int
    completed_corner_cells: int
    total_corner_cells: int
    completed_interior_cells: int
    total_interior_cells: int
    interior_blocker_cells: tuple[tuple[int, int, int], ...]
    per_cell_outcomes: tuple[str, ...]
    per_interior_cell_outcomes: tuple[str, ...]
    first_failed_cell: int | None
    blocking_conditions: tuple[str, ...]
    evidence: Mapping[str, Any]
    failure_type: str | None = None
    failure_step: int | None = None
    episode_terminated: bool = False
    terminated_step: int | None = None
    terminated_reason: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.episode_id, "episode_id")
        _require_non_negative_int(self.step_id, "step_id")
        if type(self.success) is not bool:
            raise ValueError("success must be a boolean")
        if self.outcome not in FRAME_OUTCOMES:
            raise ValueError(f"unknown outcome: {self.outcome!r}")
        if self.success != (self.outcome == OUTCOME_SUCCESS):
            raise ValueError("success must equal (outcome == 'success')")
        expected_failure_type = (
            self.outcome if self.outcome in _TERMINAL_FAILURE_OUTCOMES else None
        )
        if self.failure_type != expected_failure_type:
            raise ValueError("failure_type must match the terminal outcome")
        if (
            self.failure_type is not None
            and self.failure_type not in _TERMINAL_FAILURE_OUTCOMES
        ):
            raise ValueError(f"unknown failure_type: {self.failure_type!r}")
        _require_non_negative_int(self.completed_cells, "completed_cells")
        _require_positive_int(self.total_cells, "total_cells")
        if self.total_cells != CASTING_S_C3_TARGET_CELL_COUNT:
            raise ValueError("total_cells must equal the frozen C3 cell count")
        if self.completed_cells > self.total_cells:
            raise ValueError("completed_cells cannot exceed total_cells")
        _require_non_negative_int(
            self.completed_corner_cells, "completed_corner_cells"
        )
        _require_positive_int(
            self.total_corner_cells, "total_corner_cells"
        )
        if self.total_corner_cells != CASTING_S_C3_CORNER_CELL_COUNT:
            raise ValueError(
                "total_corner_cells must equal the frozen C3 corner count"
            )
        if self.completed_corner_cells > self.total_corner_cells:
            raise ValueError(
                "completed_corner_cells cannot exceed total_corner_cells"
            )
        _require_non_negative_int(
            self.completed_interior_cells, "completed_interior_cells"
        )
        _require_positive_int(
            self.total_interior_cells, "total_interior_cells"
        )
        if self.total_interior_cells != CASTING_S_C3_INTERIOR_CELL_COUNT:
            raise ValueError(
                "total_interior_cells must equal the frozen C3 interior count"
            )
        if self.completed_interior_cells > self.total_interior_cells:
            raise ValueError(
                "completed_interior_cells cannot exceed total_interior_cells"
            )
        for offset in self.interior_blocker_cells:
            if (
                not isinstance(offset, tuple)
                or len(offset) != 3
                or any(type(v) is not int for v in offset)
            ):
                raise ValueError(
                    "interior_blocker_cells must contain (x, y, z) int tuples"
                )
        if not isinstance(self.per_cell_outcomes, tuple):
            raise ValueError("per_cell_outcomes must be a tuple")
        if len(self.per_cell_outcomes) != self.total_cells:
            raise ValueError(
                "per_cell_outcomes length must equal total_cells"
            )
        for index, verdict in enumerate(self.per_cell_outcomes):
            if verdict not in PER_CELL_TARGET_VERDICTS:
                raise ValueError(
                    f"per_cell_outcomes[{index}] = {verdict!r} is not a "
                    "valid per-cell target verdict"
                )
        if self.completed_cells != self.per_cell_outcomes.count(PER_CELL_SUCCESS):
            raise ValueError(
                "completed_cells must match successful per-cell outcomes"
            )
        expected_corner_completed = sum(
            1
            for index, cell in enumerate(CASTING_S_C3_FRAME_CELLS)
            if cell in CASTING_S_C3_CORNER_CELLS
            and self.per_cell_outcomes[index] == PER_CELL_SUCCESS
        )
        if self.completed_corner_cells != expected_corner_completed:
            raise ValueError(
                "completed_corner_cells must match corner per-cell outcomes"
            )
        if not isinstance(self.per_interior_cell_outcomes, tuple):
            raise ValueError("per_interior_cell_outcomes must be a tuple")
        if len(self.per_interior_cell_outcomes) != self.total_interior_cells:
            raise ValueError(
                "per_interior_cell_outcomes length must equal "
                "total_interior_cells"
            )
        for index, verdict in enumerate(self.per_interior_cell_outcomes):
            if verdict not in PER_CELL_INTERIOR_VERDICTS:
                raise ValueError(
                    f"per_interior_cell_outcomes[{index}] = {verdict!r} is "
                    "not a valid per-interior-cell verdict"
                )
        if self.completed_interior_cells != self.per_interior_cell_outcomes.count(
            PER_INTERIOR_CELL_ALLOWED
        ):
            raise ValueError(
                "completed_interior_cells must match interior outcomes"
            )
        if self.first_failed_cell is not None:
            _require_non_negative_int(
                self.first_failed_cell, "first_failed_cell"
            )
            if self.first_failed_cell >= self.total_cells:
                raise ValueError(
                    "first_failed_cell must be < total_cells when set"
                )
        if not isinstance(self.blocking_conditions, tuple):
            raise ValueError("blocking_conditions must be a tuple")
        for condition in self.blocking_conditions:
            if not isinstance(condition, str) or not condition.strip():
                raise ValueError(
                    "blocking_conditions must be stable strings"
                )
        if self.failure_step is not None:
            _require_non_negative_int(self.failure_step, "failure_step")
        if type(self.episode_terminated) is not bool:
            raise ValueError("episode_terminated must be a boolean")
        if self.episode_terminated:
            if self.terminated_step is None:
                raise ValueError("terminated episode requires terminated_step")
            _require_non_negative_int(self.terminated_step, "terminated_step")
            if self.terminated_step > self.step_id:
                raise ValueError("terminated_step cannot be in the future")
            if self.terminated_reason is not None:
                _require_identifier(self.terminated_reason, "terminated_reason")
        elif self.terminated_step is not None or self.terminated_reason is not None:
            raise ValueError(
                "terminated_step/terminated_reason require episode_terminated=True"
            )
        object.__setattr__(
            self, "evidence", _freeze_json_value(self.evidence, "evidence")
        )
        object.__setattr__(
            self,
            "interior_blocker_cells",
            tuple(
                tuple(offset) for offset in self.interior_blocker_cells
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-serializable snapshot."""
        return {
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "success": self.success,
            "outcome": self.outcome,
            "completed_cells": self.completed_cells,
            "total_cells": self.total_cells,
            "completed_corner_cells": self.completed_corner_cells,
            "total_corner_cells": self.total_corner_cells,
            "completed_interior_cells": self.completed_interior_cells,
            "total_interior_cells": self.total_interior_cells,
            "interior_blocker_cells": [
                list(offset) for offset in self.interior_blocker_cells
            ],
            "per_cell_outcomes": list(self.per_cell_outcomes),
            "per_interior_cell_outcomes": list(
                self.per_interior_cell_outcomes
            ),
            "first_failed_cell": self.first_failed_cell,
            "blocking_conditions": list(self.blocking_conditions),
            "evidence": _thaw_json_value(self.evidence),
            "failure_type": self.failure_type,
            "failure_step": self.failure_step,
            "episode_terminated": self.episode_terminated,
            "terminated_step": self.terminated_step,
            "terminated_reason": self.terminated_reason,
        }


# ----------------------------------------------------------------------
# Per-cell classification helpers
# ----------------------------------------------------------------------


def _per_cell_missing(cell: FrozenFrameCellTruth) -> tuple[str, ...]:
    missing: list[str] = []
    if cell.initial_block is None:
        missing.append("initial_block")
    if cell.current_block is None:
        missing.append("current_block")
    if cell.current_block != "obsidian":
        return tuple(missing)
    if cell.water_truth is None or cell.water_truth.present is None:
        missing.append("water_truth")
    if cell.lava_truth is None or cell.lava_truth.present is None:
        missing.append("lava_truth")
    update = cell.transition_evidence
    if (
        update is None
        or update.update_step is None
        or update.before_block is None
        or update.after_block is None
    ):
        missing.append("transition_evidence")
    if not cell.relevant_action_steps:
        missing.append("relevant_action_steps")
    if not cell.action_evidence:
        missing.append("action_evidence")
    if cell.transition_action_step is None:
        missing.append("transition_action_step")
    return tuple(missing)


def _classify_target_cell(
    cell: FrozenFrameCellTruth,
    *,
    causality_window_steps: int,
    episode_terminated: bool,
    terminated_step: int | None,
) -> tuple[str, int | None, dict[str, Any]]:
    """Return ``(verdict, failure_step, per_cell_evidence)``."""
    evidence: dict[str, Any] = {
        "target_cell": list(cell.target_cell),
        "initial_block": cell.initial_block,
        "current_block": cell.current_block,
        "relevant_action_steps": list(cell.relevant_action_steps),
        "transition_action_step": cell.transition_action_step,
        "action_evidence": [
            {
                "episode_id": action.episode_id,
                "step_id": action.step_id,
                "agent_id": action.agent_id,
                "action_type": action.action_type,
                "item": action.item,
                "target_cell": list(action.target_cell),
            }
            for action in cell.action_evidence
        ],
    }
    if cell.water_truth is not None:
        evidence["water_truth"] = {
            "present": cell.water_truth.present,
            "evidence_step": cell.water_truth.evidence_step,
        }
    if cell.lava_truth is not None:
        evidence["lava_truth"] = {
            "present": cell.lava_truth.present,
            "evidence_step": cell.lava_truth.evidence_step,
        }
    if cell.transition_evidence is not None:
        evidence["transition_evidence"] = {
            "before_block": cell.transition_evidence.before_block,
            "after_block": cell.transition_evidence.after_block,
            "update_step": cell.transition_evidence.update_step,
        }
    missing = _per_cell_missing(cell)
    if missing:
        evidence["missing_truth"] = list(missing)
        return (PER_CELL_TRUTH_MISSING, terminated_step, evidence)
    if not episode_terminated:
        return (PER_CELL_NOT_EVALUATED, None, evidence)
    if cell.current_block != "obsidian":
        evidence["expected_block"] = "obsidian"
        evidence["actual_block"] = cell.current_block
        if cell.current_block in {"air", "water", "lava"}:
            return (PER_CELL_INCOMPLETE, terminated_step, evidence)
        return (PER_CELL_WRONG_BLOCK, terminated_step, evidence)
    update = cell.transition_evidence
    assert update is not None
    if update.before_block == "obsidian":
        evidence["causality_reason"] = "transition_started_as_obsidian"
        return (PER_CELL_CAUSALITY_MISSING, terminated_step, evidence)
    if update.before_block == update.after_block:
        evidence["causality_reason"] = "transition_did_not_change_block"
        return (PER_CELL_CAUSALITY_MISSING, terminated_step, evidence)
    if update.after_block != "obsidian":
        evidence["causality_reason"] = "transition_did_not_produce_obsidian"
        return (PER_CELL_CAUSALITY_MISSING, terminated_step, evidence)
    if cell.water_truth is None or cell.water_truth.present is not True:
        evidence["causality_reason"] = "water_not_present"
        return (PER_CELL_CAUSALITY_MISSING, terminated_step, evidence)
    if cell.lava_truth is None or cell.lava_truth.present is not True:
        evidence["causality_reason"] = "lava_not_present"
        return (PER_CELL_CAUSALITY_MISSING, terminated_step, evidence)
    if update.update_step is None:
        evidence["causality_reason"] = "update_step_missing"
        return (PER_CELL_CAUSALITY_MISSING, terminated_step, evidence)
    if (
        cell.water_truth.evidence_step is None
        or cell.lava_truth.evidence_step is None
    ):
        evidence["causality_reason"] = "fluid_evidence_step_missing"
        return (PER_CELL_CAUSALITY_MISSING, terminated_step, evidence)
    if (
        cell.water_truth.evidence_step > update.update_step
        or cell.lava_truth.evidence_step > update.update_step
    ):
        evidence["causality_reason"] = "fluid_evidence_after_transition"
        return (PER_CELL_CAUSALITY_MISSING, terminated_step, evidence)
    attributed_action = cell.transition_action_step
    if attributed_action not in cell.relevant_action_steps:
        evidence["causality_reason"] = "transition_not_attributed_to_action"
        evidence["update_step"] = update.update_step
        return (PER_CELL_CAUSALITY_MISSING, terminated_step, evidence)
    assert attributed_action is not None
    delta = update.update_step - attributed_action
    evidence["causality_delta_steps"] = delta
    evidence["causality_action_step"] = attributed_action
    if delta < 0 or delta > causality_window_steps:
        evidence["causality_reason"] = "outside_window"
        return (PER_CELL_CAUSALITY_MISSING, terminated_step, evidence)
    return (PER_CELL_SUCCESS, None, evidence)


def _classify_interior_cell(
    interior: FrozenFrameInteriorCellTruth,
    *,
    episode_terminated: bool,
) -> tuple[str, int | None, dict[str, Any]]:
    """Return ``(verdict, failure_step, per_cell_evidence)``."""
    evidence: dict[str, Any] = {
        "target_cell": list(interior.target_cell),
        "current_block": interior.current_block,
    }
    if interior.current_block is None:
        return (
            PER_INTERIOR_CELL_TRUTH_MISSING,
            None,
            evidence,
        )
    if not episode_terminated:
        return (PER_INTERIOR_CELL_NOT_EVALUATED, None, evidence)
    if interior.current_block in INTERIOR_ALLOWED:
        return (PER_INTERIOR_CELL_ALLOWED, None, evidence)
    evidence["actual_block"] = interior.current_block
    return (PER_INTERIOR_CELL_BLOCKED, None, evidence)


# ----------------------------------------------------------------------
# Outcome aggregation
# ----------------------------------------------------------------------


def _aggregate_target_outcome(
    per_cell: tuple[tuple[str, int | None, dict[str, Any]], ...],
    *,
    episode_terminated: bool,
    terminated_step: int | None,
) -> tuple[str, int | None, dict[str, Any]]:
    truth_missing: list[int] = []
    wrong_block: list[int] = []
    incomplete: list[int] = []
    causality_missing: list[int] = []
    success_cells: list[int] = []
    for index, (verdict, _step, _evidence) in enumerate(per_cell):
        if verdict == PER_CELL_TRUTH_MISSING:
            truth_missing.append(index)
        elif verdict == PER_CELL_WRONG_BLOCK:
            wrong_block.append(index)
        elif verdict == PER_CELL_INCOMPLETE:
            incomplete.append(index)
        elif verdict == PER_CELL_CAUSALITY_MISSING:
            causality_missing.append(index)
        elif verdict == PER_CELL_SUCCESS:
            success_cells.append(index)
    extras: dict[str, Any] = {}
    if not episode_terminated:
        if truth_missing:
            extras["missing_cells"] = truth_missing
            return (OUTCOME_TRUTH_MISSING, terminated_step, extras)
        return (OUTCOME_IN_PROGRESS, None, extras)
    if truth_missing:
        extras["missing_cells"] = truth_missing
        return (OUTCOME_TRUTH_MISSING, terminated_step, extras)
    if causality_missing:
        extras["causality_missing_cells"] = causality_missing
        return (OUTCOME_CAUSALITY_MISSING, terminated_step, extras)
    total = len(per_cell)
    completed = len(success_cells)
    if completed == total:
        extras["success_cells"] = success_cells
        return (OUTCOME_SUCCESS, None, extras)
    if (
        0 < completed < total
        and not wrong_block
        and len(incomplete) == total - completed
    ):
        extras["success_cells"] = success_cells
        extras["incomplete_cells"] = incomplete
        extras["non_success_cells"] = total - completed
        return (OUTCOME_PARTIAL_COMPLEMENT, terminated_step, extras)
    extras["wrong_block_cells"] = wrong_block
    extras["incomplete_cells"] = incomplete
    return (OUTCOME_WRONG_BLOCK, terminated_step, extras)


def _aggregate_interior_outcome(
    per_interior: tuple[tuple[str, int | None, dict[str, Any]], ...],
    *,
    episode_terminated: bool,
) -> tuple[str, int | None, dict[str, Any]]:
    blocked: list[tuple[int, int, int]] = []
    truth_missing_interior: list[int] = []
    allowed_count = 0
    for index, (verdict, _step, cell_evidence) in enumerate(per_interior):
        if verdict == PER_INTERIOR_CELL_BLOCKED:
            target_cell = cell_evidence.get("target_cell")
            if isinstance(target_cell, list) and len(target_cell) == 3:
                blocked.append(
                    (
                        int(target_cell[0]),
                        int(target_cell[1]),
                        int(target_cell[2]),
                    )
                )
        elif verdict == PER_INTERIOR_CELL_TRUTH_MISSING:
            truth_missing_interior.append(index)
        elif verdict == PER_INTERIOR_CELL_ALLOWED:
            allowed_count += 1
    if not episode_terminated:
        if truth_missing_interior:
            return (
                OUTCOME_TRUTH_MISSING,
                None,
                {"missing_interior_cells": truth_missing_interior},
            )
        return (OUTCOME_IN_PROGRESS, None, {})
    if truth_missing_interior:
        return (
            OUTCOME_TRUTH_MISSING,
            None,
            {"missing_interior_cells": truth_missing_interior},
        )
    if blocked:
        return (
            OUTCOME_INTERIOR_BLOCKED,
            None,
            {"interior_blocker_cells": [list(b) for b in blocked]},
        )
    return (OUTCOME_SUCCESS, None, {"interior_allowed_count": allowed_count})


# ----------------------------------------------------------------------
# Evaluator
# ----------------------------------------------------------------------


class FrozenFrameEvaluator:
    """Deterministic, offline evaluator for ``casting_s_c3_fixed``.

    The evaluator is a *pure* object: ``evaluate()`` has no side
    effects, reads no global state, and never inspects Agent
    prompts, images, memory, the driver surface, or workflow code.
    Its single input is a :class:`FrozenFrameEvaluationState`; its
    single output is a :class:`FrozenFrameEvaluationResult`.

    Failure classification priority (most specific first)
    ----------------------------------------------------

    1. :data:`OUTCOME_STEP_BUDGET_EXCEEDED` — step budget exceeded
       before any per-cell verdicts could be established.
    2. :data:`OUTCOME_TIME_BUDGET_EXCEEDED` — time budget exceeded.
    3. :data:`OUTCOME_INVALID_INITIAL_STATE` — at least one target
       cell was already obsidian at reset. C3 requires the full
       14-cell ring to be episode-built; a pre-existing obsidian
       cell fails closed.
    4. :data:`OUTCOME_ABNORMAL_TERMINATION` — episode ended for a
       reason outside :data:`NORMAL_TERMINATION_REASONS`.
    5. :data:`OUTCOME_TRUTH_MISSING` — at least one cell is missing
       required evaluator truth, or the cells tuple has the wrong
       length, or the canonical 14-cell / 6-cell order is violated.
    6. :data:`OUTCOME_INTERIOR_BLOCKED` — at least one interior cell
       is not in :data:`INTERIOR_ALLOWED`. This outranks
       ``partial_completion`` / ``wrong_block`` / ``success``
       because a non-empty full ring requires the interior to be
       clear.
    7. :data:`OUTCOME_IN_PROGRESS` — episode not yet terminated
       (truth is otherwise complete).
    8. :data:`OUTCOME_CAUSALITY_MISSING` — at least one target cell
       became obsidian but its block update is outside the finite
       causality window, or its water / lava truth is not ``True``,
       or its transition did not produce obsidian.
    9. :data:`OUTCOME_PARTIAL_COMPLEMENT` — any non-empty proper subset
       of target cells succeeded and all remaining cells are still
       castable (air / water / lava).
    10. :data:`OUTCOME_WRONG_BLOCK` — no target cell succeeded or at
        least one incomplete target contains a blocking wrong block.
    11. :data:`OUTCOME_SUCCESS` — all 14 target cells succeeded and
        the 6 interior cells are all in the allowlist and the
        episode terminated normally within budget.
    """

    def evaluate(
        self, state: FrozenFrameEvaluationState
    ) -> FrozenFrameEvaluationResult:
        (
            outcome,
            failure_step,
            completed_cells,
            completed_corner_cells,
            completed_interior_cells,
            interior_blocker_cells,
            per_cell_outcomes,
            per_interior_cell_outcomes,
            first_failed,
            evidence,
        ) = _classify_outcome(state)
        success = outcome == OUTCOME_SUCCESS
        if outcome in _TERMINAL_FAILURE_OUTCOMES:
            failure_type: str | None = outcome
        else:
            failure_type = None
        return FrozenFrameEvaluationResult(
            episode_id=state.episode_id,
            step_id=state.step_id,
            success=success,
            outcome=outcome,
            completed_cells=completed_cells,
            total_cells=CASTING_S_C3_TARGET_CELL_COUNT,
            completed_corner_cells=completed_corner_cells,
            total_corner_cells=CASTING_S_C3_CORNER_CELL_COUNT,
            completed_interior_cells=completed_interior_cells,
            total_interior_cells=CASTING_S_C3_INTERIOR_CELL_COUNT,
            interior_blocker_cells=interior_blocker_cells,
            per_cell_outcomes=per_cell_outcomes,
            per_interior_cell_outcomes=per_interior_cell_outcomes,
            first_failed_cell=first_failed,
            blocking_conditions=_blocking_conditions(
                outcome, per_cell_outcomes, per_interior_cell_outcomes,
                evidence, interior_blocker_cells,
            ),
            evidence=evidence,
            failure_type=failure_type,
            failure_step=failure_step,
            episode_terminated=state.episode_terminated,
            terminated_step=state.terminated_step,
            terminated_reason=state.terminated_reason,
        )


def _classify_outcome(
    state: FrozenFrameEvaluationState,
) -> tuple[
    str,
    int | None,
    int,
    int,
    int,
    tuple[tuple[int, int, int], ...],
    tuple[str, ...],
    tuple[str, ...],
    int | None,
    Mapping[str, Any],
]:
    total_cells = CASTING_S_C3_TARGET_CELL_COUNT
    total_interior = CASTING_S_C3_INTERIOR_CELL_COUNT
    evidence: dict[str, Any] = {
        "episode_id": state.episode_id,
        "step_id": state.step_id,
        "agent_id": state.agent_id,
        "max_environment_steps": state.max_environment_steps,
        "max_game_time_seconds": state.max_game_time_seconds,
        "current_time_seconds": state.current_time_seconds,
        "causality_window_steps": state.causality_window_steps,
        "target_cells": [list(c) for c in CASTING_S_C3_FRAME_CELLS],
        "interior_cells": [list(c) for c in CASTING_S_C3_INTERIOR_CELLS],
        "required_corner_count": CASTING_S_C3_CORNER_CELL_COUNT,
        "required_full_ring_count": CASTING_S_C3_TARGET_CELL_COUNT,
    }

    # 1. step budget
    observed = [state.step_id]
    if state.terminated_step is not None:
        observed.append(state.terminated_step)
    latest_observed = max(observed)
    if latest_observed > state.max_environment_steps:
        evidence["budget_exceeded_kind"] = "step"
        evidence["budget_exceeded_value"] = latest_observed
        evidence["budget_limit"] = state.max_environment_steps
        return (
            OUTCOME_STEP_BUDGET_EXCEEDED,
            latest_observed,
            0,
            0,
            0,
            (),
            (PER_CELL_NOT_EVALUATED,) * total_cells,
            (PER_INTERIOR_CELL_NOT_EVALUATED,) * total_interior,
            None,
            MappingProxyType(evidence),
        )

    # 2. time budget
    if state.current_time_seconds > state.max_game_time_seconds:
        evidence["budget_exceeded_kind"] = "time"
        evidence["budget_exceeded_value"] = state.current_time_seconds
        evidence["budget_limit"] = state.max_game_time_seconds
        return (
            OUTCOME_TIME_BUDGET_EXCEEDED,
            state.terminated_step
            if state.episode_terminated
            else state.step_id,
            0,
            0,
            0,
            (),
            (PER_CELL_NOT_EVALUATED,) * total_cells,
            (PER_INTERIOR_CELL_NOT_EVALUATED,) * total_interior,
            None,
            MappingProxyType(evidence),
        )

    # 3. invalid initial state: any target cell starts as obsidian
    for index, cell in enumerate(state.cells):
        if cell.initial_block == "obsidian":
            evidence["invalid_initial_cell"] = index
            evidence["invalid_initial_offset"] = list(cell.target_cell)
            return (
                OUTCOME_INVALID_INITIAL_STATE,
                state.step_id,
                0,
                0,
                0,
                (),
                (PER_CELL_NOT_EVALUATED,) * total_cells,
                (PER_INTERIOR_CELL_NOT_EVALUATED,) * total_interior,
                index,
                MappingProxyType(evidence),
            )

    # 4. abnormal_termination (only if terminated)
    if state.episode_terminated:
        if state.terminated_reason is None:
            evidence["missing_reason"] = "terminated_reason"
            return (
                OUTCOME_TRUTH_MISSING,
                state.terminated_step,
                0,
                0,
                0,
                (),
                (PER_CELL_NOT_EVALUATED,) * total_cells,
                (PER_INTERIOR_CELL_NOT_EVALUATED,) * total_interior,
                None,
                MappingProxyType(evidence),
            )
        if (
            state.terminated_reason not in NORMAL_TERMINATION_REASONS
        ):
            evidence["terminated_reason"] = state.terminated_reason
            return (
                OUTCOME_ABNORMAL_TERMINATION,
                state.terminated_step,
                0,
                0,
                0,
                (),
                (PER_CELL_NOT_EVALUATED,) * total_cells,
                (PER_INTERIOR_CELL_NOT_EVALUATED,) * total_interior,
                None,
                MappingProxyType(evidence),
            )

    # 5/7. Per-cell target verdicts + truth_missing aggregation
    per_target: list[tuple[str, int | None, dict[str, Any]]] = []
    for cell in state.cells:
        per_target.append(
            _classify_target_cell(
                cell,
                causality_window_steps=state.causality_window_steps,
                episode_terminated=state.episode_terminated,
                terminated_step=state.terminated_step,
            )
        )
    per_interior: list[tuple[str, int | None, dict[str, Any]]] = []
    for interior in state.interior_cells:
        per_interior.append(
            _classify_interior_cell(
                interior, episode_terminated=state.episode_terminated,
            )
        )

    per_target_outcomes = tuple(verdict for verdict, _step, _ev in per_target)
    per_interior_outcomes = tuple(
        verdict for verdict, _step, _ev in per_interior
    )
    first_failed: int | None = None
    for index, verdict in enumerate(per_target_outcomes):
        if verdict in (
            PER_CELL_TRUTH_MISSING,
            PER_CELL_CAUSALITY_MISSING,
            PER_CELL_WRONG_BLOCK,
            PER_CELL_INCOMPLETE,
        ):
            first_failed = index
            break

    completed_cells = sum(
        1 for verdict, _step, _ev in per_target
        if verdict == PER_CELL_SUCCESS
    )
    completed_corner_cells = sum(
        1
        for cell, (verdict, _step, _ev) in zip(state.cells, per_target)
        if cell.target_cell in CASTING_S_C3_CORNER_CELLS
        and verdict == PER_CELL_SUCCESS
    )
    completed_interior_cells = sum(
        1 for verdict, _step, _ev in per_interior
        if verdict == PER_INTERIOR_CELL_ALLOWED
    )

    # Aggregate target outcome
    target_outcome, target_failure_step, target_extras = (
        _aggregate_target_outcome(
            tuple(per_target),
            episode_terminated=state.episode_terminated,
            terminated_step=state.terminated_step,
        )
    )

    # Aggregate interior outcome
    interior_outcome, interior_failure_step, interior_extras = (
        _aggregate_interior_outcome(
            tuple(per_interior),
            episode_terminated=state.episode_terminated,
        )
    )

    # Interior outcome priority: truth_missing > in_progress >
    # interior_blocked > (success continues the chain).
    interior_blocker_cells: tuple[tuple[int, int, int], ...] = ()
    if interior_outcome == OUTCOME_INTERIOR_BLOCKED:
        raw_blockers = interior_extras.get("interior_blocker_cells", ())
        normalized_blockers: list[tuple[int, int, int]] = []
        for item in raw_blockers:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 3
                and all(type(v) is int for v in item)
            ):
                normalized_blockers.append(
                    (int(item[0]), int(item[1]), int(item[2]))
                )
        interior_blocker_cells = tuple(normalized_blockers)

    aggregate_evidence: dict[str, Any] = dict(evidence)
    aggregate_evidence["per_target_evidence"] = [
        _thaw_json_value(cell_evidence)
        for _verdict, _step, cell_evidence in per_target
    ]
    aggregate_evidence["per_interior_evidence"] = [
        _thaw_json_value(cell_evidence)
        for _verdict, _step, cell_evidence in per_interior
    ]
    aggregate_evidence["target_extras"] = target_extras
    aggregate_evidence["interior_extras"] = interior_extras

    # Priority composition: interior truth_missing outranks everything
    # except the budget / invalid_initial_state / abnormal_termination
    # failures that were already returned above.
    if interior_outcome == OUTCOME_TRUTH_MISSING:
        aggregate_evidence["missing_reason"] = "interior_cell"
        return (
            OUTCOME_TRUTH_MISSING,
            target_failure_step,
            completed_cells,
            completed_corner_cells,
            completed_interior_cells,
            (),
            per_target_outcomes,
            per_interior_outcomes,
            first_failed,
            MappingProxyType(aggregate_evidence),
        )
    if (
        target_outcome == OUTCOME_TRUTH_MISSING
        and interior_outcome == OUTCOME_IN_PROGRESS
    ):
        return (
            OUTCOME_TRUTH_MISSING,
            target_failure_step,
            completed_cells,
            completed_corner_cells,
            completed_interior_cells,
            (),
            per_target_outcomes,
            per_interior_outcomes,
            first_failed,
            MappingProxyType(aggregate_evidence),
        )
    if (
        interior_outcome == OUTCOME_IN_PROGRESS
        and target_outcome == OUTCOME_IN_PROGRESS
    ):
        return (
            OUTCOME_IN_PROGRESS,
            None,
            completed_cells,
            completed_corner_cells,
            completed_interior_cells,
            (),
            per_target_outcomes,
            per_interior_outcomes,
            first_failed,
            MappingProxyType(aggregate_evidence),
        )
    if interior_outcome == OUTCOME_INTERIOR_BLOCKED:
        return (
            OUTCOME_INTERIOR_BLOCKED,
            state.terminated_step if state.episode_terminated else None,
            completed_cells,
            completed_corner_cells,
            completed_interior_cells,
            interior_blocker_cells,
            per_target_outcomes,
            per_interior_outcomes,
            first_failed,
            MappingProxyType(aggregate_evidence),
        )
    # Otherwise the target outcome drives the verdict.
    return (
        target_outcome,
        target_failure_step,
        completed_cells,
        completed_corner_cells,
        completed_interior_cells,
        (),
        per_target_outcomes,
        per_interior_outcomes,
        first_failed,
        MappingProxyType(aggregate_evidence),
    )


def _blocking_conditions(
    outcome: str,
    per_cell_outcomes: tuple[str, ...],
    per_interior_cell_outcomes: tuple[str, ...],
    evidence: Mapping[str, Any],
    interior_blocker_cells: tuple[tuple[int, int, int], ...],
) -> tuple[str, ...]:
    if outcome == OUTCOME_SUCCESS:
        return ()
    if outcome == OUTCOME_IN_PROGRESS:
        return ("episode_not_terminated",)
    if outcome == OUTCOME_STEP_BUDGET_EXCEEDED:
        return ("step_budget_exceeded",)
    if outcome == OUTCOME_TIME_BUDGET_EXCEEDED:
        return ("time_budget_exceeded",)
    if outcome == OUTCOME_INVALID_INITIAL_STATE:
        offset = evidence.get("invalid_initial_offset")
        if isinstance(offset, list) and len(offset) == 3:
            return (
                f"invalid_initial_state:target_offset_{offset[0]}_"
                f"{offset[1]}_{offset[2]}",
            )
        return ("invalid_initial_state",)
    if outcome == OUTCOME_INTERIOR_BLOCKED:
        if not interior_blocker_cells:
            return ("interior_blocked",)
        return tuple(
            f"interior_blocked:offset_{x}_{y}_{z}"
            for (x, y, z) in interior_blocker_cells
        )
    if outcome == OUTCOME_TRUTH_MISSING:
        conditions: list[str] = []
        if evidence.get("missing_reason") == "terminated_reason":
            conditions.append("missing_truth:terminated_reason")
        target_evidences = evidence.get("per_target_evidence", [])
        for index, cell_evidence in enumerate(target_evidences):
            if not isinstance(cell_evidence, Mapping):
                continue
            missing = cell_evidence.get("missing_truth")
            if isinstance(missing, (list, tuple)):
                for name in missing:
                    conditions.append(
                        f"missing_truth:target_{index}.{name}"
                    )
        interior_evidences = evidence.get("per_interior_evidence", [])
        for index, cell_evidence in enumerate(interior_evidences):
            if not isinstance(cell_evidence, Mapping):
                continue
            if cell_evidence.get("current_block") is None:
                conditions.append(
                    f"missing_truth:interior_{index}.current_block"
                )
        if not conditions:
            conditions.append("missing_truth")
        return tuple(conditions)
    if outcome == OUTCOME_CAUSALITY_MISSING:
        target_evidences = evidence.get("per_target_evidence", [])
        conditions = []
        for index, cell_evidence in enumerate(target_evidences):
            if per_cell_outcomes[index] != PER_CELL_CAUSALITY_MISSING:
                continue
            reason = cell_evidence.get("causality_reason")
            if isinstance(reason, str) and reason:
                conditions.append(
                    f"causality_missing:target_{index}:{reason}"
                )
            else:
                conditions.append(f"causality_missing:target_{index}")
        if not conditions:
            return ("causality_missing",)
        return tuple(conditions)
    if outcome == OUTCOME_WRONG_BLOCK:
        target_evidences = evidence.get("per_target_evidence", [])
        conditions = []
        for index, cell_evidence in enumerate(target_evidences):
            if per_cell_outcomes[index] != PER_CELL_WRONG_BLOCK:
                continue
            actual = cell_evidence.get("actual_block")
            if isinstance(actual, str) and actual:
                conditions.append(
                    f"wrong_block:target_{index}:"
                    f"expected_obsidian_got_{actual}"
                )
            else:
                conditions.append(f"wrong_block:target_{index}")
        if not conditions:
            return ("wrong_block",)
        return tuple(conditions)
    if outcome == OUTCOME_PARTIAL_COMPLEMENT:
        success_cells = evidence.get("target_extras", {}).get(
            "success_cells", []
        )
        non_success_cells = evidence.get("target_extras", {}).get(
            "non_success_cells", 0
        )
        if isinstance(success_cells, (list, tuple)) and isinstance(
            non_success_cells, int
        ):
            return (
                f"partial_completion:completed_{len(success_cells)}_of_"
                f"{non_success_cells + len(success_cells)}",
            )
        return ("partial_completion",)
    if outcome == OUTCOME_ABNORMAL_TERMINATION:
        return ("abnormal_termination",)
    return ()


# ----------------------------------------------------------------------
# Coordinate anchor (task-origin <-> grid-origin)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenFrameOriginAnchor:
    """Pure, frozen anchor for task-origin / truth-grid coordinates.

    ``task_origin_in_grid`` is the grid offset where the scene's
    visible task-origin marker sits. ``grid_min``/``grid_max`` are
    the inclusive truth-grid bounds (the same numeric range as
    ``obsidianlink.env.portal_spec.PORTAL_GRID_MIN/MAX``). The
    anchor does not depend on Agent state, prompt, memory, or the
    world; converting an offset is a pure function.

    The default C3 anchor is :func:`default_c3_anchor`, which puts
    the task origin at grid (0, 0, 0) and the full 14-cell ring at
    grid ``x=0..3``/``y=0..4``/``z=1`` — within the existing
    ``(-3,-1,0)–(3,5,6)`` truth grid.
    """

    task_origin_in_grid: tuple[int, int, int]
    grid_min: tuple[int, int, int]
    grid_max: tuple[int, int, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "task_origin_in_grid",
            _require_xyz(self.task_origin_in_grid, "task_origin_in_grid"),
        )
        object.__setattr__(
            self, "grid_min", _require_xyz(self.grid_min, "grid_min")
        )
        object.__setattr__(
            self, "grid_max", _require_xyz(self.grid_max, "grid_max")
        )
        for axis, (low, high) in enumerate(
            zip(self.grid_min, self.grid_max)
        ):
            if low > high:
                raise ValueError(
                    f"grid bound invalid on axis {axis}: "
                    f"min {low} > max {high}"
                )

    def convert(self, task_origin_relative: tuple[int, int, int]) -> tuple[int, int, int]:
        """Return the grid offset for one task-origin-relative offset.

        Fail-closed on type errors or out-of-bounds results.
        """
        relative = _require_xyz(
            task_origin_relative, "task_origin_relative"
        )
        result = tuple(
            int(origin) + int(relative)
            for origin, relative in zip(self.task_origin_in_grid, relative)
        )
        self._validate_within(result, "task_origin_relative")
        return result

    def convert_all(
        self,
        offsets: Sequence[tuple[int, int, int]],
    ) -> tuple[tuple[int, int, int], ...]:
        """Return the grid offset for every input offset.

        Fail-closed on type errors or out-of-bounds results.
        """
        if offsets is None:
            raise TypeError("offsets must be a non-None sequence")
        try:
            sequence = tuple(offsets)
        except TypeError as exc:
            raise TypeError("offsets must be iterable") from exc
        converted: list[tuple[int, int, int]] = []
        for index, offset in enumerate(sequence):
            try:
                converted.append(self.convert(offset))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"offsets[{index}] could not be converted: {exc}"
                ) from exc
        return tuple(converted)

    def _validate_within(
        self,
        grid_offset: tuple[int, int, int],
        field_name: str,
    ) -> None:
        for axis, (value, low, high) in enumerate(
            zip(grid_offset, self.grid_min, self.grid_max)
        ):
            if value < low or value > high:
                raise ValueError(
                    f"{field_name} maps to grid offset "
                    f"{tuple(grid_offset)!r}, which is outside the truth "
                    f"grid bounds [{tuple(self.grid_min)!r}, "
                    f"{tuple(self.grid_max)!r}] on axis {axis}"
                )


def default_c3_anchor() -> FrozenFrameOriginAnchor:
    """Return the frozen C3 anchor.

    The C3 public frame plan (``min_corner=[0,0,1]``, width 4, height
    5) is expressed in ``task_origin_relative`` coordinates with the
    task origin at the scene-visible marker. The frozen anchor puts
    that marker at grid (0, 0, 0); the resulting 14 cells live at
    grid ``x=0..3``/``y=0..4``/``z=1``, which is inside the existing
    ``(-3,-1,0)–(3,5,6)`` truth grid.
    """
    return FrozenFrameOriginAnchor(
        task_origin_in_grid=(0, 0, 0),
        grid_min=(-3, -1, 0),
        grid_max=(3, 5, 6),
    )


__all__ = [
    "CASTING_S_C3_ACTION_ITEMS",
    "CASTING_S_C3_ACTION_TYPE",
    "CASTING_S_C3_AGENT_ID",
    "CASTING_S_C3_CORNER_CELL_COUNT",
    "CASTING_S_C3_CORNER_CELLS",
    "CASTING_S_C3_FRAME_CELLS",
    "CASTING_S_C3_INTERIOR_CELL_COUNT",
    "CASTING_S_C3_INTERIOR_CELLS",
    "CASTING_S_C3_REQUIRED_CELL_COUNT",
    "CASTING_S_C3_REQUIRED_CELLS",
    "CASTING_S_C3_TARGET_CELL_COUNT",
    "FRAME_OUTCOMES",
    "FrozenFrameActionEvidence",
    "FrozenFrameCellTruth",
    "FrozenFrameEvaluationResult",
    "FrozenFrameEvaluationState",
    "FrozenFrameEvaluator",
    "FrozenFrameInteriorCellTruth",
    "FrozenFrameOriginAnchor",
    "INTERIOR_ALLOWED",
    "INTERIOR_BLOCKERS",
    "OUTCOME_ABNORMAL_TERMINATION",
    "OUTCOME_CAUSALITY_MISSING",
    "OUTCOME_IN_PROGRESS",
    "OUTCOME_INVALID_INITIAL_STATE",
    "OUTCOME_INTERIOR_BLOCKED",
    "OUTCOME_PARTIAL_COMPLEMENT",
    "OUTCOME_STEP_BUDGET_EXCEEDED",
    "OUTCOME_SUCCESS",
    "OUTCOME_TIME_BUDGET_EXCEEDED",
    "OUTCOME_TRUTH_MISSING",
    "OUTCOME_WRONG_BLOCK",
    "PER_CELL_CAUSALITY_MISSING",
    "PER_CELL_INTERIOR_VERDICTS",
    "PER_CELL_INCOMPLETE",
    "PER_CELL_NOT_EVALUATED",
    "PER_CELL_SUCCESS",
    "PER_CELL_TARGET_VERDICTS",
    "PER_CELL_TRUTH_MISSING",
    "PER_CELL_WRONG_BLOCK",
    "PER_INTERIOR_CELL_ALLOWED",
    "PER_INTERIOR_CELL_BLOCKED",
    "PER_INTERIOR_CELL_NOT_EVALUATED",
    "PER_INTERIOR_CELL_TRUTH_MISSING",
    "default_c3_anchor",
]

"""R6 Casting-S-C4 ignition evaluator for ``casting_s_c4_fixed``.

R6-C4 ignition evaluator + typed frame-identity + 4-step 因果窗口。
本模块只处理 Casting-S-C4 合同：在 C3 公开 4×5 full-ring 14-cell
浇筑归因成功的基础上，要求 ``agent_1`` 在公开唯一计分目标
``[1, 1, 1]`` 执行 ``use_item(flint_and_steel)``，并且本 episode
锁存的 frame 内部在 4-step 因果窗口内出现 ``nether_portal``；
激活必须严格绑定同一个 :class:`FrozenFrameIdentity`，
且 activation.agent_id 必须与 ignition action 一致。

Design contract
---------------

:class:`FrozenIgnitionEvaluator` 是一个 *纯* 确定性函数，输入是
:class:`FrozenIgnitionEvaluationState`，输出是
:class:`FrozenIgnitionEvaluationResult`。 Evaluator ：

* 从不读取 Agent 文本、prompt 或图像；
* 从不 import planner、driver、workflows、agents 或 model-adapter 表面；
* 不调用 backend，不读 wall-clock 时间；
* 对相同输入返回完全相同结果；
* 从不读取 :class:`Observation` 字段或 Agent-visible 公开动作值；
* 对 C3 14-cell 浇筑的验证**复用**
  :class:`obsidianlink.evaluation.casting_frame_evaluator.FrozenFrameEvaluator`
  的结果，但额外加入精确目标、动作归因、4-step 因果窗口、
  typed frame-identity 绑定与 agent-id 归因一致性检查——
  不会因为复用 C3 而削弱这些 C4-specific 规则。

Layered construction contract
-----------------------------

每一类证据只在该层构造期做**结构性**校验，**语义**校验完全由
evaluator 在分类阶段统一处理：

* :class:`IgnitionActionEvidence` 构造期：episode_id / step_id /
  agent_id / action_type / item / target_cell 的类型、xyz
  整数元组、bool / 负数 / None / 非 mapping / 空字符串等
  malformed 值 fail closed；但 ``agent_id`` 是不是 ``agent_1``、
  ``action_type`` 是不是 ``use_item``、``item`` 是不是
  ``flint_and_steel``、``target_cell`` 是不是 ``(1, 1, 1)``
  等语义判断完全交给 evaluator 产出
  :data:`OUTCOME_WRONG_IGNITION_AGENT` /
  :data:`OUTCOME_WRONG_IGNITION_ACTION` /
  :data:`OUTCOME_WRONG_IGNITION_ITEM` /
  :data:`OUTCOME_WRONG_IGNITION_TARGET` 等 outcome。
* :class:`PortalActivationEvidence` 构造期：所有身份字段、
  update_step 整数范围、nether_portal_offset xyz 整数元组、
  ``latched_frame_identity`` 必须是已构造的
  :class:`FrozenFrameIdentity` 实例；但 nether_portal_offset
  是否在 frame interior、agent_id 是不是 ``agent_1``、
  frame identity 几何是否匹配 C3 固定门框等语义判断交给
  evaluator 产出 :data:`OUTCOME_EXTERNAL_ACTIVATION` /
  :data:`OUTCOME_FRAME_IDENTITY_MISMATCH` 等 outcome。
* :class:`FrozenFrameIdentity` 构造期：orientation / min_corner /
  max_corner / width / height / target_offsets / interior_offsets
  / activation_offsets / required_corner_count /
  required_full_ring_count / episode_id / step_id / agent_id
  的类型、xyz 元组、范围、bool 拒绝等结构性校验；
  几何是否与 C3 固定 frame plan 完全一致交给 evaluator 产出
  :data:`OUTCOME_FRAME_IDENTITY_MISMATCH`。

Stability contract
------------------

* :class:`FrozenFrameIdentity` /
  :class:`IgnitionActionEvidence` /
  :class:`PortalActivationEvidence` /
  :class:`FrozenIgnitionEvaluationState` /
  :class:`FrozenIgnitionEvaluationResult` 都是 frozen dataclass；
  所有字段在 ``__post_init__`` 中校验，非法 state 在
  evaluator 看到之前就已经 ``ValueError``/``TypeError`` 失败。
* :data:`IGNITION_OUTCOMES` 是闭集的 outcome id 集合。
* 优先级由 :func:`_classify_outcome` 编码并由测试套件中的
  ``test_priority_is_stable_for_same_input`` 锁定。
* :meth:`FrozenIgnitionEvaluationResult.as_dict` 返回 detached、
  JSON-serializable 快照；state 侧的 evidence 不会重新导出。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from obsidianlink.evaluation.casting import (
    DEFAULT_CAUSALITY_WINDOW_STEPS,
    MAX_CAUSALITY_WINDOW_STEPS,
    NORMAL_TERMINATION_REASONS,
)
from obsidianlink.evaluation.casting_frame_evaluator import (
    CASTING_S_C3_ACTION_ITEMS,
    CASTING_S_C3_CORNER_CELL_COUNT,
    CASTING_S_C3_CORNER_CELLS,
    CASTING_S_C3_FRAME_CELLS,
    CASTING_S_C3_INTERIOR_CELL_COUNT,
    CASTING_S_C3_INTERIOR_CELLS,
    CASTING_S_C3_REQUIRED_CELL_COUNT,
    CASTING_S_C3_REQUIRED_CELLS,
    CASTING_S_C3_TARGET_CELL_COUNT,
    FrozenFrameEvaluationState,
    FrozenFrameEvaluator,
    OUTCOME_SUCCESS as FRAME_OUTCOME_SUCCESS,
    OUTCOME_IN_PROGRESS as FRAME_OUTCOME_IN_PROGRESS,
)


# ----------------------------------------------------------------------
# Frozen public ignition plan
# ----------------------------------------------------------------------

#: The single Agent authorized to ignite the C4 portal. Mirrors
#: ``casting_s_c4_fixed`` instance ``agent_ids = ['agent_1']``.
CASTING_S_C4_AGENT_ID: str = "agent_1"

#: The single closed item id that counts as the C4 ignition tool.
CASTING_S_C4_IGNITION_ITEM: str = "flint_and_steel"

#: The single closed action type that counts as the C4 ignition.
CASTING_S_C4_IGNITION_ACTION_TYPE: str = "use_item"

#: The single public exact ignition target. Mirrors
#: ``public_task_spec.ignition_plan.target_offset = [1, 1, 1]``.
CASTING_S_C4_PUBLIC_IGNITION_TARGET: tuple[int, int, int] = (1, 1, 1)

#: Six interior cells of the public 4×5 full-ring frame plan
#: (``min_corner=[0,0,1]``, width 4, height 5).
CASTING_S_C4_FRAME_INTERIOR_CELLS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 1),
    (2, 1, 1),
    (1, 2, 1),
    (2, 2, 1),
    (1, 3, 1),
    (2, 3, 1),
)

#: Closed set of cells an activation nether_portal may land on.
#: The activation offset must be a member of this set or the
#: evaluator fails closed with :data:`OUTCOME_EXTERNAL_ACTIVATION`.
CASTING_S_C4_FRAME_INTERIOR_SET: frozenset[tuple[int, int, int]] = frozenset(
    CASTING_S_C4_FRAME_INTERIOR_CELLS
)

#: Default C4 causality window (in environment steps) between the
#: ignition ``step_id`` and the activation ``update_step``. The
#: window is inclusive on both ends: ``delta ∈ [0, 4]`` is within
#: window, ``delta > 4`` is outside. Re-uses the R3 / R5 / C3
#: default of 4 so the C4 contract is consistent with the rest
#: of the family.
CASTING_S_C4_CAUSALITY_WINDOW_STEPS: int = DEFAULT_CAUSALITY_WINDOW_STEPS

#: The single closed orientation for the C4 fixed frame plan.
CASTING_S_C4_FRAME_ORIENTATION: str = "plane_z"

#: The single closed ``min_corner`` (task-origin-relative) for the
#: C4 fixed 4×5 full-ring frame plan.
CASTING_S_C4_FRAME_MIN_CORNER: tuple[int, int, int] = (0, 0, 1)

#: The single closed ``max_corner`` for the C4 fixed 4×5
#: full-ring frame plan.
CASTING_S_C4_FRAME_MAX_CORNER: tuple[int, int, int] = (3, 4, 1)

#: The single closed ``width`` for the C4 fixed 4×5 full-ring
#: frame plan.
CASTING_S_C4_FRAME_WIDTH: int = 4

#: The single closed ``height`` for the C4 fixed 4×5 full-ring
#: frame plan.
CASTING_S_C4_FRAME_HEIGHT: int = 5


# ----------------------------------------------------------------------
# Outcome constants
# ----------------------------------------------------------------------

OUTCOME_SUCCESS: str = "success"
OUTCOME_IN_PROGRESS: str = "in_progress"
OUTCOME_FRAME_NOT_BUILT: str = "frame_not_built"
OUTCOME_IGNITION_ACTION_MISSING: str = "ignition_action_missing"
OUTCOME_WRONG_IGNITION_AGENT: str = "wrong_ignition_agent"
OUTCOME_WRONG_IGNITION_ACTION: str = "wrong_ignition_action"
OUTCOME_WRONG_IGNITION_ITEM: str = "wrong_ignition_item"
OUTCOME_WRONG_IGNITION_TARGET: str = "wrong_ignition_target"
OUTCOME_ACTIVATION_MISSING: str = "portal_activation_missing"
OUTCOME_ACTIVATION_BEFORE_IGNITION: str = "activation_before_ignition"
OUTCOME_ACTIVATION_OUTSIDE_WINDOW: str = "activation_outside_window"
OUTCOME_EXTERNAL_ACTIVATION: str = "external_activation"
OUTCOME_FRAME_IDENTITY_MISSING: str = "frame_identity_missing"
OUTCOME_FRAME_IDENTITY_MISMATCH: str = "frame_identity_mismatch"
OUTCOME_IGNITION_AGENT_MISMATCH: str = "ignition_agent_mismatch"
OUTCOME_IGNITION_ACTION_MISMATCH: str = "ignition_action_mismatch"
OUTCOME_IGNITION_ITEM_MISMATCH: str = "ignition_item_mismatch"
OUTCOME_IGNITION_TARGET_MISMATCH: str = "ignition_target_mismatch"
OUTCOME_TRUTH_MISSING: str = "truth_missing"
OUTCOME_STEP_BUDGET_EXCEEDED: str = "step_budget_exceeded"
OUTCOME_TIME_BUDGET_EXCEEDED: str = "time_budget_exceeded"
OUTCOME_INVALID_INITIAL_STATE: str = "invalid_initial_state"
OUTCOME_ABNORMAL_TERMINATION: str = "abnormal_termination"

#: Closed set of outcome ids the C4 evaluator may emit.
IGNITION_OUTCOMES: frozenset[str] = frozenset(
    {
        OUTCOME_SUCCESS,
        OUTCOME_IN_PROGRESS,
        OUTCOME_FRAME_NOT_BUILT,
        OUTCOME_IGNITION_ACTION_MISSING,
        OUTCOME_WRONG_IGNITION_AGENT,
        OUTCOME_WRONG_IGNITION_ACTION,
        OUTCOME_WRONG_IGNITION_ITEM,
        OUTCOME_WRONG_IGNITION_TARGET,
        OUTCOME_ACTIVATION_MISSING,
        OUTCOME_ACTIVATION_BEFORE_IGNITION,
        OUTCOME_ACTIVATION_OUTSIDE_WINDOW,
        OUTCOME_EXTERNAL_ACTIVATION,
        OUTCOME_FRAME_IDENTITY_MISSING,
        OUTCOME_FRAME_IDENTITY_MISMATCH,
        OUTCOME_TRUTH_MISSING,
        OUTCOME_STEP_BUDGET_EXCEEDED,
        OUTCOME_TIME_BUDGET_EXCEEDED,
        OUTCOME_INVALID_INITIAL_STATE,
        OUTCOME_ABNORMAL_TERMINATION,
    }
)

#: Terminal failure outcomes (drive ``failure_type``).
_TERMINAL_FAILURE_OUTCOMES: frozenset[str] = frozenset(
    {
        OUTCOME_FRAME_NOT_BUILT,
        OUTCOME_IGNITION_ACTION_MISSING,
        OUTCOME_WRONG_IGNITION_AGENT,
        OUTCOME_WRONG_IGNITION_ACTION,
        OUTCOME_WRONG_IGNITION_ITEM,
        OUTCOME_WRONG_IGNITION_TARGET,
        OUTCOME_ACTIVATION_MISSING,
        OUTCOME_ACTIVATION_BEFORE_IGNITION,
        OUTCOME_ACTIVATION_OUTSIDE_WINDOW,
        OUTCOME_EXTERNAL_ACTIVATION,
        OUTCOME_FRAME_IDENTITY_MISSING,
        OUTCOME_FRAME_IDENTITY_MISMATCH,
        OUTCOME_TRUTH_MISSING,
        OUTCOME_STEP_BUDGET_EXCEEDED,
        OUTCOME_TIME_BUDGET_EXCEEDED,
        OUTCOME_INVALID_INITIAL_STATE,
        OUTCOME_ABNORMAL_TERMINATION,
    }
)


# ----------------------------------------------------------------------
# Per-event verdict sentinels
# ----------------------------------------------------------------------

IGNITION_VERDICT_OBSERVED: str = "ignition_observed"
IGNITION_VERDICT_MISSING: str = "ignition_missing"
ACTIVATION_VERDICT_OBSERVED: str = "activation_observed"
ACTIVATION_VERDICT_MISSING: str = "activation_missing"
IGNITION_AGENT_VERDICT_OK: str = "ignition_agent_ok"
IGNITION_AGENT_VERDICT_WRONG: str = "ignition_agent_wrong"
IGNITION_ACTION_VERDICT_OK: str = "ignition_action_ok"
IGNITION_ACTION_VERDICT_WRONG: str = "ignition_action_wrong"
IGNITION_ITEM_VERDICT_OK: str = "ignition_item_ok"
IGNITION_ITEM_VERDICT_WRONG: str = "ignition_item_wrong"
IGNITION_TARGET_VERDICT_OK: str = "ignition_target_ok"
IGNITION_TARGET_VERDICT_WRONG: str = "ignition_target_wrong"
ACTIVATION_WINDOW_VERDICT_OK: str = "activation_window_ok"
ACTIVATION_WINDOW_VERDICT_BEFORE: str = "activation_before_ignition"
ACTIVATION_WINDOW_VERDICT_OUTSIDE: str = "activation_outside_window"
ACTIVATION_OFFSET_VERDICT_INTERNAL: str = "activation_offset_internal"
ACTIVATION_OFFSET_VERDICT_EXTERNAL: str = "activation_offset_external"
ACTIVATION_AGENT_VERDICT_OK: str = "activation_agent_ok"
ACTIVATION_AGENT_VERDICT_WRONG: str = "activation_agent_wrong"
FRAME_IDENTITY_VERDICT_MATCH: str = "frame_identity_match"
FRAME_IDENTITY_VERDICT_MISSING: str = "frame_identity_missing"
FRAME_IDENTITY_VERDICT_MISMATCH: str = "frame_identity_mismatch"
FRAME_IDENTITY_VERDICT_GEOMETRY_MISMATCH: str = "frame_identity_geometry_mismatch"

IGNITION_VERDICTS: frozenset[str] = frozenset(
    {IGNITION_VERDICT_OBSERVED, IGNITION_VERDICT_MISSING}
)
ACTIVATION_VERDICTS: frozenset[str] = frozenset(
    {ACTIVATION_VERDICT_OBSERVED, ACTIVATION_VERDICT_MISSING}
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


def _require_int(value: int, field_name: str) -> int:
    """Validate a strict int (no bool, no None). May be negative."""
    if type(value) is not int or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a strict integer")
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


def _require_xyz_set(
    value: Any, field_name: str
) -> frozenset[tuple[int, int, int]]:
    """Validate a set of ``(x, y, z)`` tuples of strict ints."""
    if not isinstance(value, (set, frozenset, list, tuple)):
        raise ValueError(f"{field_name} must be a set-like of xyz tuples")
    result: set[tuple[int, int, int]] = set()
    for index, item in enumerate(value):
        result.add(_require_xyz(item, f"{field_name}[{index}]"))
    return frozenset(result)


def _require_xyz_list(
    value: Any, field_name: str
) -> tuple[tuple[int, int, int], ...]:
    """Validate an ordered list of ``(x, y, z)`` tuples of strict ints."""
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a list of xyz tuples")
    return tuple(
        _require_xyz(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )


def _freeze_xyz_list(
    value: Sequence[tuple[int, int, int]],
) -> tuple[tuple[int, int, int], ...]:
    return tuple(tuple(cell) for cell in value)


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


# ----------------------------------------------------------------------
# Typed frame identity (C4 episode-built frame contract)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenFrameIdentity:
    """Typed, immutable, JSON-serializable C4 frame identity.

    A frame identity is the *authoritative* description of the
    episode-built frame. It is a strict superset of the C3 public
    frame plan and is bound to the same episode / step / agent
    identity. Two arbitrary mappings cannot pass as a valid
    identity: the construction-time structural checks reject
    missing or malformed fields, and the C4 evaluator additionally
    enforces that all geometry fields agree with the C3 frozen
    contract (``CASTING_S_C3_FRAME_CELLS``,
    ``CASTING_S_C3_INTERIOR_CELLS``, ``CASTING_S_C3_CORNER_CELLS``)
    and with the C4 public fixed values
    (``CASTING_S_C4_FRAME_ORIENTATION``,
    ``CASTING_S_C4_FRAME_MIN_CORNER``,
    ``CASTING_S_C4_FRAME_MAX_CORNER``,
    ``CASTING_S_C4_FRAME_WIDTH``, ``CASTING_S_C4_FRAME_HEIGHT``).

    Identity binding
    ----------------

    A successful C4 evaluation requires the ``latched_frame_identity``
    on the wrapping :class:`FrozenIgnitionEvaluationState` and the
    ``latched_frame_identity`` on the
    :class:`PortalActivationEvidence` to be **structurally
    identical** :class:`FrozenFrameIdentity` instances (deep
    equality of every field). Mismatched geometry or
    mismatched identity / step / agent metadata fails closed with
    :data:`OUTCOME_FRAME_IDENTITY_MISMATCH`.

    Episode / step / agent
    ----------------------

    ``episode_id`` / ``step_id`` / ``agent_id`` are mandatory and
    identify which episode + which observation step latched the
    frame identity. The C4 evaluator cross-checks that the wrapping
    state's :class:`FrozenFrameEvaluationState.episode_id` and
    ``agent_id`` agree with the identity, and that the
    :class:`PortalActivationEvidence.episode_id` /
    ``update_step`` agree with the identity.
    """

    orientation: str
    min_corner: tuple[int, int, int]
    max_corner: tuple[int, int, int]
    width: int
    height: int
    target_offsets: tuple[tuple[int, int, int], ...]
    interior_offsets: tuple[tuple[int, int, int], ...]
    required_corner_count: int
    required_full_ring_count: int
    activation_offsets: tuple[tuple[int, int, int], ...]
    episode_id: str
    step_id: int
    agent_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.orientation, str) or not self.orientation.strip():
            raise ValueError("orientation must be a non-empty string")
        object.__setattr__(
            self, "min_corner", _require_xyz(self.min_corner, "min_corner")
        )
        object.__setattr__(
            self, "max_corner", _require_xyz(self.max_corner, "max_corner")
        )
        for axis, (low, high) in enumerate(
            zip(self.min_corner, self.max_corner)
        ):
            if low > high:
                raise ValueError(
                    f"min_corner > max_corner on axis {axis}: "
                    f"{low} > {high}"
                )
        _require_positive_int(self.width, "width")
        _require_positive_int(self.height, "height")
        object.__setattr__(
            self,
            "target_offsets",
            _require_xyz_list(self.target_offsets, "target_offsets"),
        )
        object.__setattr__(
            self,
            "interior_offsets",
            _require_xyz_list(self.interior_offsets, "interior_offsets"),
        )
        _require_non_negative_int(
            self.required_corner_count, "required_corner_count"
        )
        _require_non_negative_int(
            self.required_full_ring_count, "required_full_ring_count"
        )
        object.__setattr__(
            self,
            "activation_offsets",
            _require_xyz_list(
                self.activation_offsets, "activation_offsets"
            ),
        )
        _require_identifier(self.episode_id, "episode_id")
        _require_non_negative_int(self.step_id, "step_id")
        _require_identifier(self.agent_id, "agent_id")

    def as_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-serializable snapshot."""
        return {
            "orientation": self.orientation,
            "min_corner": list(self.min_corner),
            "max_corner": list(self.max_corner),
            "width": self.width,
            "height": self.height,
            "target_offsets": [list(c) for c in self.target_offsets],
            "interior_offsets": [list(c) for c in self.interior_offsets],
            "required_corner_count": self.required_corner_count,
            "required_full_ring_count": self.required_full_ring_count,
            "activation_offsets": [
                list(c) for c in self.activation_offsets
            ],
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "agent_id": self.agent_id,
        }


def build_c4_c3_frame_identity(
    *,
    episode_id: str,
    step_id: int,
    agent_id: str = CASTING_S_C4_AGENT_ID,
    activation_offsets: tuple[tuple[int, int, int], ...] = (),
) -> FrozenFrameIdentity:
    """Build the frozen C3-based C4 frame identity.

    This is the **only** helper allowed to construct a
    :class:`FrozenFrameIdentity` that the C4 evaluator will treat
    as the canonical C4 episode-built identity. The geometry
    comes directly from the C3 frozen frame plan
    (:data:`CASTING_S_C3_FRAME_CELLS`,
    :data:`CASTING_S_C3_INTERIOR_CELLS`,
    :data:`CASTING_S_C3_CORNER_CELL_COUNT`,
    :data:`CASTING_S_C3_REQUIRED_CELL_COUNT`,
    :data:`CASTING_S_C3_TARGET_CELL_COUNT`) and the C4 public
    fixed orientation / corners / width / height. Any other
    geometry (or arbitrary mapping) is rejected by the C4
    evaluator at runtime with
    :data:`OUTCOME_FRAME_IDENTITY_MISMATCH` /
    :data:`OUTCOME_FRAME_IDENTITY_GEOMETRY_MISMATCH` /
    :data:`OUTCOME_FRAME_IDENTITY_MISSING`.
    """
    return FrozenFrameIdentity(
        orientation=CASTING_S_C4_FRAME_ORIENTATION,
        min_corner=CASTING_S_C4_FRAME_MIN_CORNER,
        max_corner=CASTING_S_C4_FRAME_MAX_CORNER,
        width=CASTING_S_C4_FRAME_WIDTH,
        height=CASTING_S_C4_FRAME_HEIGHT,
        target_offsets=_freeze_xyz_list(CASTING_S_C3_FRAME_CELLS),
        interior_offsets=_freeze_xyz_list(CASTING_S_C3_INTERIOR_CELLS),
        required_corner_count=CASTING_S_C3_CORNER_CELL_COUNT,
        required_full_ring_count=CASTING_S_C3_TARGET_CELL_COUNT,
        activation_offsets=_freeze_xyz_list(
            activation_offsets
            if activation_offsets
            else CASTING_S_C4_PUBLIC_IGNITION_TARGET
            and (CASTING_S_C4_PUBLIC_IGNITION_TARGET,)
        ),
        episode_id=episode_id,
        step_id=step_id,
        agent_id=agent_id,
    )


def _frame_identities_are_equal(
    left: FrozenFrameIdentity, right: FrozenFrameIdentity
) -> bool:
    """Strict structural equality between two frozen identities."""
    return left.as_dict() == right.as_dict()


# ----------------------------------------------------------------------
# Evidence types
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class IgnitionActionEvidence:
    """One audited ignition action.

    Structural checks (constructor fail-closed):
    ``episode_id`` is a non-empty identifier; ``step_id`` is a
    non-negative strict int; ``agent_id`` is a non-empty
    identifier; ``action_type`` is a non-empty identifier;
    ``item`` is a non-empty identifier; ``target_cell`` is a
    3-tuple of strict ints.

    Semantic checks (evaluator):
    * ``agent_id`` must equal :data:`CASTING_S_C4_AGENT_ID` →
      :data:`OUTCOME_WRONG_IGNITION_AGENT`;
    * ``action_type`` must equal
      :data:`CASTING_S_C4_IGNITION_ACTION_TYPE` →
      :data:`OUTCOME_WRONG_IGNITION_ACTION`;
    * ``item`` must equal :data:`CASTING_S_C4_IGNITION_ITEM` →
      :data:`OUTCOME_WRONG_IGNITION_ITEM`;
    * ``target_cell`` must equal
      :data:`CASTING_S_C4_PUBLIC_IGNITION_TARGET` →
      :data:`OUTCOME_WRONG_IGNITION_TARGET`.

    Construction is therefore deliberately *lenient* on the
    ``action_type`` / ``item`` / ``target_cell`` whitelist: the
    evaluator is the single source of truth for the C4 ignition
    whitelist, and the four wrong_* outcomes must remain
    reachable through the public construction API.
    """

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
        _require_identifier(self.action_type, "action_type")
        _require_identifier(self.item, "item")
        object.__setattr__(
            self,
            "target_cell",
            _require_xyz(self.target_cell, "target_cell"),
        )


@dataclass(frozen=True)
class PortalActivationEvidence:
    """One audited ``nether_portal`` block appearance.

    Structural checks (constructor fail-closed):
    ``episode_id`` non-empty identifier; ``update_step`` non-negative
    strict int; ``agent_id`` non-empty identifier (and required,
    not optional — the C4 contract binds the activation to the
    same agent who performed the ignition); ``nether_portal_offset``
    3-tuple of strict ints; ``latched_frame_identity`` must be an
    already-constructed :class:`FrozenFrameIdentity` instance.

    Semantic checks (evaluator):
    * ``nether_portal_offset`` must lie in
      :data:`CASTING_S_C4_FRAME_INTERIOR_SET` →
      :data:`OUTCOME_EXTERNAL_ACTIVATION`;
    * ``agent_id`` must equal :data:`CASTING_S_C4_AGENT_ID` →
      :data:`OUTCOME_WRONG_IGNITION_AGENT` (the activation
      evidence is bound to the same agent as the ignition);
    * ``latched_frame_identity`` must match the wrapping state's
      :class:`FrozenFrameIdentity` field →
      :data:`OUTCOME_FRAME_IDENTITY_MISMATCH`;
    * ``latched_frame_identity`` geometry must match the C3 fixed
      frame plan →
      :data:`OUTCOME_FRAME_IDENTITY_MISMATCH`;
    * ``update_step`` must fall inside the
      ``[ignition.step_id, ignition.step_id + window]`` range →
      :data:`OUTCOME_ACTIVATION_OUTSIDE_WINDOW` /
      :data:`OUTCOME_ACTIVATION_BEFORE_IGNITION`.

    Construction is therefore deliberately *lenient* on the
    interior-set membership and the agent-id semantic: the
    evaluator is the single source of truth for the C4 activation
    contract, and ``external_activation`` /
    ``frame_identity_mismatch`` outcomes must remain reachable
    through the public construction API.
    """

    episode_id: str
    update_step: int
    agent_id: str
    nether_portal_offset: tuple[int, int, int]
    latched_frame_identity: FrozenFrameIdentity

    def __post_init__(self) -> None:
        _require_identifier(self.episode_id, "episode_id")
        _require_non_negative_int(self.update_step, "update_step")
        _require_identifier(self.agent_id, "agent_id")
        object.__setattr__(
            self,
            "nether_portal_offset",
            _require_xyz(self.nether_portal_offset, "nether_portal_offset"),
        )
        if not isinstance(self.latched_frame_identity, FrozenFrameIdentity):
            raise ValueError(
                "latched_frame_identity must be a FrozenFrameIdentity, "
                f"got {type(self.latched_frame_identity).__name__}"
            )


# ----------------------------------------------------------------------
# State
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenIgnitionEvaluationState:
    """Evaluator-only truth for ``casting_s_c4_fixed``.

    The state is the *only* input the C4 evaluator accepts. It
    combines the C3 frame evaluation state
    (:class:`FrozenFrameEvaluationState`) with C4-specific ignition
    and activation evidence plus a typed
    :class:`FrozenFrameIdentity` snapshot.

    Identity consistency
    -------------------

    All identity fields (``episode_id`` / ``step_id`` / ``agent_id``
    / ``max_environment_steps`` / ``max_game_time_seconds`` /
    ``terminated_step``) must agree across the wrapping state, the
    embedded frame state, the frame identity, the ignition action
    and the activation evidence. A disagreement fails closed at
    construction.

    Causality window
    ----------------

    ``causality_window_steps`` is the inclusive upper bound on
    ``activation.update_step - ignition.step_id``. The default
    (:data:`CASTING_S_C4_CAUSALITY_WINDOW_STEPS`, which equals the
    R3 / R5 / C3 default of 4) is inclusive: ``delta ∈ [0, 4]``
    is within window, ``delta > 4`` is outside.

    Frame identity
    --------------

    The wrapping state's ``latched_frame_identity`` is the ground
    truth typed frame identity. The C4 evaluator cross-checks that
    (a) the identity geometry agrees with the C3 frozen frame plan
    and the C4 fixed orientation / corners / width / height, and
    (b) the activation's ``latched_frame_identity`` is
    structurally identical to this one. A mismatch fails closed
    with :data:`OUTCOME_FRAME_IDENTITY_MISMATCH`.
    """

    episode_id: str
    step_id: int
    frame_state: FrozenFrameEvaluationState
    latched_frame_identity: FrozenFrameIdentity
    ignition_action: IgnitionActionEvidence | None = None
    activation_evidence: PortalActivationEvidence | None = None
    agent_id: str = CASTING_S_C4_AGENT_ID
    causality_window_steps: int = CASTING_S_C4_CAUSALITY_WINDOW_STEPS
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
        if self.agent_id != CASTING_S_C4_AGENT_ID:
            raise ValueError(
                f"agent_id must be {CASTING_S_C4_AGENT_ID!r} "
                "for casting_s_c4_fixed"
            )

        # --- frame state ------------------------------------------------
        if not isinstance(self.frame_state, FrozenFrameEvaluationState):
            raise ValueError(
                "frame_state must be a FrozenFrameEvaluationState, "
                f"got {type(self.frame_state).__name__}"
            )
        if self.frame_state.episode_id != self.episode_id:
            raise ValueError(
                "frame_state.episode_id must match the C4 state episode_id"
            )
        if self.frame_state.step_id != self.step_id:
            raise ValueError(
                "frame_state.step_id must match the C4 state step_id"
            )
        if self.frame_state.agent_id != self.agent_id:
            raise ValueError(
                "frame_state.agent_id must match the C4 state agent_id"
            )
        if (
            self.frame_state.max_environment_steps
            != self.max_environment_steps
        ):
            raise ValueError(
                "frame_state.max_environment_steps must match the C4 state"
            )
        if (
            self.frame_state.max_game_time_seconds
            != self.max_game_time_seconds
        ):
            raise ValueError(
                "frame_state.max_game_time_seconds must match the C4 state"
            )

        # --- latched frame identity ------------------------------------
        if not isinstance(self.latched_frame_identity, FrozenFrameIdentity):
            raise ValueError(
                "latched_frame_identity must be a FrozenFrameIdentity, "
                f"got {type(self.latched_frame_identity).__name__}"
            )
        if self.latched_frame_identity.episode_id != self.episode_id:
            raise ValueError(
                "latched_frame_identity.episode_id must match the C4 state"
            )
        if self.latched_frame_identity.agent_id != self.agent_id:
            raise ValueError(
                "latched_frame_identity.agent_id must match the C4 state"
            )
        if self.latched_frame_identity.step_id > self.step_id:
            raise ValueError(
                "latched_frame_identity.step_id cannot be in the future"
            )

        # --- ignition action -------------------------------------------
        if self.ignition_action is not None:
            if not isinstance(self.ignition_action, IgnitionActionEvidence):
                raise ValueError(
                    "ignition_action must be an IgnitionActionEvidence, "
                    f"got {type(self.ignition_action).__name__}"
                )
            if self.ignition_action.episode_id != self.episode_id:
                raise ValueError(
                    "ignition_action.episode_id must match the C4 state"
                )
            if self.ignition_action.step_id > self.step_id:
                raise ValueError(
                    "ignition_action.step_id cannot be in the future"
                )

        # --- activation evidence ---------------------------------------
        if self.activation_evidence is not None:
            if not isinstance(
                self.activation_evidence, PortalActivationEvidence
            ):
                raise ValueError(
                    "activation_evidence must be a PortalActivationEvidence, "
                    f"got {type(self.activation_evidence).__name__}"
                )
            if self.activation_evidence.episode_id != self.episode_id:
                raise ValueError(
                    "activation_evidence.episode_id must match the C4 state"
                )
            if self.activation_evidence.update_step > self.step_id:
                raise ValueError(
                    "activation_evidence.update_step cannot be in the future"
                )

        # --- causality window -------------------------------------------
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

        # --- termination ------------------------------------------------
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
            if self.frame_state.terminated_step != self.terminated_step:
                raise ValueError(
                    "frame_state.terminated_step must match the C4 "
                    "terminated_step when episode_terminated=True"
                )
            if self.terminated_reason is not None:
                _require_identifier(
                    self.terminated_reason, "terminated_reason"
                )
            if self.frame_state.terminated_reason != self.terminated_reason:
                raise ValueError(
                    "frame_state.terminated_reason must match the C4 "
                    "terminated_reason"
                )
        elif (
            self.terminated_step is not None
            or self.terminated_reason is not None
        ):
            raise ValueError(
                "terminated_step/terminated_reason require "
                "episode_terminated=True"
            )
        if (
            not self.frame_state.episode_terminated
            == self.episode_terminated
        ):
            raise ValueError(
                "frame_state.episode_terminated must match the C4 "
                "episode_terminated"
            )

        # --- budgets ----------------------------------------------------
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

        object.__setattr__(
            self, "evidence", _freeze_json_value(self.evidence, "evidence")
        )


# ----------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class FrozenIgnitionEvaluationResult:
    """Typed, frozen, JSON-serializable C4 ignition result.

    ``outcome`` ∈ :data:`IGNITION_OUTCOMES`. ``success`` is derived
    (``outcome == OUTCOME_SUCCESS``). The result carries per-event
    verdicts for the ignition and activation checks plus the C3
    frame outcome in :attr:`frame_outcome` so callers can correlate
    C4 verdicts with the underlying C3 state without re-running the
    C3 evaluator themselves.
    """

    episode_id: str
    step_id: int
    success: bool
    outcome: str
    frame_outcome: str
    ignition_verdict: str
    ignition_agent_verdict: str
    ignition_action_verdict: str
    ignition_item_verdict: str
    ignition_target_verdict: str
    activation_verdict: str
    activation_window_verdict: str
    activation_offset_verdict: str
    activation_agent_verdict: str
    activation_observed_offset: tuple[int, int, int] | None
    activation_delta_steps: int | None
    frame_identity_verdict: str
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
        if self.outcome not in IGNITION_OUTCOMES:
            raise ValueError(f"unknown outcome: {self.outcome!r}")
        if self.success != (self.outcome == OUTCOME_SUCCESS):
            raise ValueError("success must equal (outcome == 'success')")
        if not isinstance(self.frame_outcome, str) or not self.frame_outcome:
            raise ValueError("frame_outcome must be a non-empty string")
        if self.ignition_verdict not in IGNITION_VERDICTS:
            raise ValueError(
                f"unknown ignition_verdict: {self.ignition_verdict!r}"
            )
        if self.activation_verdict not in ACTIVATION_VERDICTS:
            raise ValueError(
                f"unknown activation_verdict: {self.activation_verdict!r}"
            )
        for verdict_name, allowed in (
            ("ignition_agent_verdict", _IGNITION_AGENT_VERDICTS),
            ("ignition_action_verdict", _IGNITION_ACTION_VERDICTS),
            ("ignition_item_verdict", _IGNITION_ITEM_VERDICTS),
            ("ignition_target_verdict", _IGNITION_TARGET_VERDICTS),
            ("activation_window_verdict", _ACTIVATION_WINDOW_VERDICTS),
            ("activation_offset_verdict", _ACTIVATION_OFFSET_VERDICTS),
            ("activation_agent_verdict", _ACTIVATION_AGENT_VERDICTS),
            ("frame_identity_verdict", _FRAME_IDENTITY_VERDICTS),
        ):
            if getattr(self, verdict_name) not in allowed:
                raise ValueError(
                    f"unknown {verdict_name}: "
                    f"{getattr(self, verdict_name)!r}"
                )
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
        if self.activation_observed_offset is not None:
            object.__setattr__(
                self,
                "activation_observed_offset",
                _require_xyz(
                    self.activation_observed_offset,
                    "activation_observed_offset",
                ),
            )
        if self.activation_delta_steps is not None:
            _require_int(
                self.activation_delta_steps, "activation_delta_steps"
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
                raise ValueError(
                    "terminated episode requires terminated_step"
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
        object.__setattr__(
            self, "evidence", _freeze_json_value(self.evidence, "evidence")
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-serializable snapshot."""
        return {
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "success": self.success,
            "outcome": self.outcome,
            "frame_outcome": self.frame_outcome,
            "ignition_verdict": self.ignition_verdict,
            "ignition_agent_verdict": self.ignition_agent_verdict,
            "ignition_action_verdict": self.ignition_action_verdict,
            "ignition_item_verdict": self.ignition_item_verdict,
            "ignition_target_verdict": self.ignition_target_verdict,
            "activation_verdict": self.activation_verdict,
            "activation_window_verdict": self.activation_window_verdict,
            "activation_offset_verdict": self.activation_offset_verdict,
            "activation_agent_verdict": self.activation_agent_verdict,
            "activation_observed_offset": (
                list(self.activation_observed_offset)
                if self.activation_observed_offset is not None
                else None
            ),
            "activation_delta_steps": self.activation_delta_steps,
            "frame_identity_verdict": self.frame_identity_verdict,
            "blocking_conditions": list(self.blocking_conditions),
            "evidence": _thaw_json_value(self.evidence),
            "failure_type": self.failure_type,
            "failure_step": self.failure_step,
            "episode_terminated": self.episode_terminated,
            "terminated_step": self.terminated_step,
            "terminated_reason": self.terminated_reason,
        }


_IGNITION_AGENT_VERDICTS: frozenset[str] = frozenset(
    {IGNITION_AGENT_VERDICT_OK, IGNITION_AGENT_VERDICT_WRONG}
)
_IGNITION_ACTION_VERDICTS: frozenset[str] = frozenset(
    {IGNITION_ACTION_VERDICT_OK, IGNITION_ACTION_VERDICT_WRONG}
)
_IGNITION_ITEM_VERDICTS: frozenset[str] = frozenset(
    {IGNITION_ITEM_VERDICT_OK, IGNITION_ITEM_VERDICT_WRONG}
)
_IGNITION_TARGET_VERDICTS: frozenset[str] = frozenset(
    {IGNITION_TARGET_VERDICT_OK, IGNITION_TARGET_VERDICT_WRONG}
)
_ACTIVATION_WINDOW_VERDICTS: frozenset[str] = frozenset(
    {
        ACTIVATION_WINDOW_VERDICT_OK,
        ACTIVATION_WINDOW_VERDICT_BEFORE,
        ACTIVATION_WINDOW_VERDICT_OUTSIDE,
    }
)
_ACTIVATION_OFFSET_VERDICTS: frozenset[str] = frozenset(
    {
        ACTIVATION_OFFSET_VERDICT_INTERNAL,
        ACTIVATION_OFFSET_VERDICT_EXTERNAL,
    }
)
_ACTIVATION_AGENT_VERDICTS: frozenset[str] = frozenset(
    {ACTIVATION_AGENT_VERDICT_OK, ACTIVATION_AGENT_VERDICT_WRONG}
)
_FRAME_IDENTITY_VERDICTS: frozenset[str] = frozenset(
    {
        FRAME_IDENTITY_VERDICT_MATCH,
        FRAME_IDENTITY_VERDICT_MISSING,
        FRAME_IDENTITY_VERDICT_MISMATCH,
        FRAME_IDENTITY_VERDICT_GEOMETRY_MISMATCH,
    }
)


# ----------------------------------------------------------------------
# Outcome classification
# ----------------------------------------------------------------------


def _classify_ignition_action(
    action: IgnitionActionEvidence | None,
) -> tuple[
    str,
    str,
    str,
    str,
    str,
    str,
    dict[str, Any],
]:
    """Return ignition verdicts and per-ignition evidence.

    Returns ``(ignition_verdict, agent_verdict, action_verdict,
    item_verdict, target_verdict, missing_truth, per_event_evidence)``.
    ``missing_truth`` is one of ``None`` (everything supplied and
    correct) or a short string identifying the first missing /
    wrong component for blocking-conditions construction.
    """
    if action is None:
        per_event: dict[str, Any] = {
            "ignition_action": None,
        }
        return (
            IGNITION_VERDICT_MISSING,
            IGNITION_AGENT_VERDICT_WRONG,
            IGNITION_ACTION_VERDICT_WRONG,
            IGNITION_ITEM_VERDICT_WRONG,
            IGNITION_TARGET_VERDICT_WRONG,
            "ignition_action",
            per_event,
        )
    per_event = {
        "ignition_action": {
            "episode_id": action.episode_id,
            "step_id": action.step_id,
            "agent_id": action.agent_id,
            "action_type": action.action_type,
            "item": action.item,
            "target_cell": list(action.target_cell),
        }
    }
    if action.agent_id != CASTING_S_C4_AGENT_ID:
        per_event["expected_agent_id"] = CASTING_S_C4_AGENT_ID
        per_event["actual_agent_id"] = action.agent_id
        return (
            IGNITION_VERDICT_OBSERVED,
            IGNITION_AGENT_VERDICT_WRONG,
            IGNITION_ACTION_VERDICT_OK,
            IGNITION_ITEM_VERDICT_OK,
            IGNITION_TARGET_VERDICT_OK,
            "ignition_agent",
            per_event,
        )
    if action.action_type != CASTING_S_C4_IGNITION_ACTION_TYPE:
        per_event["expected_action_type"] = CASTING_S_C4_IGNITION_ACTION_TYPE
        per_event["actual_action_type"] = action.action_type
        return (
            IGNITION_VERDICT_OBSERVED,
            IGNITION_AGENT_VERDICT_OK,
            IGNITION_ACTION_VERDICT_WRONG,
            IGNITION_ITEM_VERDICT_OK,
            IGNITION_TARGET_VERDICT_OK,
            "ignition_action_type",
            per_event,
        )
    if action.item != CASTING_S_C4_IGNITION_ITEM:
        per_event["expected_item"] = CASTING_S_C4_IGNITION_ITEM
        per_event["actual_item"] = action.item
        return (
            IGNITION_VERDICT_OBSERVED,
            IGNITION_AGENT_VERDICT_OK,
            IGNITION_ACTION_VERDICT_OK,
            IGNITION_ITEM_VERDICT_WRONG,
            IGNITION_TARGET_VERDICT_OK,
            "ignition_item",
            per_event,
        )
    if action.target_cell != CASTING_S_C4_PUBLIC_IGNITION_TARGET:
        per_event["expected_target_cell"] = list(
            CASTING_S_C4_PUBLIC_IGNITION_TARGET
        )
        per_event["actual_target_cell"] = list(action.target_cell)
        return (
            IGNITION_VERDICT_OBSERVED,
            IGNITION_AGENT_VERDICT_OK,
            IGNITION_ACTION_VERDICT_OK,
            IGNITION_ITEM_VERDICT_OK,
            IGNITION_TARGET_VERDICT_WRONG,
            "ignition_target",
            per_event,
        )
    return (
        IGNITION_VERDICT_OBSERVED,
        IGNITION_AGENT_VERDICT_OK,
        IGNITION_ACTION_VERDICT_OK,
        IGNITION_ITEM_VERDICT_OK,
        IGNITION_TARGET_VERDICT_OK,
        None,
        per_event,
    )


def _frame_identity_matches_c4_c3_plan(
    identity: FrozenFrameIdentity,
) -> bool:
    """Return True iff the typed identity matches the C4 / C3 frozen plan.

    This is the *only* way the C4 evaluator accepts a
    :class:`FrozenFrameIdentity` as the canonical episode-built
    C4 frame identity. Two arbitrary but equal mappings can no
    longer pass; the geometry must be exactly the C3 frozen
    public 4×5 full-ring frame plan with the C4 fixed orientation
    / corners / width / height.
    """
    if identity.orientation != CASTING_S_C4_FRAME_ORIENTATION:
        return False
    if identity.min_corner != CASTING_S_C4_FRAME_MIN_CORNER:
        return False
    if identity.max_corner != CASTING_S_C4_FRAME_MAX_CORNER:
        return False
    if identity.width != CASTING_S_C4_FRAME_WIDTH:
        return False
    if identity.height != CASTING_S_C4_FRAME_HEIGHT:
        return False
    if identity.required_corner_count != CASTING_S_C3_CORNER_CELL_COUNT:
        return False
    if identity.required_full_ring_count != CASTING_S_C3_TARGET_CELL_COUNT:
        return False
    # Identity is a deterministic snapshot, not merely an unordered
    # collection. Preserve the exact C3 contract order so reordered or
    # duplicate offsets cannot masquerade as the canonical frame.
    if identity.target_offsets != tuple(CASTING_S_C3_FRAME_CELLS):
        return False
    if identity.interior_offsets != tuple(CASTING_S_C3_INTERIOR_CELLS):
        return False

    # Activation offsets are evaluator-only truth latched from the
    # episode-built frame. They must be a non-empty, duplicate-free subset
    # of the fixed interior, in the same canonical order as the C3 plan.
    if not identity.activation_offsets:
        return False
    activation_set = set(identity.activation_offsets)
    if len(activation_set) != len(identity.activation_offsets):
        return False
    expected_activation_order = tuple(
        cell
        for cell in CASTING_S_C3_INTERIOR_CELLS
        if cell in activation_set
    )
    if identity.activation_offsets != expected_activation_order:
        return False
    return True


def _classify_activation(
    activation: PortalActivationEvidence | None,
    ignition: IgnitionActionEvidence | None,
    *,
    causality_window_steps: int,
    latched_frame_identity: FrozenFrameIdentity,
) -> tuple[
    str,
    str,
    str,
    str,
    str,
    tuple[int, int, int] | None,
    int | None,
    dict[str, Any],
]:
    """Return activation verdicts and per-activation evidence.

    Returns ``(activation_verdict, window_verdict, offset_verdict,
    agent_verdict, identity_verdict, observed_offset, delta,
    per_event_evidence)``.
    """
    if activation is None:
        return (
            ACTIVATION_VERDICT_MISSING,
            ACTIVATION_WINDOW_VERDICT_OUTSIDE,
            ACTIVATION_OFFSET_VERDICT_EXTERNAL,
            ACTIVATION_AGENT_VERDICT_WRONG,
            FRAME_IDENTITY_VERDICT_MISSING,
            None,
            None,
            {"activation_evidence": None},
        )
    per_event: dict[str, Any] = {
        "activation_evidence": {
            "episode_id": activation.episode_id,
            "update_step": activation.update_step,
            "agent_id": activation.agent_id,
            "nether_portal_offset": list(activation.nether_portal_offset),
            "latched_frame_identity": (
                activation.latched_frame_identity.as_dict()
            ),
        }
    }

    # --- offset (must be in public frame interior) -------------------
    if activation.nether_portal_offset in CASTING_S_C4_FRAME_INTERIOR_SET:
        offset_verdict = ACTIVATION_OFFSET_VERDICT_INTERNAL
    else:
        offset_verdict = ACTIVATION_OFFSET_VERDICT_EXTERNAL
        per_event["expected_interior_cells"] = [
            list(c) for c in CASTING_S_C4_FRAME_INTERIOR_CELLS
        ]
        per_event["actual_offset"] = list(activation.nether_portal_offset)

    # --- agent id (must match ignition agent, default agent_1) -------
    if activation.agent_id == CASTING_S_C4_AGENT_ID:
        agent_verdict = ACTIVATION_AGENT_VERDICT_OK
    else:
        agent_verdict = ACTIVATION_AGENT_VERDICT_WRONG
        per_event["expected_agent_id"] = CASTING_S_C4_AGENT_ID
        per_event["actual_agent_id"] = activation.agent_id

    # --- window / order ---------------------------------------------
    if ignition is None:
        window_verdict = ACTIVATION_WINDOW_VERDICT_BEFORE
        delta: int | None = None
    else:
        delta = activation.update_step - ignition.step_id
        per_event["ignition_step_id"] = ignition.step_id
        per_event["activation_delta_steps"] = delta
        if delta < 0:
            window_verdict = ACTIVATION_WINDOW_VERDICT_BEFORE
        elif delta > causality_window_steps:
            window_verdict = ACTIVATION_WINDOW_VERDICT_OUTSIDE
        else:
            window_verdict = ACTIVATION_WINDOW_VERDICT_OK

    # --- frame identity ---------------------------------------------
    if not _frame_identities_are_equal(
        latched_frame_identity, activation.latched_frame_identity
    ):
        identity_verdict = FRAME_IDENTITY_VERDICT_MISMATCH
        per_event["expected_frame_identity"] = latched_frame_identity.as_dict()
        per_event["actual_frame_identity"] = (
            activation.latched_frame_identity.as_dict()
        )
    elif not _frame_identity_matches_c4_c3_plan(
        activation.latched_frame_identity
    ):
        identity_verdict = FRAME_IDENTITY_VERDICT_GEOMETRY_MISMATCH
        per_event["actual_frame_identity"] = (
            activation.latched_frame_identity.as_dict()
        )
        per_event["expected_c3_full_ring"] = [
            list(c) for c in CASTING_S_C3_FRAME_CELLS
        ]
    elif (
        activation.nether_portal_offset
        not in activation.latched_frame_identity.activation_offsets
    ):
        identity_verdict = FRAME_IDENTITY_VERDICT_MISMATCH
        per_event["actual_frame_identity"] = (
            activation.latched_frame_identity.as_dict()
        )
        per_event["observed_activation_offset"] = list(
            activation.nether_portal_offset
        )
        per_event["latched_activation_offsets"] = [
            list(cell)
            for cell in activation.latched_frame_identity.activation_offsets
        ]
    else:
        identity_verdict = FRAME_IDENTITY_VERDICT_MATCH

    return (
        ACTIVATION_VERDICT_OBSERVED,
        window_verdict,
        offset_verdict,
        agent_verdict,
        identity_verdict,
        activation.nether_portal_offset,
        delta,
        per_event,
    )


# ----------------------------------------------------------------------
# Outcome aggregation
# ----------------------------------------------------------------------


def _check_frame_identity(
    state: FrozenIgnitionEvaluationState,
) -> tuple[str, dict[str, Any]] | None:
    """Return a (outcome, evidence) tuple if the frame identity
    must fail closed, otherwise ``None``.

    The check is independent of C3 success and is the single source
    of truth for "this is the C3 episode-built frame identity".
    Two arbitrary mappings can no longer pass; the geometry
    must be exactly the C3 frozen public 4×5 full-ring frame plan
    with the C4 fixed orientation / corners / width / height.
    """
    identity = state.latched_frame_identity
    evidence: dict[str, Any] = {
        "episode_id": state.episode_id,
        "step_id": state.step_id,
        "agent_id": state.agent_id,
        "latched_frame_identity": identity.as_dict(),
    }
    if not _frame_identity_matches_c4_c3_plan(identity):
        evidence["expected_c3_full_ring"] = [
            list(c) for c in CASTING_S_C3_FRAME_CELLS
        ]
        evidence["expected_c3_interior"] = [
            list(c) for c in CASTING_S_C3_INTERIOR_CELLS
        ]
        return (
            OUTCOME_FRAME_IDENTITY_MISMATCH,
            evidence,
        )
    return None


def _classify_outcome(
    state: FrozenIgnitionEvaluationState,
    frame_outcome: str,
) -> tuple[
    str,
    int | None,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    tuple[int, int, int] | None,
    int | None,
    str,
    tuple[str, ...],
    Mapping[str, Any],
]:
    """Classify the C4 outcome from a state + the C3 frame outcome.

    The C3 frame outcome is supplied by the caller
    (``FrozenIgnitionEvaluator``) so we do not re-run the C3
    evaluator and avoid double counting in the evidence tree.
    """
    # 1. step budget
    observed = [state.step_id]
    if state.terminated_step is not None:
        observed.append(state.terminated_step)
    latest_observed = max(observed)
    if latest_observed > state.max_environment_steps:
        evidence: dict[str, Any] = {
            "episode_id": state.episode_id,
            "step_id": state.step_id,
            "agent_id": state.agent_id,
            "max_environment_steps": state.max_environment_steps,
            "max_game_time_seconds": state.max_game_time_seconds,
            "current_time_seconds": state.current_time_seconds,
            "causality_window_steps": state.causality_window_steps,
            "frame_outcome": frame_outcome,
            "budget_exceeded_kind": "step",
            "budget_exceeded_value": latest_observed,
            "budget_limit": state.max_environment_steps,
        }
        return _build_result_tuple(
            state, frame_outcome, OUTCOME_STEP_BUDGET_EXCEEDED,
            latest_observed, evidence,
        )

    # 2. time budget
    if state.current_time_seconds > state.max_game_time_seconds:
        evidence = {
            "episode_id": state.episode_id,
            "step_id": state.step_id,
            "agent_id": state.agent_id,
            "max_environment_steps": state.max_environment_steps,
            "max_game_time_seconds": state.max_game_time_seconds,
            "current_time_seconds": state.current_time_seconds,
            "causality_window_steps": state.causality_window_steps,
            "frame_outcome": frame_outcome,
            "budget_exceeded_kind": "time",
            "budget_exceeded_value": state.current_time_seconds,
            "budget_limit": state.max_game_time_seconds,
        }
        failure_step = (
            state.terminated_step
            if state.episode_terminated
            else state.step_id
        )
        return _build_result_tuple(
            state, frame_outcome, OUTCOME_TIME_BUDGET_EXCEEDED,
            failure_step, evidence,
        )

    # 3. abnormal_termination (only if terminated)
    if state.episode_terminated:
        if state.terminated_reason is None:
            evidence = {
                "episode_id": state.episode_id,
                "step_id": state.step_id,
                "agent_id": state.agent_id,
                "frame_outcome": frame_outcome,
                "missing_reason": "terminated_reason",
            }
            return _build_result_tuple(
                state, frame_outcome, OUTCOME_TRUTH_MISSING,
                state.terminated_step, evidence,
            )
        if state.terminated_reason not in NORMAL_TERMINATION_REASONS:
            evidence = {
                "episode_id": state.episode_id,
                "step_id": state.step_id,
                "agent_id": state.agent_id,
                "frame_outcome": frame_outcome,
                "terminated_reason": state.terminated_reason,
            }
            return _build_result_tuple(
                state, frame_outcome, OUTCOME_ABNORMAL_TERMINATION,
                state.terminated_step, evidence,
            )

    # 4. C3 frame identity geometry must match the C3 frozen plan
    #    AND the wrapping state's latched_frame_identity must be
    #    the same identity observed by the activation. Two arbitrary
    #    but equal mappings can no longer pass; the geometry
    #    itself must be the C3 frozen public 4×5 full-ring plan.
    frame_identity_failure = _check_frame_identity(state)
    if frame_identity_failure is not None:
        outcome, evidence = frame_identity_failure
        return _build_result_tuple(
            state, frame_outcome, outcome, state.step_id, evidence,
            ignition_verdict=IGNITION_VERDICT_MISSING,
            ignition_agent_verdict=IGNITION_AGENT_VERDICT_WRONG,
            ignition_action_verdict=IGNITION_ACTION_VERDICT_WRONG,
            ignition_item_verdict=IGNITION_ITEM_VERDICT_WRONG,
            ignition_target_verdict=IGNITION_TARGET_VERDICT_WRONG,
            activation_verdict=ACTIVATION_VERDICT_MISSING,
            activation_window_verdict=ACTIVATION_WINDOW_VERDICT_OUTSIDE,
            activation_offset_verdict=ACTIVATION_OFFSET_VERDICT_EXTERNAL,
            activation_agent_verdict=ACTIVATION_AGENT_VERDICT_WRONG,
            frame_identity_verdict=FRAME_IDENTITY_VERDICT_GEOMETRY_MISMATCH,
        )

    # 5. C3 frame outcome drives frame_not_built (or success path
    #    if the frame is built). C3 ``in_progress`` is propagated
    #    so the C4 evaluator reports ``in_progress`` too.
    if frame_outcome == FRAME_OUTCOME_IN_PROGRESS:
        evidence = {
            "episode_id": state.episode_id,
            "step_id": state.step_id,
            "agent_id": state.agent_id,
            "max_environment_steps": state.max_environment_steps,
            "max_game_time_seconds": state.max_game_time_seconds,
            "current_time_seconds": state.current_time_seconds,
            "causality_window_steps": state.causality_window_steps,
            "frame_outcome": frame_outcome,
        }
        return _build_result_tuple(
            state, frame_outcome, OUTCOME_IN_PROGRESS, None, evidence,
            ignition_verdict=IGNITION_VERDICT_MISSING,
            ignition_agent_verdict=IGNITION_AGENT_VERDICT_WRONG,
            ignition_action_verdict=IGNITION_ACTION_VERDICT_WRONG,
            ignition_item_verdict=IGNITION_ITEM_VERDICT_WRONG,
            ignition_target_verdict=IGNITION_TARGET_VERDICT_WRONG,
            activation_verdict=ACTIVATION_VERDICT_MISSING,
            activation_window_verdict=ACTIVATION_WINDOW_VERDICT_OUTSIDE,
            activation_offset_verdict=ACTIVATION_OFFSET_VERDICT_EXTERNAL,
            activation_agent_verdict=ACTIVATION_AGENT_VERDICT_WRONG,
            frame_identity_verdict=FRAME_IDENTITY_VERDICT_MATCH,
        )
    if frame_outcome != FRAME_OUTCOME_SUCCESS:
        evidence = {
            "episode_id": state.episode_id,
            "step_id": state.step_id,
            "agent_id": state.agent_id,
            "max_environment_steps": state.max_environment_steps,
            "max_game_time_seconds": state.max_game_time_seconds,
            "current_time_seconds": state.current_time_seconds,
            "causality_window_steps": state.causality_window_steps,
            "frame_outcome": frame_outcome,
        }
        return _build_result_tuple(
            state, frame_outcome, OUTCOME_FRAME_NOT_BUILT,
            state.step_id, evidence,
        )

    # 6. ignition action checks (priority: missing > agent > action > item > target)
    (
        ignition_verdict,
        ignition_agent_verdict,
        ignition_action_verdict,
        ignition_item_verdict,
        ignition_target_verdict,
        ignition_missing_reason,
        ignition_evidence,
    ) = _classify_ignition_action(state.ignition_action)
    if ignition_verdict == IGNITION_VERDICT_MISSING:
        evidence = {
            "episode_id": state.episode_id,
            "step_id": state.step_id,
            "agent_id": state.agent_id,
            "max_environment_steps": state.max_environment_steps,
            "max_game_time_seconds": state.max_game_time_seconds,
            "current_time_seconds": state.current_time_seconds,
            "causality_window_steps": state.causality_window_steps,
            "frame_outcome": frame_outcome,
            "ignition": ignition_evidence,
        }
        return _build_result_tuple(
            state, frame_outcome,
            OUTCOME_IGNITION_ACTION_MISSING, state.step_id, evidence,
            ignition_verdict=ignition_verdict,
            ignition_agent_verdict=ignition_agent_verdict,
            ignition_action_verdict=ignition_action_verdict,
            ignition_item_verdict=ignition_item_verdict,
            ignition_target_verdict=ignition_target_verdict,
            activation_verdict=ACTIVATION_VERDICT_MISSING,
            activation_window_verdict=ACTIVATION_WINDOW_VERDICT_OUTSIDE,
            activation_offset_verdict=ACTIVATION_OFFSET_VERDICT_EXTERNAL,
            activation_agent_verdict=ACTIVATION_AGENT_VERDICT_WRONG,
            frame_identity_verdict=FRAME_IDENTITY_VERDICT_MATCH,
        )

    ignition_failure_outcome: str | None = None
    if ignition_missing_reason == "ignition_agent":
        ignition_failure_outcome = OUTCOME_WRONG_IGNITION_AGENT
    elif ignition_missing_reason == "ignition_action_type":
        ignition_failure_outcome = OUTCOME_WRONG_IGNITION_ACTION
    elif ignition_missing_reason == "ignition_item":
        ignition_failure_outcome = OUTCOME_WRONG_IGNITION_ITEM
    elif ignition_missing_reason == "ignition_target":
        ignition_failure_outcome = OUTCOME_WRONG_IGNITION_TARGET
    if ignition_failure_outcome is not None:
        evidence = {
            "episode_id": state.episode_id,
            "step_id": state.step_id,
            "agent_id": state.agent_id,
            "frame_outcome": frame_outcome,
            "ignition": ignition_evidence,
        }
        return _build_result_tuple(
            state, frame_outcome,
            ignition_failure_outcome, state.step_id, evidence,
            ignition_verdict=ignition_verdict,
            ignition_agent_verdict=ignition_agent_verdict,
            ignition_action_verdict=ignition_action_verdict,
            ignition_item_verdict=ignition_item_verdict,
            ignition_target_verdict=ignition_target_verdict,
            activation_verdict=ACTIVATION_VERDICT_MISSING,
            activation_window_verdict=ACTIVATION_WINDOW_VERDICT_OUTSIDE,
            activation_offset_verdict=ACTIVATION_OFFSET_VERDICT_EXTERNAL,
            activation_agent_verdict=ACTIVATION_AGENT_VERDICT_WRONG,
            frame_identity_verdict=FRAME_IDENTITY_VERDICT_MATCH,
        )

    # 7. activation checks (only if ignition action is correct)
    (
        activation_verdict,
        activation_window_verdict,
        activation_offset_verdict,
        activation_agent_verdict,
        frame_identity_verdict,
        observed_offset,
        activation_delta,
        activation_evidence,
    ) = _classify_activation(
        state.activation_evidence,
        state.ignition_action,
        causality_window_steps=state.causality_window_steps,
        latched_frame_identity=state.latched_frame_identity,
    )

    if activation_verdict == ACTIVATION_VERDICT_MISSING:
        evidence = {
            "episode_id": state.episode_id,
            "step_id": state.step_id,
            "agent_id": state.agent_id,
            "frame_outcome": frame_outcome,
            "ignition": ignition_evidence,
            "activation": activation_evidence,
        }
        return _build_result_tuple(
            state, frame_outcome,
            OUTCOME_ACTIVATION_MISSING, state.step_id, evidence,
            ignition_verdict=ignition_verdict,
            ignition_agent_verdict=ignition_agent_verdict,
            ignition_action_verdict=ignition_action_verdict,
            ignition_item_verdict=ignition_item_verdict,
            ignition_target_verdict=ignition_target_verdict,
            activation_verdict=activation_verdict,
            activation_window_verdict=activation_window_verdict,
            activation_offset_verdict=activation_offset_verdict,
            activation_agent_verdict=activation_agent_verdict,
            frame_identity_verdict=frame_identity_verdict,
        )

    # 8. activation agent mismatch
    if activation_agent_verdict == ACTIVATION_AGENT_VERDICT_WRONG:
        evidence = {
            "episode_id": state.episode_id,
            "step_id": state.step_id,
            "agent_id": state.agent_id,
            "frame_outcome": frame_outcome,
            "ignition": ignition_evidence,
            "activation": activation_evidence,
        }
        return _build_result_tuple(
            state, frame_outcome,
            OUTCOME_WRONG_IGNITION_AGENT, state.step_id, evidence,
            ignition_verdict=ignition_verdict,
            ignition_agent_verdict=ignition_agent_verdict,
            ignition_action_verdict=ignition_action_verdict,
            ignition_item_verdict=ignition_item_verdict,
            ignition_target_verdict=ignition_target_verdict,
            activation_verdict=activation_verdict,
            activation_window_verdict=activation_window_verdict,
            activation_offset_verdict=activation_offset_verdict,
            activation_agent_verdict=activation_agent_verdict,
            frame_identity_verdict=frame_identity_verdict,
            observed_offset=observed_offset,
            activation_delta=activation_delta,
        )

    # 9. window / step order
    if activation_window_verdict == ACTIVATION_WINDOW_VERDICT_BEFORE:
        evidence = {
            "episode_id": state.episode_id,
            "step_id": state.step_id,
            "agent_id": state.agent_id,
            "frame_outcome": frame_outcome,
            "ignition": ignition_evidence,
            "activation": activation_evidence,
        }
        return _build_result_tuple(
            state, frame_outcome,
            OUTCOME_ACTIVATION_BEFORE_IGNITION,
            state.step_id, evidence,
            ignition_verdict=ignition_verdict,
            ignition_agent_verdict=ignition_agent_verdict,
            ignition_action_verdict=ignition_action_verdict,
            ignition_item_verdict=ignition_item_verdict,
            ignition_target_verdict=ignition_target_verdict,
            activation_verdict=activation_verdict,
            activation_window_verdict=activation_window_verdict,
            activation_offset_verdict=activation_offset_verdict,
            activation_agent_verdict=activation_agent_verdict,
            frame_identity_verdict=frame_identity_verdict,
            observed_offset=observed_offset,
            activation_delta=activation_delta,
        )
    if activation_window_verdict == ACTIVATION_WINDOW_VERDICT_OUTSIDE:
        evidence = {
            "episode_id": state.episode_id,
            "step_id": state.step_id,
            "agent_id": state.agent_id,
            "frame_outcome": frame_outcome,
            "ignition": ignition_evidence,
            "activation": activation_evidence,
        }
        return _build_result_tuple(
            state, frame_outcome,
            OUTCOME_ACTIVATION_OUTSIDE_WINDOW,
            state.step_id, evidence,
            ignition_verdict=ignition_verdict,
            ignition_agent_verdict=ignition_agent_verdict,
            ignition_action_verdict=ignition_action_verdict,
            ignition_item_verdict=ignition_item_verdict,
            ignition_target_verdict=ignition_target_verdict,
            activation_verdict=activation_verdict,
            activation_window_verdict=activation_window_verdict,
            activation_offset_verdict=activation_offset_verdict,
            activation_agent_verdict=activation_agent_verdict,
            frame_identity_verdict=frame_identity_verdict,
            observed_offset=observed_offset,
            activation_delta=activation_delta,
        )

    # 10. external activation (offset not in frame interior)
    if activation_offset_verdict == ACTIVATION_OFFSET_VERDICT_EXTERNAL:
        evidence = {
            "episode_id": state.episode_id,
            "step_id": state.step_id,
            "agent_id": state.agent_id,
            "frame_outcome": frame_outcome,
            "ignition": ignition_evidence,
            "activation": activation_evidence,
        }
        return _build_result_tuple(
            state, frame_outcome,
            OUTCOME_EXTERNAL_ACTIVATION, state.step_id, evidence,
            ignition_verdict=ignition_verdict,
            ignition_agent_verdict=ignition_agent_verdict,
            ignition_action_verdict=ignition_action_verdict,
            ignition_item_verdict=ignition_item_verdict,
            ignition_target_verdict=ignition_target_verdict,
            activation_verdict=activation_verdict,
            activation_window_verdict=activation_window_verdict,
            activation_offset_verdict=activation_offset_verdict,
            activation_agent_verdict=activation_agent_verdict,
            frame_identity_verdict=frame_identity_verdict,
            observed_offset=observed_offset,
            activation_delta=activation_delta,
        )

    # 11. frame identity mismatch (state's identity != activation's)
    if frame_identity_verdict == FRAME_IDENTITY_VERDICT_MISMATCH:
        evidence = {
            "episode_id": state.episode_id,
            "step_id": state.step_id,
            "agent_id": state.agent_id,
            "frame_outcome": frame_outcome,
            "ignition": ignition_evidence,
            "activation": activation_evidence,
        }
        return _build_result_tuple(
            state, frame_outcome,
            OUTCOME_FRAME_IDENTITY_MISMATCH, state.step_id, evidence,
            ignition_verdict=ignition_verdict,
            ignition_agent_verdict=ignition_agent_verdict,
            ignition_action_verdict=ignition_action_verdict,
            ignition_item_verdict=ignition_item_verdict,
            ignition_target_verdict=ignition_target_verdict,
            activation_verdict=activation_verdict,
            activation_window_verdict=activation_window_verdict,
            activation_offset_verdict=activation_offset_verdict,
            activation_agent_verdict=activation_agent_verdict,
            frame_identity_verdict=frame_identity_verdict,
            observed_offset=observed_offset,
            activation_delta=activation_delta,
        )
    if frame_identity_verdict == FRAME_IDENTITY_VERDICT_GEOMETRY_MISMATCH:
        evidence = {
            "episode_id": state.episode_id,
            "step_id": state.step_id,
            "agent_id": state.agent_id,
            "frame_outcome": frame_outcome,
            "ignition": ignition_evidence,
            "activation": activation_evidence,
        }
        return _build_result_tuple(
            state, frame_outcome,
            OUTCOME_FRAME_IDENTITY_MISMATCH, state.step_id, evidence,
            ignition_verdict=ignition_verdict,
            ignition_agent_verdict=ignition_agent_verdict,
            ignition_action_verdict=ignition_action_verdict,
            ignition_item_verdict=ignition_item_verdict,
            ignition_target_verdict=ignition_target_verdict,
            activation_verdict=activation_verdict,
            activation_window_verdict=activation_window_verdict,
            activation_offset_verdict=activation_offset_verdict,
            activation_agent_verdict=activation_agent_verdict,
            frame_identity_verdict=frame_identity_verdict,
            observed_offset=observed_offset,
            activation_delta=activation_delta,
        )

    # 12. success path (episode must be terminated, within budget)
    if not state.episode_terminated:
        evidence = {
            "episode_id": state.episode_id,
            "step_id": state.step_id,
            "agent_id": state.agent_id,
            "frame_outcome": frame_outcome,
            "ignition": ignition_evidence,
            "activation": activation_evidence,
        }
        return _build_result_tuple(
            state, frame_outcome,
            OUTCOME_IN_PROGRESS, None, evidence,
            ignition_verdict=ignition_verdict,
            ignition_agent_verdict=ignition_agent_verdict,
            ignition_action_verdict=ignition_action_verdict,
            ignition_item_verdict=ignition_item_verdict,
            ignition_target_verdict=ignition_target_verdict,
            activation_verdict=activation_verdict,
            activation_window_verdict=activation_window_verdict,
            activation_offset_verdict=activation_offset_verdict,
            activation_agent_verdict=activation_agent_verdict,
            frame_identity_verdict=frame_identity_verdict,
            observed_offset=observed_offset,
            activation_delta=activation_delta,
        )

    # All checks pass; emit success.
    evidence = {
        "episode_id": state.episode_id,
        "step_id": state.step_id,
        "agent_id": state.agent_id,
        "max_environment_steps": state.max_environment_steps,
        "max_game_time_seconds": state.max_game_time_seconds,
        "current_time_seconds": state.current_time_seconds,
        "causality_window_steps": state.causality_window_steps,
        "frame_outcome": frame_outcome,
        "ignition": ignition_evidence,
        "activation": activation_evidence,
    }
    return _build_result_tuple(
        state, frame_outcome,
        OUTCOME_SUCCESS, None, evidence,
        ignition_verdict=ignition_verdict,
        ignition_agent_verdict=ignition_agent_verdict,
        ignition_action_verdict=ignition_action_verdict,
        ignition_item_verdict=ignition_item_verdict,
        ignition_target_verdict=ignition_target_verdict,
        activation_verdict=activation_verdict,
        activation_window_verdict=activation_window_verdict,
        activation_offset_verdict=activation_offset_verdict,
        activation_agent_verdict=activation_agent_verdict,
        frame_identity_verdict=frame_identity_verdict,
        observed_offset=observed_offset,
        activation_delta=activation_delta,
    )


def _build_result_tuple(
    state: FrozenIgnitionEvaluationState,
    frame_outcome: str,
    outcome: str,
    failure_step: int | None,
    evidence: dict[str, Any],
    *,
    ignition_verdict: str = IGNITION_VERDICT_MISSING,
    ignition_agent_verdict: str = IGNITION_AGENT_VERDICT_WRONG,
    ignition_action_verdict: str = IGNITION_ACTION_VERDICT_WRONG,
    ignition_item_verdict: str = IGNITION_ITEM_VERDICT_WRONG,
    ignition_target_verdict: str = IGNITION_TARGET_VERDICT_WRONG,
    activation_verdict: str = ACTIVATION_VERDICT_MISSING,
    activation_window_verdict: str = ACTIVATION_WINDOW_VERDICT_OUTSIDE,
    activation_offset_verdict: str = ACTIVATION_OFFSET_VERDICT_EXTERNAL,
    activation_agent_verdict: str = ACTIVATION_AGENT_VERDICT_WRONG,
    frame_identity_verdict: str = FRAME_IDENTITY_VERDICT_MISSING,
    observed_offset: tuple[int, int, int] | None = None,
    activation_delta: int | None = None,
) -> tuple[
    str,
    int | None,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    tuple[int, int, int] | None,
    int | None,
    str,
    tuple[str, ...],
    Mapping[str, Any],
]:
    """Compose the per-event evidence + blocking conditions + return tuple.

    The blocking conditions follow the C3 convention: a small set of
    stable ``"<key>:<value>"`` strings. The C4 evaluator adds the
    activation offset / agent / action / item / target variants
    plus the frame-identity / external-activation labels.
    """
    blocking = _blocking_conditions_for(
        outcome, evidence, frame_outcome
    )
    return (
        outcome,
        failure_step,
        frame_outcome,
        ignition_verdict,
        ignition_agent_verdict,
        ignition_action_verdict,
        ignition_item_verdict,
        ignition_target_verdict,
        activation_verdict,
        activation_window_verdict,
        activation_offset_verdict,
        activation_agent_verdict,
        frame_identity_verdict,
        observed_offset,
        activation_delta,
        FRAME_OUTCOME_SUCCESS,  # placeholder: not a result field
        blocking,
        MappingProxyType(evidence),
    )


def _blocking_conditions_for(
    outcome: str,
    evidence: Mapping[str, Any],
    frame_outcome: str,
) -> tuple[str, ...]:
    """Return a tuple of stable, human-readable blocking labels."""
    if outcome == OUTCOME_SUCCESS:
        return ()
    if outcome == OUTCOME_IN_PROGRESS:
        return ("episode_not_terminated",)
    if outcome == OUTCOME_STEP_BUDGET_EXCEEDED:
        return ("step_budget_exceeded",)
    if outcome == OUTCOME_TIME_BUDGET_EXCEEDED:
        return ("time_budget_exceeded",)
    if outcome == OUTCOME_ABNORMAL_TERMINATION:
        return ("abnormal_termination",)
    if outcome == OUTCOME_FRAME_NOT_BUILT:
        return (
            f"frame_not_built:frame_outcome={frame_outcome}",
        )
    if outcome == OUTCOME_IGNITION_ACTION_MISSING:
        return ("ignition_action_missing",)
    if outcome == OUTCOME_WRONG_IGNITION_AGENT:
        return ("ignition_agent_mismatch",)
    if outcome == OUTCOME_WRONG_IGNITION_ACTION:
        return ("ignition_action_type_mismatch",)
    if outcome == OUTCOME_WRONG_IGNITION_ITEM:
        return ("ignition_item_mismatch",)
    if outcome == OUTCOME_WRONG_IGNITION_TARGET:
        return ("ignition_target_mismatch",)
    if outcome == OUTCOME_ACTIVATION_MISSING:
        return ("portal_activation_missing",)
    if outcome == OUTCOME_ACTIVATION_BEFORE_IGNITION:
        return ("activation_before_ignition",)
    if outcome == OUTCOME_ACTIVATION_OUTSIDE_WINDOW:
        return ("activation_outside_window",)
    if outcome == OUTCOME_EXTERNAL_ACTIVATION:
        return ("external_activation",)
    if outcome == OUTCOME_FRAME_IDENTITY_MISSING:
        return ("frame_identity_missing",)
    if outcome == OUTCOME_FRAME_IDENTITY_MISMATCH:
        return ("frame_identity_mismatch",)
    if outcome == OUTCOME_TRUTH_MISSING:
        return ("truth_missing",)
    return ()


# ----------------------------------------------------------------------
# Evaluator
# ----------------------------------------------------------------------


class FrozenIgnitionEvaluator:
    """Deterministic, offline evaluator for ``casting_s_c4_fixed``.

    The evaluator is a *pure* object: ``evaluate()`` has no side
    effects, reads no global state, and never inspects Agent
    prompts, images, memory, the driver surface, or workflow code.
    Its single input is a :class:`FrozenIgnitionEvaluationState`;
    its single output is a :class:`FrozenIgnitionEvaluationResult`.

    The evaluator internally runs the C3 frame evaluator
    (:class:`FrozenFrameEvaluator`) on the embedded
    :attr:`FrozenIgnitionEvaluationState.frame_state` to re-verify
    the C3 14-cell conditions. The C3 outcome is preserved in
    :attr:`FrozenIgnitionEvaluationResult.frame_outcome` so callers
    can correlate the two without re-running the C3 evaluator
    themselves.

    Failure classification priority (most specific first)
    ----------------------------------------------------

    1. :data:`OUTCOME_STEP_BUDGET_EXCEEDED` — step budget exceeded
       before any per-event verdict could be established.
    2. :data:`OUTCOME_TIME_BUDGET_EXCEEDED` — time budget exceeded.
    3. :data:`OUTCOME_ABNORMAL_TERMINATION` — episode ended for a
       reason outside :data:`NORMAL_TERMINATION_REASONS`.
    4. :data:`OUTCOME_FRAME_IDENTITY_MISMATCH` — the wrapping
       state's :attr:`FrozenIgnitionEvaluationState.latched_frame_identity`
       does not match the C3 / C4 frozen frame geometry. Two
       arbitrary equal mappings can no longer pass; the
       ``FrozenFrameIdentity`` must agree with the C3 frozen
       public 4×5 full-ring frame plan.
    5. C3 ``in_progress`` → :data:`OUTCOME_IN_PROGRESS` (propagated).
    6. :data:`OUTCOME_FRAME_NOT_BUILT` — C3 frame evaluator did
       not return ``success`` on the embedded frame state.
    7. :data:`OUTCOME_IGNITION_ACTION_MISSING` — no ignition action
       evidence supplied.
    8. :data:`OUTCOME_WRONG_IGNITION_AGENT` / ACTION / ITEM /
       TARGET — the ignition action was constructed but does not
       match the C4 whitelist (``agent_1`` / ``use_item`` /
       ``flint_and_steel`` / ``(1, 1, 1)``).
    9. :data:`OUTCOME_ACTIVATION_MISSING` — no nether_portal
       activation evidence.
    10. :data:`OUTCOME_WRONG_IGNITION_AGENT` — activation
        evidence by non-``agent_1``.
    11. :data:`OUTCOME_ACTIVATION_BEFORE_IGNITION` — activation
        ``update_step`` earlier than ignition ``step_id``.
    12. :data:`OUTCOME_ACTIVATION_OUTSIDE_WINDOW` — activation
        delta is greater than :attr:`causality_window_steps`.
    13. :data:`OUTCOME_EXTERNAL_ACTIVATION` — activation
        ``nether_portal_offset`` is not in
        :data:`CASTING_S_C4_FRAME_INTERIOR_CELLS`.
    14. :data:`OUTCOME_FRAME_IDENTITY_MISMATCH` — activation
        ``latched_frame_identity`` is structurally different from
        the state's :class:`FrozenFrameIdentity`, or the
        activation's identity geometry does not match the C3 frozen
        plan.
    15. :data:`OUTCOME_FRAME_IDENTITY_MISSING` — identity
        comparisons cannot be performed (defensive: the
        construction-time guards should make this unreachable
        under normal use).
    16. :data:`OUTCOME_IN_PROGRESS` — episode not yet terminated
        (all other checks passed).
    17. :data:`OUTCOME_SUCCESS` — episode terminated normally and
        every check passed.
    """

    def evaluate(
        self, state: FrozenIgnitionEvaluationState
    ) -> FrozenIgnitionEvaluationResult:
        # Re-verify C3 frame conditions. The C3 evaluator is the
        # single source of truth for the 14-cell 浇筑 verdict.
        frame_result = FrozenFrameEvaluator().evaluate(state.frame_state)
        frame_outcome = frame_result.outcome
        (
            outcome,
            failure_step,
            _frame_outcome_dup,
            ignition_verdict,
            ignition_agent_verdict,
            ignition_action_verdict,
            ignition_item_verdict,
            ignition_target_verdict,
            activation_verdict,
            activation_window_verdict,
            activation_offset_verdict,
            activation_agent_verdict,
            frame_identity_verdict,
            observed_offset,
            activation_delta,
            _placeholder,
            blocking,
            evidence,
        ) = _classify_outcome(state, frame_outcome)
        success = outcome == OUTCOME_SUCCESS
        if outcome in _TERMINAL_FAILURE_OUTCOMES:
            failure_type: str | None = outcome
        else:
            failure_type = None
        return FrozenIgnitionEvaluationResult(
            episode_id=state.episode_id,
            step_id=state.step_id,
            success=success,
            outcome=outcome,
            frame_outcome=frame_outcome,
            ignition_verdict=ignition_verdict,
            ignition_agent_verdict=ignition_agent_verdict,
            ignition_action_verdict=ignition_action_verdict,
            ignition_item_verdict=ignition_item_verdict,
            ignition_target_verdict=ignition_target_verdict,
            activation_verdict=activation_verdict,
            activation_window_verdict=activation_window_verdict,
            activation_offset_verdict=activation_offset_verdict,
            activation_agent_verdict=activation_agent_verdict,
            activation_observed_offset=observed_offset,
            activation_delta_steps=activation_delta,
            frame_identity_verdict=frame_identity_verdict,
            blocking_conditions=blocking,
            evidence=evidence,
            failure_type=failure_type,
            failure_step=failure_step,
            episode_terminated=state.episode_terminated,
            terminated_step=state.terminated_step,
            terminated_reason=state.terminated_reason,
        )


__all__ = [
    "ACTIVATION_OFFSET_VERDICT_EXTERNAL",
    "ACTIVATION_OFFSET_VERDICT_INTERNAL",
    "ACTIVATION_VERDICT_MISSING",
    "ACTIVATION_VERDICT_OBSERVED",
    "ACTIVATION_VERDICTS",
    "ACTIVATION_WINDOW_VERDICT_BEFORE",
    "ACTIVATION_WINDOW_VERDICT_OK",
    "ACTIVATION_WINDOW_VERDICT_OUTSIDE",
    "CASTING_S_C4_AGENT_ID",
    "CASTING_S_C4_CAUSALITY_WINDOW_STEPS",
    "CASTING_S_C4_FRAME_HEIGHT",
    "CASTING_S_C4_FRAME_INTERIOR_CELLS",
    "CASTING_S_C4_FRAME_INTERIOR_SET",
    "CASTING_S_C4_FRAME_MAX_CORNER",
    "CASTING_S_C4_FRAME_MIN_CORNER",
    "CASTING_S_C4_FRAME_ORIENTATION",
    "CASTING_S_C4_FRAME_WIDTH",
    "CASTING_S_C4_IGNITION_ACTION_TYPE",
    "CASTING_S_C4_IGNITION_ITEM",
    "CASTING_S_C4_PUBLIC_IGNITION_TARGET",
    "FRAME_IDENTITY_VERDICT_GEOMETRY_MISMATCH",
    "FRAME_IDENTITY_VERDICT_MATCH",
    "FRAME_IDENTITY_VERDICT_MISSING",
    "FRAME_IDENTITY_VERDICT_MISMATCH",
    "FrozenFrameIdentity",
    "FrozenIgnitionEvaluationResult",
    "FrozenIgnitionEvaluationState",
    "FrozenIgnitionEvaluator",
    "IGNITION_OUTCOMES",
    "IGNITION_VERDICT_MISSING",
    "IGNITION_VERDICT_OBSERVED",
    "IGNITION_VERDICTS",
    "IGNITION_AGENT_VERDICT_OK",
    "IGNITION_AGENT_VERDICT_WRONG",
    "IGNITION_ACTION_VERDICT_OK",
    "IGNITION_ACTION_VERDICT_WRONG",
    "IGNITION_ITEM_VERDICT_OK",
    "IGNITION_ITEM_VERDICT_WRONG",
    "IGNITION_TARGET_VERDICT_OK",
    "IGNITION_TARGET_VERDICT_WRONG",
    "IgnitionActionEvidence",
    "OUTCOME_ABNORMAL_TERMINATION",
    "OUTCOME_ACTIVATION_BEFORE_IGNITION",
    "OUTCOME_ACTIVATION_MISSING",
    "OUTCOME_ACTIVATION_OUTSIDE_WINDOW",
    "OUTCOME_EXTERNAL_ACTIVATION",
    "OUTCOME_FRAME_IDENTITY_MISSING",
    "OUTCOME_FRAME_IDENTITY_MISMATCH",
    "OUTCOME_FRAME_NOT_BUILT",
    "OUTCOME_IGNITION_ACTION_MISSING",
    "OUTCOME_IGNITION_ACTION_MISMATCH",
    "OUTCOME_IGNITION_AGENT_MISMATCH",
    "OUTCOME_IGNITION_ITEM_MISMATCH",
    "OUTCOME_IGNITION_TARGET_MISMATCH",
    "OUTCOME_IN_PROGRESS",
    "OUTCOME_INVALID_INITIAL_STATE",
    "OUTCOME_STEP_BUDGET_EXCEEDED",
    "OUTCOME_SUCCESS",
    "OUTCOME_TIME_BUDGET_EXCEEDED",
    "OUTCOME_TRUTH_MISSING",
    "OUTCOME_WRONG_IGNITION_ACTION",
    "OUTCOME_WRONG_IGNITION_AGENT",
    "OUTCOME_WRONG_IGNITION_ITEM",
    "OUTCOME_WRONG_IGNITION_TARGET",
    "PortalActivationEvidence",
    "build_c4_c3_frame_identity",
]

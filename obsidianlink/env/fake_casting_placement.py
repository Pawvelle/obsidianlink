"""Independent evaluator-only C1 placement geometry for FakeBackend.

The oracle deliberately does not import the C1 driver or its constants.  It
models MineRL camera values as relative deltas, checks a frozen set of solid
top faces, enforces reach, and records typed casting milestones.  None of this
state is copied into Agent-visible observations.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from obsidianlink.core.types import MacroAction
from obsidianlink.evaluation.casting import (
    CastingEvaluationState,
    CastingFluidTruth,
    CastingTransitionEvidence,
)


CASTING_C1_WORKFLOW = "casting_c1_fixed"
FROZEN_TARGET_CELL: tuple[int, int, int] = (2, 4, 3)
PLAYER_EYE: tuple[float, float, float] = (0.5, 5.62, 0.5)
SUPPORT_FACE_POINTS: tuple[tuple[float, float, float], ...] = (
    (1.5, 4.0, 3.5),
    (1.5, 5.0, 3.5),
)
LAVA_FACE_POINT: tuple[float, float, float] = (2.5, 4.0, 3.5)
WATER_FACE_POINT: tuple[float, float, float] = (2.5, 4.0, 2.5)
ALWAYS_SOLID_TOP_FACES = frozenset(
    (SUPPORT_FACE_POINTS[0], LAVA_FACE_POINT, WATER_FACE_POINT)
)
MAX_REACH = 4.5
AIM_TOLERANCE_DEGREES = 0.25

PLACEMENT_FAILURE_MODES: frozenset[str] = frozenset(
    {"not_aimed", "too_far", "no_valid_face", "no_world_effect"}
)


def _copy_inventory(inventory: Mapping[str, int]) -> dict[str, int]:
    return {
        str(item): int(quantity)
        for item, quantity in inventory.items()
        if isinstance(quantity, int) and not isinstance(quantity, bool)
    }


def _aim_angles(
    point: tuple[float, float, float],
) -> tuple[float, float]:
    delta_x = point[0] - PLAYER_EYE[0]
    delta_y = point[1] - PLAYER_EYE[1]
    delta_z = point[2] - PLAYER_EYE[2]
    horizontal = math.hypot(delta_x, delta_z)
    return (
        -math.degrees(math.atan2(delta_x, delta_z)),
        -math.degrees(math.atan2(delta_y, horizontal)),
    )


def _distance(point: tuple[float, float, float]) -> float:
    return math.dist(PLAYER_EYE, point)


class CastingPlacementState:
    """Mutable evaluator-only placement state for one C1 episode."""

    def __init__(self, initial_inventory: Mapping[str, int]) -> None:
        self.yaw = 0.0
        self.pitch = 0.0
        self.inventory = _copy_inventory(initial_inventory)
        self.selected_item: str | None = None
        self.target_block = "air"
        self.lava_present = False
        self.water_present = False
        self.support_blocks_placed = 0
        self.pending_obsidian = False
        self.failure_mode: str | None = None
        self.diagnostics: list[dict[str, Any]] = []
        self.grid_revision = 0
        self.lava_step: int | None = None
        self.water_step: int | None = None
        self.obsidian_step: int | None = None
        self.relevant_action_steps: list[int] = []

    def set_failure_mode(self, mode: str | None) -> None:
        if mode is not None and mode not in PLACEMENT_FAILURE_MODES:
            raise ValueError(f"unknown casting placement failure mode: {mode!r}")
        self.failure_mode = mode

    def _expected_face(
        self, action_type: str, target: str | None
    ) -> tuple[float, float, float] | None:
        if action_type == "place_block" and target == "cobblestone":
            index = min(self.support_blocks_placed, len(SUPPORT_FACE_POINTS) - 1)
            return SUPPORT_FACE_POINTS[index]
        if action_type == "use_item" and target == "lava_bucket":
            return LAVA_FACE_POINT
        if action_type == "use_item" and target == "water_bucket":
            return WATER_FACE_POINT
        return None

    def aimed_at(self, point: tuple[float, float, float] | None) -> bool:
        if point is None or self.failure_mode == "not_aimed":
            return False
        expected_yaw, expected_pitch = _aim_angles(point)
        return (
            abs(self.yaw - expected_yaw) <= AIM_TOLERANCE_DEGREES
            and abs(self.pitch - expected_pitch) <= AIM_TOLERANCE_DEGREES
        )

    def in_range(self, point: tuple[float, float, float] | None) -> bool:
        if point is None or self.failure_mode == "too_far":
            return False
        return _distance(point) <= MAX_REACH

    def valid_face(self, point: tuple[float, float, float] | None) -> bool:
        if point is None or self.failure_mode == "no_valid_face":
            return False
        if point in ALWAYS_SOLID_TOP_FACES:
            return True
        # The second click face is the top of the first placed cobblestone;
        # it must not exist in the oracle before that world effect occurs.
        return point == SUPPORT_FACE_POINTS[1] and self.support_blocks_placed >= 1

    def _record(
        self,
        *,
        step_id: int,
        action_type: str,
        target: str | None,
        reason: str,
        effect: bool,
        expected_face: tuple[float, float, float] | None = None,
    ) -> None:
        self.diagnostics.append(
            {
                "step_id": step_id,
                "action_type": action_type,
                "target": target,
                "reason": reason,
                "effect": effect,
                "camera_yaw": self.yaw,
                "camera_pitch": self.pitch,
                "aimed": self.aimed_at(expected_face),
                "in_range": self.in_range(expected_face),
                "valid_face": self.valid_face(expected_face),
                "target_cell": list(FROZEN_TARGET_CELL),
                "target_block": self.target_block,
                "grid_revision": self.grid_revision,
            }
        )

    def apply(self, action: MacroAction, *, step_id: int) -> None:
        action_type = action.action_type
        if action_type == "look":
            if self.failure_mode != "not_aimed":
                self.yaw += float(action.parameters["yaw"])
                self.pitch = max(
                    -90.0,
                    min(90.0, self.pitch + float(action.parameters["pitch"])),
                )
            self._record(
                step_id=step_id,
                action_type=action_type,
                target=None,
                reason="relative_look_applied",
                effect=True,
            )
            return
        if action_type == "move":
            self._record(
                step_id=step_id,
                action_type=action_type,
                target=None,
                reason="bounded_move_observed",
                effect=True,
            )
            return
        if action_type == "equip_item":
            target = action.target
            effect = bool(target in self.inventory and self.inventory[target] > 0)
            if effect:
                self.selected_item = target
            self._record(
                step_id=step_id,
                action_type=action_type,
                target=target,
                reason="equip_ok" if effect else "equip_missing_item",
                effect=effect,
            )
            return
        if action_type == "wait":
            if self.pending_obsidian and self.lava_present and self.water_present:
                self.target_block = "obsidian"
                self.pending_obsidian = False
                self.obsidian_step = step_id
                self.grid_revision += 1
                self._record(
                    step_id=step_id,
                    action_type=action_type,
                    target=None,
                    reason="obsidian_formed",
                    effect=True,
                )
            else:
                self._record(
                    step_id=step_id,
                    action_type=action_type,
                    target=None,
                    reason="wait",
                    effect=False,
                )
            return
        if action_type in {"place_block", "use_item"}:
            self._apply_world_action(action, step_id=step_id)
            return
        self._record(
            step_id=step_id,
            action_type=action_type,
            target=action.target,
            reason="ignored_action",
            effect=False,
        )

    def _apply_world_action(self, action: MacroAction, *, step_id: int) -> None:
        target = action.target
        face = self._expected_face(action.action_type, target)
        if self.failure_mode == "no_world_effect":
            reason = "no_world_effect"
        elif not self.aimed_at(face):
            reason = "not_aimed"
        elif not self.in_range(face):
            reason = "too_far"
        elif not self.valid_face(face):
            reason = "no_valid_face"
        elif target not in {"cobblestone", "lava_bucket", "water_bucket"}:
            reason = "unsupported_item"
        elif self.selected_item != target:
            reason = "wrong_selected_item"
        elif self.inventory.get(target, 0) < 1:
            reason = "missing_item"
        else:
            reason = "effect_applied"
        if reason != "effect_applied":
            self._record(
                step_id=step_id,
                action_type=action.action_type,
                target=target,
                reason=reason,
                effect=False,
                expected_face=face,
            )
            return

        self.inventory[target] -= 1
        self.relevant_action_steps.append(step_id)
        self.grid_revision += 1
        if target == "cobblestone":
            self.support_blocks_placed += 1
            reason = "support_placed"
        elif target == "lava_bucket":
            self.target_block = "lava"
            self.lava_present = True
            self.lava_step = step_id
            reason = "lava_placed"
        else:
            self.water_present = True
            self.water_step = step_id
            self.pending_obsidian = self.lava_present
            reason = "water_placed_adjacent"
        self._record(
            step_id=step_id,
            action_type=action.action_type,
            target=target,
            reason=reason,
            effect=True,
            expected_face=face,
        )

    def build_evaluation_state(
        self,
        *,
        episode_id: str,
        step_id: int,
        max_environment_steps: int,
        max_game_time_seconds: float,
        terminated: bool,
    ) -> CastingEvaluationState:
        """Build independent evaluator-only truth for offline acceptance."""
        transition = None
        if self.obsidian_step is not None:
            transition = CastingTransitionEvidence(
                before_block="water",
                after_block="obsidian",
                update_step=self.obsidian_step,
            )
        return CastingEvaluationState(
            episode_id=episode_id,
            step_id=step_id,
            agent_id="agent_1",
            target_cell=FROZEN_TARGET_CELL,
            initial_target_block="air",
            current_target_block=self.target_block,
            target_update_evidence=transition,
            water_truth=CastingFluidTruth(
                present=self.water_present,
                evidence_step=self.water_step if self.water_present else step_id,
            ),
            lava_truth=CastingFluidTruth(
                present=self.lava_present,
                evidence_step=self.lava_step if self.lava_present else step_id,
            ),
            relevant_action_steps=tuple(self.relevant_action_steps),
            causality_window_steps=4,
            episode_terminated=terminated,
            terminated_step=step_id if terminated else None,
            terminated_reason="driver_done" if terminated else None,
            current_time_seconds=float(step_id),
            max_environment_steps=max_environment_steps,
            max_game_time_seconds=max_game_time_seconds,
        )

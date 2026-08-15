"""Narrow MineRL bridge for the P1 E6 block-placement calibration.

Importing this module does not import MineRL or construct a production
backend. Block truth comes only from the backend-retained evaluator grid
at the frozen E6 world cell, converted to atSpawn grid-local coordinates
inside the backend. The requested ``place_block`` target is never used
as observed world truth.
"""

from __future__ import annotations

from typing import Any, Mapping

from obsidianlink.core.types import BackendStep, MacroAction
from obsidianlink.env.integration.e0_adapter import MineRLE0LifecycleAdapter, public_initial_state
from obsidianlink.env.integration.e6_config import (
    E6_AGENT_ID,
    E6_CALIBRATION_BLOCK,
    E6_DURATION_TICKS,
    E6_TARGET_WORLD_CELL,
    build_e6_compatibility_task,
)
from obsidianlink.env.validation.placement import (
    BlockPlacementTruthSnapshot,
    PlacementActionExecution,
    validate_block_name,
    validate_cell_coordinate,
    validate_target_cell,
)


def _scalar(value: object) -> object:
    shape = getattr(value, "shape", None)
    item = getattr(value, "item", None)
    if shape == () and callable(item):
        return item()
    return value


def block_placement_snapshot(
    value: object,
    *,
    expected_cell: tuple[int, int, int] = E6_TARGET_WORLD_CELL,
) -> BlockPlacementTruthSnapshot:
    """Project an exact backend mapping to typed evaluator-only E6 truth.

    Snapshot ``x,y,z`` are the inspected **world** cell. Grid lookup happens
    inside the backend; this type never stores ObservationFromGrid indices.
    """

    if not isinstance(value, Mapping):
        raise ValueError("block placement truth is missing")
    required = {"episode_id", "agent_id", "step_id", "x", "y", "z", "block"}
    if set(value) != required:
        raise ValueError("block placement truth fields are missing or unknown")
    cell = (
        validate_cell_coordinate(_scalar(value["x"]), "x"),
        validate_cell_coordinate(_scalar(value["y"]), "y"),
        validate_cell_coordinate(_scalar(value["z"]), "z"),
    )
    expected_cell = validate_target_cell(expected_cell)
    if cell != expected_cell:
        raise ValueError("block placement truth cell differs from frozen E6 target")
    return BlockPlacementTruthSnapshot(
        value["episode_id"],
        value["agent_id"],
        value["step_id"],
        cell[0],
        cell[1],
        cell[2],
        validate_block_name(_scalar(value["block"]), "block"),
    )


class MineRLE6PlacementAdapter(MineRLE0LifecycleAdapter):
    """Own one backend and expose exactly the action/truth E6 requires."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tested_action_count = 0

    @staticmethod
    def _build_compatibility_task(episode_id: str) -> object:
        return build_e6_compatibility_task(episode_id)

    def reset(self) -> Mapping[str, dict[str, object]]:
        if not self._opened:
            self.open()
        backend = self._ensure_backend()
        reset = getattr(backend, "reset", None)
        if not callable(reset):
            raise RuntimeError("MineRL backend reset is not callable")
        self._tested_action_count = 0
        raw = reset(self._compatibility_task)
        return public_initial_state(raw, episode_id=self.episode_id)

    def block_placement_truth(self) -> BlockPlacementTruthSnapshot | None:
        backend = self._ensure_backend()
        getter = getattr(backend, "get_block_placement_truth", None)
        if not callable(getter):
            raise RuntimeError("MineRL backend block-placement truth is not callable")
        value = getter(E6_TARGET_WORLD_CELL)
        return None if value is None else block_placement_snapshot(value)

    def reset_failure_audit(self) -> dict[str, int]:
        """Project backend reset counters without exposing MineRL state."""

        backend = self._ensure_backend()
        getter = getattr(backend, "get_reset_audit", None)
        if not callable(getter):
            raise RuntimeError("MineRL backend reset audit is not callable")
        value = getter()
        required = {"reset_attempt_count", "environment_launch_count"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("backend reset audit fields are missing or unknown")
        result: dict[str, int] = {}
        for field_name in sorted(required):
            field_value = value[field_name]
            if type(field_value) is not int or field_value < 0:
                raise ValueError(f"{field_name} must be a non-negative int")
            result[field_name] = field_value
        return result

    def execute_placement_action(self, action: MacroAction) -> PlacementActionExecution:
        if not isinstance(action, MacroAction) or action.action_type != "place_block":
            raise ValueError("E6 tested action must be MacroAction('place_block')")
        if (
            action.target != E6_CALIBRATION_BLOCK
            or action.duration_ticks != E6_DURATION_TICKS
            or dict(action.parameters)
        ):
            raise ValueError("E6 tested place_block differs from frozen calibration")
        self._tested_action_count += 1
        if self._tested_action_count != 1:
            raise RuntimeError("E6 permits exactly one tested placement action")
        backend = self._ensure_backend()
        step = getattr(backend, "step", None)
        if not callable(step):
            raise RuntimeError("MineRL backend step is not callable")
        result = step({E6_AGENT_ID: action})
        if not isinstance(result, BackendStep):
            raise TypeError("MineRL backend step must return BackendStep")
        accepted = result.info.get("translation_accepted")
        if type(accepted) is not bool:
            raise ValueError("translation_accepted must be bool")
        return PlacementActionExecution(
            episode_id=result.episode_id,
            agent_id=E6_AGENT_ID,
            step_id=result.step_id,
            action_type=action.action_type,
            target=action.target,
            duration_ticks=action.duration_ticks,
            translated_action_accepted=accepted,
            tested_action_count=self._tested_action_count,
        )

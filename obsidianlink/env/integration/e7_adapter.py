"""Narrow MineRL bridge for the P1 E7 bucket-usage calibration.

Importing this module does not import MineRL or construct a production
backend. Fluid truth comes only from the backend-retained evaluator grid
at the frozen E7 world cell. The requested ``use_item`` target is never
used as observed world truth. Public inventory is projected through the
existing E2 adapter path.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from obsidianlink.core.types import BackendStep, MacroAction
from obsidianlink.env.integration.e0_adapter import (
    MineRLE0LifecycleAdapter,
    public_initial_state,
)
from obsidianlink.env.integration.e2_adapter import public_inventory_observation
from obsidianlink.env.integration.e7_config import (
    E7_AGENT_ID,
    E7_DURATION_TICKS,
    E7_TARGET_GRID_CELL,
    E7_TARGET_WORLD_CELL,
    build_e7_compatibility_task,
    e7_calibration,
)
from obsidianlink.env.validation.bucket import (
    BucketActionExecution,
    BucketCalibrationVariant,
    BucketFluidTruthSnapshot,
    BucketInventorySnapshot,
    frozen_expected_fluid,
    validate_bucket_variant,
    validate_fluid_class,
)
from obsidianlink.env.validation.placement import validate_cell_coordinate, validate_target_cell


def _scalar(value: object) -> object:
    shape = getattr(value, "shape", None)
    item = getattr(value, "item", None)
    if shape == () and callable(item):
        return item()
    return value


def bucket_inventory_snapshot(value: object) -> BucketInventorySnapshot:
    """Project an exact public inventory mapping to typed E7 inventory truth."""

    if isinstance(value, BucketInventorySnapshot):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("bucket inventory is missing")
    required = {"episode_id", "agent_id", "step_id", "inventory"}
    if not required.issubset(value):
        raise ValueError("bucket inventory fields are missing")
    leaked = [
        key
        for key in value
        if key not in required
    ]
    if leaked:
        raise ValueError("bucket inventory leaked non-inventory fields")
    return BucketInventorySnapshot(
        value["episode_id"],
        value["agent_id"],
        value["step_id"],
        value["inventory"],
    )


def bucket_fluid_snapshot(
    value: object,
    *,
    expected_world_cell: tuple[int, int, int] = E7_TARGET_WORLD_CELL,
    expected_grid_cell: tuple[int, int, int] = E7_TARGET_GRID_CELL,
) -> BucketFluidTruthSnapshot:
    """Project an exact backend mapping to typed evaluator-only E7 fluid truth."""

    if not isinstance(value, Mapping):
        raise ValueError("bucket fluid truth is missing")
    required = {
        "episode_id",
        "agent_id",
        "step_id",
        "world_x",
        "world_y",
        "world_z",
        "grid_x",
        "grid_y",
        "grid_z",
        "fluid",
        "fluid_present",
    }
    if set(value) != required:
        raise ValueError("bucket fluid truth fields are missing or unknown")
    world_cell = (
        validate_cell_coordinate(_scalar(value["world_x"]), "world_x"),
        validate_cell_coordinate(_scalar(value["world_y"]), "world_y"),
        validate_cell_coordinate(_scalar(value["world_z"]), "world_z"),
    )
    grid_cell = (
        validate_cell_coordinate(_scalar(value["grid_x"]), "grid_x"),
        validate_cell_coordinate(_scalar(value["grid_y"]), "grid_y"),
        validate_cell_coordinate(_scalar(value["grid_z"]), "grid_z"),
    )
    expected_world_cell = validate_target_cell(expected_world_cell, "target_world_cell")
    expected_grid_cell = validate_target_cell(expected_grid_cell, "target_grid_cell")
    if world_cell != expected_world_cell:
        raise ValueError("bucket fluid truth world cell differs from frozen E7 target")
    if grid_cell != expected_grid_cell:
        raise ValueError("bucket fluid truth grid cell differs from frozen E7 target")
    fluid = validate_fluid_class(_scalar(value["fluid"]), "fluid")
    present = _scalar(value["fluid_present"])
    if type(present) is not bool:
        raise ValueError("fluid_present must be bool")
    return BucketFluidTruthSnapshot(
        value["episode_id"],
        value["agent_id"],
        value["step_id"],
        world_cell[0],
        world_cell[1],
        world_cell[2],
        grid_cell[0],
        grid_cell[1],
        grid_cell[2],
        fluid,
        present,
    )


class MineRLE7BucketAdapter(MineRLE0LifecycleAdapter):
    """Own one backend and expose exactly the action/truth E7 requires."""

    def __init__(self, *, variant: object = BucketCalibrationVariant.WATER, **kwargs: Any) -> None:
        self._variant = validate_bucket_variant(variant)
        self._tested_action_count = 0
        self._latest_public: Mapping[str, dict[str, object]] | None = None
        self._latest_selected: str | None = None
        super().__init__(**kwargs)

    def _build_compatibility_task(self, episode_id: str) -> object:
        return build_e7_compatibility_task(episode_id, self._variant)

    @property
    def variant(self) -> BucketCalibrationVariant:
        return self._variant

    def _capture_public(self, raw: object) -> None:
        self._latest_public = public_inventory_observation(raw, episode_id=self.episode_id)
        selected = None
        if isinstance(raw, Mapping):
            value = raw.get(E7_AGENT_ID)
            selected = getattr(value, "selected_item", None)
            if selected is None and isinstance(value, Mapping):
                selected = value.get("selected_item")
        self._latest_selected = (
            selected.strip() if isinstance(selected, str) and selected.strip() else None
        )

    def reset(self) -> Mapping[str, dict[str, object]]:
        if not self._opened:
            self.open()
        backend = self._ensure_backend()
        reset = getattr(backend, "reset", None)
        if not callable(reset):
            raise RuntimeError("MineRL backend reset is not callable")
        self._tested_action_count = 0
        raw = reset(self._compatibility_task)
        self._capture_public(raw)
        return public_initial_state(raw, episode_id=self.episode_id)

    def public_bucket_inventory(self) -> BucketInventorySnapshot | None:
        if self._latest_public is None:
            return None
        payload = self._latest_public.get(E7_AGENT_ID)
        if payload is None or "inventory" not in payload:
            return None
        return bucket_inventory_snapshot(payload)

    def public_bucket_selected_item(self) -> str | None:
        return self._latest_selected

    def bucket_fluid_truth(self) -> BucketFluidTruthSnapshot | None:
        backend = self._ensure_backend()
        getter = getattr(backend, "get_bucket_fluid_truth", None)
        if not callable(getter):
            raise RuntimeError("MineRL backend bucket-fluid truth is not callable")
        value = getter(E7_TARGET_WORLD_CELL)
        return None if value is None else bucket_fluid_snapshot(value)

    def reset_failure_audit(self) -> dict[str, int]:
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

    def execute_bucket_action(self, action: MacroAction) -> BucketActionExecution:
        calibration = e7_calibration(self._variant)
        if not isinstance(action, MacroAction) or action.action_type != "use_item":
            raise ValueError("E7 tested action must be MacroAction('use_item')")
        if (
            action.target != calibration.bucket_item
            or action.duration_ticks != E7_DURATION_TICKS
            or dict(action.parameters)
        ):
            raise ValueError("E7 tested use_item differs from frozen calibration")
        self._tested_action_count += 1
        if self._tested_action_count != 1:
            raise RuntimeError("E7 permits exactly one tested bucket action")
        backend = self._ensure_backend()
        step = getattr(backend, "step", None)
        if not callable(step):
            raise RuntimeError("MineRL backend step is not callable")
        result = step({E7_AGENT_ID: action})
        if not isinstance(result, BackendStep):
            raise TypeError("MineRL backend step must return BackendStep")
        self._capture_public(result.observations)
        accepted = result.info.get("translation_accepted")
        if type(accepted) is not bool:
            raise ValueError("translation_accepted must be bool")
        return BucketActionExecution(
            episode_id=result.episode_id,
            agent_id=E7_AGENT_ID,
            step_id=result.step_id,
            action_type=action.action_type,
            target=action.target,
            duration_ticks=action.duration_ticks,
            translated_action_accepted=accepted,
            tested_action_count=self._tested_action_count,
            variant=self._variant.value,
            expected_fluid=frozen_expected_fluid(self._variant),
        )

    @classmethod
    def lifecycle_factory(
        cls,
        *,
        episode_id: str,
        variant: object = BucketCalibrationVariant.WATER,
        backend_cls: type | None = None,
        backend_kwargs: Mapping[str, Any] | None = None,
    ) -> Callable[[], "MineRLE7BucketAdapter"]:
        def factory() -> MineRLE7BucketAdapter:
            return cls(
                episode_id=episode_id,
                variant=variant,
                backend_cls=backend_cls,
                backend_kwargs=backend_kwargs,
            )

        return factory

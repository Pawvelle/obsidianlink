"""Narrow MineRL bridge for the P1 E9 server-side fluid-truth calibration.

Importing this module does not import MineRL or construct a production
backend. Region fluid truth comes only from the backend-retained evaluator
grid. The requested ``use_item`` target is never used as observed world
truth. ServerTruthSnapshot never enters Observation, prompt, memory, or
shared agent state.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from obsidianlink.core.types import BackendStep, MacroAction
from obsidianlink.env.integration.e0_adapter import MineRLE0LifecycleAdapter, public_initial_state
from obsidianlink.env.integration.e8_adapter import server_truth_snapshot
from obsidianlink.env.integration.e9_config import (
    E9_AGENT_ID,
    E9_DURATION_TICKS,
    E9_PROBE_WORLD_CELLS,
    E9_STIMULUS_ACTION_TYPE,
    build_e9_compatibility_task,
    e9_calibration,
)
from obsidianlink.env.validation.truth import (
    EVALUATOR_TRUTH_LEAK_KEYS,
    FluidCalibrationVariant,
    FluidTruthActionExecution,
    ServerTruthSnapshot,
    validate_fluid_variant,
)


class MineRLE9FluidTruthAdapter(MineRLE0LifecycleAdapter):
    """Own one backend and expose exactly the action/truth E9 requires."""

    def __init__(self, *, variant: object = FluidCalibrationVariant.WATER, **kwargs: Any) -> None:
        self._variant = validate_fluid_variant(variant)
        self._tested_action_count = 0
        super().__init__(**kwargs)

    def _build_compatibility_task(self, episode_id: str) -> object:
        return build_e9_compatibility_task(episode_id, self._variant)

    @property
    def variant(self) -> FluidCalibrationVariant:
        return self._variant

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

    def server_truth_snapshot(self) -> ServerTruthSnapshot | None:
        backend = self._ensure_backend()
        getter = getattr(backend, "get_server_truth_snapshot", None)
        if not callable(getter):
            raise RuntimeError("MineRL backend server-truth snapshot is not callable")
        value = getter(E9_PROBE_WORLD_CELLS)
        return None if value is None else server_truth_snapshot(
            value, expected_cells=E9_PROBE_WORLD_CELLS
        )

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

    def execute_fluid_stimulus(self, action: MacroAction) -> FluidTruthActionExecution:
        calibration = e9_calibration(self._variant)
        if not isinstance(action, MacroAction) or action.action_type != E9_STIMULUS_ACTION_TYPE:
            raise ValueError("E9 stimulus must be MacroAction('use_item')")
        if (
            action.target != calibration.bucket_item
            or action.duration_ticks != E9_DURATION_TICKS
            or dict(action.parameters)
        ):
            raise ValueError("E9 stimulus differs from frozen calibration")
        self._tested_action_count += 1
        if self._tested_action_count != 1:
            raise RuntimeError("E9 permits exactly one stimulus action")
        backend = self._ensure_backend()
        step = getattr(backend, "step", None)
        if not callable(step):
            raise RuntimeError("MineRL backend step is not callable")
        result = step({E9_AGENT_ID: action})
        if not isinstance(result, BackendStep):
            raise TypeError("MineRL backend step must return BackendStep")
        accepted = result.info.get("translation_accepted")
        if type(accepted) is not bool:
            raise ValueError("translation_accepted must be bool")
        leaked = sorted(key for key in result.info if key in EVALUATOR_TRUTH_LEAK_KEYS)
        if leaked:
            raise ValueError("E9 backend info leaked evaluator truth: " + ", ".join(leaked))
        return FluidTruthActionExecution(
            episode_id=result.episode_id,
            agent_id=E9_AGENT_ID,
            step_id=result.step_id,
            action_type=action.action_type,
            target=action.target,
            duration_ticks=action.duration_ticks,
            translated_action_accepted=accepted,
            tested_action_count=self._tested_action_count,
            variant=self._variant.value,
        )

    @classmethod
    def lifecycle_factory(
        cls,
        *,
        episode_id: str,
        variant: object = FluidCalibrationVariant.WATER,
        backend_cls: type | None = None,
        backend_kwargs: Mapping[str, Any] | None = None,
    ) -> Callable[[], "MineRLE9FluidTruthAdapter"]:
        def factory() -> MineRLE9FluidTruthAdapter:
            return cls(
                episode_id=episode_id,
                variant=variant,
                backend_cls=backend_cls,
                backend_kwargs=backend_kwargs,
            )

        return factory

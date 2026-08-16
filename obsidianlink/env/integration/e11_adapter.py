"""Narrow MineRL bridge for the P1 E11 vanilla portal-activation calibration.

Importing this module does not import MineRL or construct a production
backend. Activation truth comes only from the backend-retained evaluator
grid. RGB is never the success source. ServerTruthSnapshot never enters
Observation, prompt, memory, or shared agent state.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from obsidianlink.core.types import BackendStep, MacroAction
from obsidianlink.env.integration.e0_adapter import MineRLE0LifecycleAdapter, public_initial_state
from obsidianlink.env.integration.e8_adapter import server_truth_snapshot
from obsidianlink.env.integration.e11_config import (
    E11_AGENT_ID,
    E11_DURATION_TICKS,
    E11_PROBE_WORLD_CELLS,
    E11_STIMULUS_ACTION_TYPE,
    E11_STIMULUS_ITEM_NAME,
    build_e11_compatibility_task,
)
from obsidianlink.env.validation.truth import (
    EVALUATOR_TRUTH_LEAK_KEYS,
    PortalActivationActionExecution,
    ServerTruthSnapshot,
)


class MineRLE11PortalActivationAdapter(MineRLE0LifecycleAdapter):
    """Own one backend and expose exactly the action/truth E11 requires."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tested_action_count = 0
        self._observation_wait_count = 0

    @staticmethod
    def _build_compatibility_task(episode_id: str) -> object:
        return build_e11_compatibility_task(episode_id)

    def reset(self) -> Mapping[str, dict[str, object]]:
        if not self._opened:
            self.open()
        backend = self._ensure_backend()
        reset = getattr(backend, "reset", None)
        if not callable(reset):
            raise RuntimeError("MineRL backend reset is not callable")
        self._tested_action_count = 0
        self._observation_wait_count = 0
        raw = reset(self._compatibility_task)
        return public_initial_state(raw, episode_id=self.episode_id)

    def server_truth_snapshot(self) -> ServerTruthSnapshot | None:
        backend = self._ensure_backend()
        getter = getattr(backend, "get_server_truth_snapshot", None)
        if not callable(getter):
            raise RuntimeError("MineRL backend server-truth snapshot is not callable")
        value = getter(E11_PROBE_WORLD_CELLS)
        return None if value is None else server_truth_snapshot(
            value, expected_cells=E11_PROBE_WORLD_CELLS
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

    def execute_activation_stimulus(self, action: MacroAction) -> PortalActivationActionExecution:
        if not isinstance(action, MacroAction) or action.action_type != E11_STIMULUS_ACTION_TYPE:
            raise ValueError("E11 stimulus must be MacroAction('use_item')")
        if (
            action.target != E11_STIMULUS_ITEM_NAME
            or action.duration_ticks != E11_DURATION_TICKS
            or dict(action.parameters)
        ):
            raise ValueError("E11 stimulus differs from frozen calibration")
        self._tested_action_count += 1
        if self._tested_action_count != 1:
            raise RuntimeError("E11 permits exactly one stimulus action")
        backend = self._ensure_backend()
        step = getattr(backend, "step", None)
        if not callable(step):
            raise RuntimeError("MineRL backend step is not callable")
        result = step({E11_AGENT_ID: action})
        if not isinstance(result, BackendStep):
            raise TypeError("MineRL backend step must return BackendStep")
        accepted = result.info.get("translation_accepted")
        if type(accepted) is not bool:
            raise ValueError("translation_accepted must be bool")
        leaked = sorted(key for key in result.info if key in EVALUATOR_TRUTH_LEAK_KEYS)
        if leaked:
            raise ValueError("E11 backend info leaked evaluator truth: " + ", ".join(leaked))
        return PortalActivationActionExecution(
            episode_id=result.episode_id,
            agent_id=E11_AGENT_ID,
            step_id=result.step_id,
            action_type=action.action_type,
            target=action.target,
            duration_ticks=action.duration_ticks,
            translated_action_accepted=accepted,
            tested_action_count=self._tested_action_count,
            observation_wait_count=self._observation_wait_count,
        )

    def observe_wait(self) -> None:
        """Advance one evaluator tick without counting a second tested action."""

        if self._tested_action_count != 1:
            raise RuntimeError("E11 observation waits require the stimulus to have run")
        backend = self._ensure_backend()
        step = getattr(backend, "step", None)
        if not callable(step):
            raise RuntimeError("MineRL backend step is not callable")
        result = step({E11_AGENT_ID: MacroAction.wait()})
        if not isinstance(result, BackendStep):
            raise TypeError("MineRL backend step must return BackendStep")
        leaked = sorted(key for key in result.info if key in EVALUATOR_TRUTH_LEAK_KEYS)
        if leaked:
            raise ValueError("E11 wait info leaked evaluator truth: " + ", ".join(leaked))
        self._observation_wait_count += 1

    @classmethod
    def lifecycle_factory(
        cls,
        *,
        episode_id: str,
        backend_cls: type | None = None,
        backend_kwargs: Mapping[str, Any] | None = None,
    ) -> Callable[[], "MineRLE11PortalActivationAdapter"]:
        def factory() -> MineRLE11PortalActivationAdapter:
            return cls(
                episode_id=episode_id,
                backend_cls=backend_cls,
                backend_kwargs=backend_kwargs,
            )

        return factory

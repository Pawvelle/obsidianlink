from __future__ import annotations

import time
from typing import Mapping

from obsidianlink.core.types import BackendStep, MacroAction, Observation, TaskInstance
from obsidianlink.env.capabilities import (
    BackendCapabilities,
    assert_backend_can_start_task,
)
from obsidianlink.evaluation.casting import CastingEvaluationState
from obsidianlink.evaluation.casting_frame_evaluator import (
    FrozenFrameEvaluationState,
)
from obsidianlink.evaluation.casting_ignition_evaluator import (
    FrozenIgnitionEvaluationState,
)
from obsidianlink.evaluation.casting_nether_entry_evaluator import (
    FrozenNetherEntryEvaluationState,
)
from obsidianlink.evaluation.continuous_casting import (
    ContinuousCastingEvaluationState,
)
from obsidianlink.evaluation.portal import EvaluationState


class FakeEnvironmentBackend:
    """Standard-library backend used to validate contracts without Minecraft.

    The backend carries a :class:`BackendCapabilities` manifest that
    describes which casting-c1 features it claims to support. By
    default the manifest reports :meth:`BackendCapabilities.full` so
    the standard fake backend acts as the "complete" half of the
    positive / negative test pair. Tests that need to exercise the
    "missing capability" half must build the instance via
    :meth:`with_capabilities` (or pass a custom manifest to the
    constructor).

    The manifest is purely a declaration. It does not change the
    backend's :meth:`step` behaviour: the fake backend still ignores
    every action and never simulates water, lava, or obsidian
    transitions. Tests use the manifest together with
    :func:`assert_casting_c1_capabilities` to assert that the
    pre-episode gate fails closed when the manifest is incomplete.
    """

    def __init__(
        self,
        capabilities: BackendCapabilities | None = None,
    ) -> None:
        if capabilities is None:
            capabilities = BackendCapabilities.full()
        if not isinstance(capabilities, BackendCapabilities):
            raise ValueError("capabilities must be a BackendCapabilities instance")
        self._capabilities: BackendCapabilities = capabilities
        self._opened = False
        self._task: TaskInstance | None = None
        self._step_id = 0
        self._evaluation_state: EvaluationState | None = None
        # Casting evaluator-only truth lives on a *separate* slot so
        # the casting evaluator never reads legacy Portal state and
        # the Portal evaluator never reads casting truth. R3 keeps
        # both surfaces frozen and strictly identity-guarded.
        self._casting_evaluation_state: CastingEvaluationState | None = None
        # R5 continuous-casting truth lives on a *third* slot so the
        # single-cell R3 surface and the multi-cell R5 surface never
        # cross-contaminate. The driver / orchestrator contract
        # guarantees that the single-cell slot is only used for
        # ``casting_c1_fixed`` and the multi-cell slot only for
        # ``casting_c3_fixed``.
        self._continuous_casting_evaluation_state: ContinuousCastingEvaluationState | None = None
        # R6 Casting-S-C3 frozen-frame truth lives on a *fourth* slot
        # so the C2 multi-cell surface and the C3 frame surface never
        # cross-contaminate. The driver / orchestrator contract
        # guarantees that the multi-cell slot is only used for
        # ``casting_c3_fixed`` (C2) and the frame slot only for
        # ``casting_s_c3_fixed`` (C3). The frame slot is the
        # evaluator-only truth path required by R6-C3-FRAME-EVALUATOR.
        self._frame_evaluation_state: FrozenFrameEvaluationState | None = None
        # R6 Casting-S-C4 ignition truth lives on a *fifth* slot so
        # the C3 frame surface and the C4 ignition surface never
        # cross-contaminate. The C4 state itself embeds the C3 frame
        # state, but it still rides on its own backend slot so the
        # FakeBackend can validate ``workflow`` / ``episode_id`` /
        # ``step_id`` / ``agent_id`` independently of the C3 surface.
        # This slot is the evaluator-only truth path required by
        # R6-C4-IGNITION-EVALUATOR.
        self._ignition_evaluation_state: FrozenIgnitionEvaluationState | None = None
        # R6 Casting-S-C5 entry truth is isolated from every lower-level
        # surface.  Although it embeds C4 truth, only the C5 workflow may
        # inject or retrieve this sixth evaluator-only slot.
        self._nether_entry_evaluation_state: FrozenNetherEntryEvaluationState | None = None

    @classmethod
    def with_capabilities(
        cls, capabilities: BackendCapabilities
    ) -> "FakeEnvironmentBackend":
        """Build a ``FakeEnvironmentBackend`` that declares a custom manifest.

        Used by tests to exercise the "missing capability" half of
        the positive / negative pair without mutating the shared
        default.
        """
        return cls(capabilities=capabilities)

    def capabilities(self) -> BackendCapabilities:
        """Return the immutable manifest this backend declares."""
        return self._capabilities

    @property
    def agent_ids(self) -> tuple[str, ...]:
        if self._task is None:
            return ()
        return self._task.agent_ids

    def open(self) -> None:
        if self._opened:
            raise RuntimeError("backend is already open")
        self._opened = True

    def reset(self, task: TaskInstance) -> Mapping[str, Observation]:
        self._require_open()
        # Pre-episode capability gate. Runs *before* ``self._task`` is
        # set, *before* the evaluation baseline is constructed, and
        # *before* any observation is generated, so a backend whose
        # casting-c1 manifest is incomplete fails closed before the
        # fake runtime has produced anything. Non-casting workflows
        # are not gated by this check, so the legacy Route A0
        # contract continues to work.
        assert_backend_can_start_task(self, task)
        self._task = task
        self._step_id = 0
        self._evaluation_state = EvaluationState(
            episode_id=task.task_id,
            step_id=0,
        )
        # Casting truth is reset on every episode boundary; tests
        # inject a fresh state via ``set_casting_evaluation_state``
        # after each step.
        self._casting_evaluation_state = None
        # Continuous casting truth follows the same rule: cleared
        # on every reset so a stale read fails closed.
        self._continuous_casting_evaluation_state = None
        # R6 C3 frame truth follows the same rule.
        self._frame_evaluation_state = None
        # R6 C4 ignition truth follows the same rule.
        self._ignition_evaluation_state = None
        # R6 C5 Nether-entry truth follows the same rule.
        self._nether_entry_evaluation_state = None
        return self._observations()

    def step(self, actions: Mapping[str, MacroAction]) -> BackendStep:
        task = self._require_task()
        if set(actions) != set(task.agent_ids):
            raise ValueError("actions must contain every task agent exactly once")
        if any(not isinstance(action, MacroAction) for action in actions.values()):
            raise ValueError("actions must contain MacroAction values")
        self._step_id += 1
        if self._evaluation_state is not None:
            state = self._evaluation_state
            self._evaluation_state = EvaluationState(
                episode_id=state.episode_id,
                step_id=self._step_id,
                portal_built_by_episode=state.portal_built_by_episode,
                valid_portal_frame=state.valid_portal_frame,
                portal_activated=state.portal_activated,
                agents_in_nether=state.agents_in_nether,
                evidence=state.evidence,
            )
        # Casting truth must be re-injected for the new step; the
        # previous state no longer matches the new ``step_id``. We
        # clear it so a stale read fails closed instead of being
        # silently accepted at the wrong step.
        self._casting_evaluation_state = None
        # Continuous casting truth follows the same rule.
        self._continuous_casting_evaluation_state = None
        # R6 C3 frame truth follows the same rule.
        self._frame_evaluation_state = None
        # R6 C4 ignition truth follows the same rule.
        self._ignition_evaluation_state = None
        # R6 C5 Nether-entry truth follows the same rule.
        self._nether_entry_evaluation_state = None
        return BackendStep(
            episode_id=task.task_id,
            step_id=self._step_id,
            observations=self._observations(),
            rewards={agent_id: 0.0 for agent_id in task.agent_ids},
            terminated=False,
            truncated=False,
            info={"backend": "fake"},
        )

    def set_evaluation_state(self, state: EvaluationState) -> None:
        task = self._require_task()
        if state.episode_id != task.task_id:
            raise ValueError("evaluation state episode_id must match current task")
        if state.step_id != self._step_id:
            raise ValueError("evaluation state step_id must match current backend step")
        self._evaluation_state = state

    def get_evaluation_state(self) -> EvaluationState:
        self._require_task()
        if self._evaluation_state is None:
            raise RuntimeError("evaluation state is unavailable")
        return self._evaluation_state

    def set_casting_evaluation_state(
        self, state: CastingEvaluationState
    ) -> None:
        """Inject evaluator-only casting truth for the current episode.

        The state is stored on a *separate* slot from the legacy
        Portal :class:`EvaluationState`; the casting evaluator must
        never see Portal geometry and the Portal evaluator must
        never see casting truth. The two surfaces share identity
        guards (``episode_id`` matches the current task and
        ``step_id`` matches the current backend step) and an open
        / reset requirement.

        The fake backend never copies casting truth into an
        :class:`Observation`, so this method is safe for the
        information-isolation contract.
        """
        task = self._require_task()
        if not isinstance(state, CastingEvaluationState):
            raise TypeError(
                "casting evaluation state must be a CastingEvaluationState, "
                f"got {type(state).__name__}"
            )
        if state.episode_id != task.task_id:
            raise ValueError(
                "casting evaluation state episode_id must match current task"
            )
        if state.step_id != self._step_id:
            raise ValueError(
                "casting evaluation state step_id must match current backend step"
            )
        self._casting_evaluation_state = state

    def get_casting_evaluation_state(self) -> CastingEvaluationState:
        """Return the previously injected casting truth, or raise.

        The fake backend refuses to fabricate casting truth itself
        (it never reads MineRL grid state). If nothing has been
        injected via :meth:`set_casting_evaluation_state`, this
        method raises so callers do not silently receive a
        fabricated state. Reading before :meth:`reset` is treated
        as a programming error.
        """
        self._require_task()
        if self._casting_evaluation_state is None:
            raise RuntimeError("casting evaluation state is unavailable")
        return self._casting_evaluation_state

    def set_continuous_casting_evaluation_state(
        self, state: ContinuousCastingEvaluationState
    ) -> None:
        """Inject evaluator-only R5 continuous-casting truth.

        Mirrors :meth:`set_casting_evaluation_state` for the R5
        multi-cell surface. The state lives on a separate slot so
        the single-cell R3 surface and the multi-cell R5 surface
        never cross-contaminate. The fake backend never copies
        truth into an :class:`Observation`.
        """
        task = self._require_task()
        if not isinstance(state, ContinuousCastingEvaluationState):
            raise TypeError(
                "continuous casting evaluation state must be a "
                "ContinuousCastingEvaluationState, "
                f"got {type(state).__name__}"
            )
        if state.episode_id != task.task_id:
            raise ValueError(
                "continuous casting evaluation state episode_id must match "
                "current task"
            )
        if state.step_id != self._step_id:
            raise ValueError(
                "continuous casting evaluation state step_id must match "
                "current backend step"
            )
        self._continuous_casting_evaluation_state = state

    def get_continuous_casting_evaluation_state(
        self,
    ) -> ContinuousCastingEvaluationState:
        """Return the previously injected R5 continuous-casting truth."""
        self._require_task()
        if self._continuous_casting_evaluation_state is None:
            raise RuntimeError(
                "continuous casting evaluation state is unavailable"
            )
        return self._continuous_casting_evaluation_state

    def set_frame_evaluation_state(
        self, state: FrozenFrameEvaluationState
    ) -> None:
        """Inject evaluator-only R6 Casting-S-C3 frame truth.

        Mirrors :meth:`set_casting_evaluation_state` and
        :meth:`set_continuous_casting_evaluation_state` for the R6
        C3 frozen-frame surface. The state lives on a separate slot
        so the C2 multi-cell surface and the C3 frame surface never
        cross-contaminate. The fake backend never copies truth into
        an :class:`Observation`.

        The state must match the current ``task_id`` and the
        current backend ``step_id``; ``agent_id`` must be in the task's
        ``agent_ids``. A wrong workflow / episode / step / agent fails
        closed by raising.
        """
        task = self._require_task()
        if not isinstance(state, FrozenFrameEvaluationState):
            raise TypeError(
                "frame evaluation state must be a "
                "FrozenFrameEvaluationState, "
                f"got {type(state).__name__}"
            )
        if task.workflow != "casting_s_c3_fixed":
            raise ValueError(
                "frame evaluation state requires casting_s_c3_fixed workflow"
            )
        if state.episode_id != task.task_id:
            raise ValueError(
                "frame evaluation state episode_id must match current task"
            )
        if state.step_id != self._step_id:
            raise ValueError(
                "frame evaluation state step_id must match current backend step"
            )
        if state.agent_id not in task.agent_ids:
            raise ValueError(
                "frame evaluation state agent_id must be in task.agent_ids"
            )
        self._frame_evaluation_state = state

    def get_frame_evaluation_state(self) -> FrozenFrameEvaluationState:
        """Return the previously injected R6 C3 frame truth."""
        self._require_task()
        if self._frame_evaluation_state is None:
            raise RuntimeError("frame evaluation state is unavailable")
        return self._frame_evaluation_state

    def clear_frame_evaluation_state(self) -> None:
        """Drop the R6 C3 frame truth slot.

        Equivalent to ``reset``/``step``/``close`` clearing; exposed
        so tests can prove the cleanup contract explicitly.
        """
        self._frame_evaluation_state = None

    def set_ignition_evaluation_state(
        self, state: FrozenIgnitionEvaluationState
    ) -> None:
        """Inject evaluator-only R6 Casting-S-C4 ignition truth.

        Mirrors :meth:`set_frame_evaluation_state` for the R6 C4
        ignition surface. The state lives on a separate slot so the
        C3 frame surface and the C4 ignition surface never
        cross-contaminate even though the C4 state embeds the C3
        frame state. The fake backend never copies truth into an
        :class:`Observation`.

        The state must match the current ``task_id``, the current
        backend ``step_id``, and ``task.workflow`` must be
        ``casting_s_c4_fixed``. ``agent_id`` must be in the task's
        ``agent_ids``. A wrong workflow / episode / step / agent
        fails closed by raising.
        """
        task = self._require_task()
        if not isinstance(state, FrozenIgnitionEvaluationState):
            raise TypeError(
                "ignition evaluation state must be a "
                "FrozenIgnitionEvaluationState, "
                f"got {type(state).__name__}"
            )
        if task.workflow != "casting_s_c4_fixed":
            raise ValueError(
                "ignition evaluation state requires casting_s_c4_fixed "
                "workflow"
            )
        if state.episode_id != task.task_id:
            raise ValueError(
                "ignition evaluation state episode_id must match current task"
            )
        if state.step_id != self._step_id:
            raise ValueError(
                "ignition evaluation state step_id must match current backend "
                "step"
            )
        if state.agent_id not in task.agent_ids:
            raise ValueError(
                "ignition evaluation state agent_id must be in task.agent_ids"
            )
        self._ignition_evaluation_state = state

    def get_ignition_evaluation_state(self) -> FrozenIgnitionEvaluationState:
        """Return the previously injected R6 C4 ignition truth."""
        self._require_task()
        if self._ignition_evaluation_state is None:
            raise RuntimeError("ignition evaluation state is unavailable")
        return self._ignition_evaluation_state

    def clear_ignition_evaluation_state(self) -> None:
        """Drop the R6 C4 ignition truth slot.

        Equivalent to ``reset``/``step``/``close`` clearing; exposed
        so tests can prove the cleanup contract explicitly.
        """
        self._ignition_evaluation_state = None

    def set_nether_entry_evaluation_state(
        self, state: FrozenNetherEntryEvaluationState
    ) -> None:
        """Inject identity-guarded evaluator-only C5 transition truth."""
        task = self._require_task()
        if not isinstance(state, FrozenNetherEntryEvaluationState):
            raise TypeError(
                "nether entry evaluation state must be a "
                "FrozenNetherEntryEvaluationState, "
                f"got {type(state).__name__}"
            )
        if task.workflow != "casting_s_c5_fixed":
            raise ValueError(
                "nether entry evaluation state requires casting_s_c5_fixed "
                "workflow"
            )
        if state.episode_id != task.task_id:
            raise ValueError(
                "nether entry evaluation state episode_id must match current task"
            )
        if state.step_id != self._step_id:
            raise ValueError(
                "nether entry evaluation state step_id must match current backend step"
            )
        if state.agent_id not in task.agent_ids:
            raise ValueError(
                "nether entry evaluation state agent_id must be in task.agent_ids"
            )
        self._nether_entry_evaluation_state = state

    def get_nether_entry_evaluation_state(
        self,
    ) -> FrozenNetherEntryEvaluationState:
        """Return injected C5 truth, or fail closed when unavailable."""
        self._require_task()
        if self._nether_entry_evaluation_state is None:
            raise RuntimeError("nether entry evaluation state is unavailable")
        return self._nether_entry_evaluation_state

    def clear_nether_entry_evaluation_state(self) -> None:
        """Drop the C5 evaluator-only truth slot."""
        self._nether_entry_evaluation_state = None

    def close(self) -> None:
        self._opened = False
        self._task = None
        self._step_id = 0
        self._evaluation_state = None
        self._casting_evaluation_state = None
        self._continuous_casting_evaluation_state = None
        self._frame_evaluation_state = None
        self._ignition_evaluation_state = None
        self._nether_entry_evaluation_state = None

    def _require_open(self) -> None:
        if not self._opened:
            raise RuntimeError("backend is not open")

    def _require_task(self) -> TaskInstance:
        self._require_open()
        if self._task is None:
            raise RuntimeError("backend has not been reset")
        return self._task

    def _observations(self) -> Mapping[str, Observation]:
        task = self._require_task()
        timestamp = time.time()
        return {
            agent_id: Observation(
                episode_id=task.task_id,
                agent_id=agent_id,
                step_id=self._step_id,
                timestamp=timestamp,
                frame={"backend": "fake", "step_id": self._step_id},
                visible_inventory=task.initial_inventories[agent_id],
                workflow_stage=task.workflow,
            )
            for agent_id in task.agent_ids
        }

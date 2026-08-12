from __future__ import annotations

import threading
import time
from collections import Counter
from typing import Any, Callable, Mapping

import numpy as np

from obsidianlink.actions.minerl_translator import (
    PORTAL_A0_HOTBAR,
    TRANSLATOR_EQUIPPABLE_ITEMS,
    TRANSLATOR_PLACEABLE_ITEMS,
    build_hotbar_mapping,
    translate_macro_action,
)
from obsidianlink.core.types import BackendStep, MacroAction, Observation, TaskInstance
from obsidianlink.env.capabilities import (
    BackendCapabilities,
    assert_backend_can_start_task,
)
from obsidianlink.env.portal_spec import (
    PORTAL_GRID_BLOCKS,
    PORTAL_GRID_MAX,
    PORTAL_GRID_MIN,
    PORTAL_GRID_MISSING_ID,
    PORTAL_GRID_SIZE,
    PORTAL_SELECTABLE_ITEMS,
    PORTAL_SELECTED_ITEM_NAME,
    PortalA0EnvSpec,
    PortalGridObservation,
)
from obsidianlink.evaluation import EvaluationState
from obsidianlink.evaluation.casting import (
    CastingFluidTruth,
    CastingTransitionEvidence,
)
from obsidianlink.evaluation.casting_frame_evaluator import (
    CASTING_S_C3_AGENT_ID,
    CASTING_S_C3_FRAME_CELLS,
    CASTING_S_C3_INTERIOR_CELLS,
    CASTING_S_C3_REQUIRED_CELL_COUNT,
    FrozenFrameActionEvidence,
    FrozenFrameCellTruth,
    FrozenFrameEvaluationState,
    FrozenFrameInteriorCellTruth,
)
from obsidianlink.evaluation.casting_ignition_evaluator import (
    CASTING_S_C4_AGENT_ID,
    FrozenFrameIdentity,
    FrozenIgnitionEvaluationState,
    IgnitionActionEvidence,
    PortalActivationEvidence,
    build_c4_c3_frame_identity,
)
from obsidianlink.evaluation.casting_nether_entry_evaluator import (
    CASTING_S_C5_AGENT_ID,
    CASTING_S_C5_SOURCE_DIMENSION,
    CASTING_S_C5_TARGET_DIMENSION,
    FrozenNetherEntryEvaluationState,
    NetherEntryEvidence,
)
from obsidianlink.evaluation.continuous_casting import (
    CASTING_C3_TARGET_CELLS,
    ContinuousCastingCellTruth,
    ContinuousCastingEvaluationState,
)
from obsidianlink.evaluation.casting import CastingEvaluationState
from obsidianlink.evaluation.frame_geometry import (
    CellOffset,
    detect_portal_frame_from_int_grid,
)


EnvFactory = Callable[[TaskInstance], Any]


BLOCK_ID_TO_NAME: dict[int, str] = {
    index: name for index, name in enumerate(PORTAL_GRID_BLOCKS)
}

OBSIDIAN_ID = PORTAL_GRID_BLOCKS.index("obsidian")
NETHER_PORTAL_ID = PORTAL_GRID_BLOCKS.index("nether_portal")
MISSING_ID = PORTAL_GRID_BLOCKS.index("missing")

FLUID_AIR_ID = PORTAL_GRID_BLOCKS.index("air")
FLUID_WATER_ID = PORTAL_GRID_BLOCKS.index("water")
FLUID_FLOWING_WATER_ID = PORTAL_GRID_BLOCKS.index("flowing_water")
FLUID_LAVA_ID = PORTAL_GRID_BLOCKS.index("lava")
FLUID_FLOWING_LAVA_ID = PORTAL_GRID_BLOCKS.index("flowing_lava")
FLUID_OTHER_ID = PORTAL_GRID_BLOCKS.index("other")
FLUID_MISSING_ID = PORTAL_GRID_MISSING_ID

#: A fluid block id that counts as a *positive* water observation.
FLUID_WATER_IDS: frozenset[int] = frozenset(
    {FLUID_WATER_ID, FLUID_FLOWING_WATER_ID}
)
#: A fluid block id that counts as a *positive* lava observation.
FLUID_LAVA_IDS: frozenset[int] = frozenset(
    {FLUID_LAVA_ID, FLUID_FLOWING_LAVA_ID}
)
#: A fluid block id that counts as a *negative* water / lava
#: observation.
FLUID_AIR_IDS: frozenset[int] = frozenset({FLUID_AIR_ID})

#: Maximum number of recent use_item / place_block / equip actions
#: the backend retains for per-cell attribution. The C3 / C4 / C5
#: driver plans are < 800 steps and emit at most two cast actions
#: per cell, so 64 covers several cells of headroom and still
#: keeps the attribution deterministic.
CAST_CREDIT_HISTORY_MAX: int = 64

SUPPORTED_WORKFLOWS: frozenset[str] = frozenset(
    {
        "route_a_a0",
        "casting_c1_fixed",
        "casting_c3_fixed",
        "casting_s_c3_fixed",
        "casting_s_c4_fixed",
        "casting_s_c5_fixed",
    }
)


def _default_env_factory(task: TaskInstance) -> Any:
    initial_inventory = tuple(
        {"type": item, "quantity": quantity}
        for item, quantity in task.initial_inventories["agent_1"].items()
        if quantity > 0
    )
    specification = PortalA0EnvSpec(
        max_episode_steps=task.limits["max_environment_steps"],
        max_game_time_seconds=task.limits["max_game_time_seconds"],
        initial_inventory=initial_inventory,
        initial_position=task.spawn_positions["agent_1"],
        # MineRL 1.0.2 ignores absolute AgentStart placement on this bridge.
        # Omitting it for casting makes both the player and atSpawn truth grid
        # use the real generated world spawn. Legacy Route A0 retains its
        # historical XML contract.
        include_agent_start_placement=task.workflow == "route_a_a0",
        # C1 does not move. A player-relative grid avoids Minecraft's
        # spawnRadius offset separating the player from the shared-spawn
        # truth anchor when absolute AgentStart placement is unavailable.
        grid_at_spawn=task.workflow != "casting_c1_fixed",
    )
    return specification.make()


class MineRLEnvironmentBackend:
    """Single-owner MineRL backend for the controlled Route A0 task.

    Phase 2 evaluator wiring:

    * The portal frame geometry is detected on every step from the
      evaluator-only grid. The first episode-built frame is *latched*
      together with its full geometry (``_latched["frame_identity"]``)
      so the verdict survives the Overworld grid being replaced by
      the Nether grid.
    * Activation is observed only when ``nether_portal`` appears in
      the *latched* episode-built frame's interior. A pre-existing
      portal with a nether_portal block does not activate the
      current episode.
    * Termination is taken from the MineRL ``done`` flag. Failure
      classification is only emitted after explicit termination.
    * All milestone events are emitted as ``StructuredEvent`` and
      each carries a latched timestamp recorded at first observation.
    """

    def __init__(
        self,
        env_factory: EnvFactory = _default_env_factory,
        *,
        reset_warmup_steps: int = 2,
        max_reset_attempts: int = 2,
    ) -> None:
        if type(reset_warmup_steps) is not int or reset_warmup_steps < 0:
            raise ValueError("reset_warmup_steps must be a non-negative integer")
        if type(max_reset_attempts) is not int or max_reset_attempts < 1:
            raise ValueError("max_reset_attempts must be a positive integer")
        self._env_factory = env_factory
        self._reset_warmup_steps = reset_warmup_steps
        self._max_reset_attempts = max_reset_attempts
        self._owner_thread: int | None = None
        self._opened = False
        self._env: Any | None = None
        self._task: TaskInstance | None = None
        self._step_id = 0
        self._latest_raw: dict[str, Any] | None = None
        self._baseline_grid: np.ndarray | None = None
        self._latched: dict[str, Any] = self._fresh_latched_state()
        # Used by tests / replay scripts to mark termination without
        # requiring a real MineRL ``done`` flag.
        self._forced_termination: tuple[int, str] | None = None
        self._hotbar_mapping: Mapping[str, str] = dict(PORTAL_A0_HOTBAR)

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return self._task.agent_ids if self._task is not None else ()

    @property
    def action_space(self) -> Any:
        self._require_env()
        return self._env.action_space

    @staticmethod
    def casting_c1_capabilities() -> BackendCapabilities:
        """Return the honest manifest for the current MineRL backend.

        The R6-C5-LIVE-MINERL-BACKEND-WIRING milestone wires the
        full MineRL action translator surface that the C3 / C4 / C5
        deterministic driver family needs:

        * ``equip_item(water_bucket)`` / ``equip_item(lava_bucket)`` /
          ``equip_item(cobblestone)`` / ``equip_item(flint_and_steel)``;
        * ``use_item(water_bucket)`` / ``use_item(lava_bucket)`` /
          ``use_item(flint_and_steel)``;
        * ``place_block(cobblestone)``;
        * bounded forward-only ``move`` and ``wait``;
        * the legacy A0 ``obsidian`` / ``flint_and_steel`` / ``dirt``
          actions used by the Phase 1 fixtures.

        The translator enforces the closed allowlist, strict type
        checks, and bounded numeric ranges; a regression test
        suite (``tests/test_r6_c5_live_minerl_backend_wiring.py``)
        exercises every supported action and every fail-closed
        negative path.

        The bridge now also exposes a typed
        :data:`~obsidianlink.env.portal_spec.PORTAL_SELECTED_ITEM_NAME`
        observation that the backend surfaces on
        :attr:`Observation.selected_item`. The value is read
        directly from the bridge; the backend never derives the
        selected item from the agent's request stream.

        The backend exposes a typed
        :class:`~obsidianlink.evaluation.casting.CastingEvaluationState`
        surface for the C1 single-cell task via
        :meth:`get_casting_evaluation_state`; the C2 three-cell
        :class:`~obsidianlink.evaluation.continuous_casting.ContinuousCastingEvaluationState`
        surface via
        :meth:`get_continuous_casting_evaluation_state`; the C3
        14-cell frozen-frame
        :class:`~obsidianlink.evaluation.casting_frame_evaluator.FrozenFrameEvaluationState`
        surface via :meth:`get_frame_evaluation_state`; the C4
        ignition
        :class:`~obsidianlink.evaluation.casting_ignition_evaluator.FrozenIgnitionEvaluationState`
        surface via :meth:`get_ignition_evaluation_state`; and
        the C5 Nether-entry
        :class:`~obsidianlink.evaluation.casting_nether_entry_evaluator.FrozenNetherEntryEvaluationState`
        surface via
        :meth:`get_nether_entry_evaluation_state`. All five
        surfaces derive target-block and fluid verdicts from the
        bridge's supported
        :class:`~obsidianlink.env.portal_spec.PortalGridObservation`
        raw observation and from per-cell, world-confirmed action credit
        history; the backend never fabricates world truth from
        driver intent.

        The manifest is a *static* declaration: it must never read
        MineRL state and must never depend on the current task.
        """
        return BackendCapabilities(
            can_select_water_bucket=True,
            can_select_lava_bucket=True,
            can_use_water_bucket=True,
            can_use_lava_bucket=True,
            exposes_public_inventory=True,
            exposes_selected_item=True,
            exposes_target_block_truth=True,
            exposes_fluid_truth=True,
        )

    def capabilities(self) -> BackendCapabilities:
        """Return the immutable manifest this backend declares.

        Currently identical to :meth:`casting_c1_capabilities`
        because the manifest is static. The instance method is kept
        so the project can later introduce per-backend configuration
        without changing call sites.
        """
        return self.casting_c1_capabilities()

    def open(self) -> None:
        if self._opened:
            raise RuntimeError("backend is already open")
        self._opened = True
        self._owner_thread = threading.get_ident()

    def reset(self, task: TaskInstance) -> Mapping[str, Observation]:
        self._assert_owner()
        # Pre-episode capability gate. Runs *before* ``self._task`` is
        # set, *before* the env factory is called, *before* any MineRL
        # reset / warmup, and *before* the baseline grid is captured.
        # An incomplete manifest must fail closed so the MineRL
        # runtime is never touched for a casting-c1 task it cannot
        # serve today. Non-casting workflows pass through untouched
        # so the legacy Route A0 baseline keeps working.
        assert_backend_can_start_task(self, task)
        if task.agent_ids != ("agent_1",):
            raise ValueError("PortalA0 currently supports exactly agent_1")
        if task.workflow not in SUPPORTED_WORKFLOWS:
            raise ValueError(f"unsupported MineRL workflow: {task.workflow!r}")
        # The R6-C5-LIVE-MINERL-BACKEND-WIRING milestone extends
        # the backend to serve the R6 casting task family. The
        # legacy ``route_a_a0`` shape still requires
        # ``Route A difficulty 1``; every other workflow (the R6
        # C3 / C4 / C5 frozen-frame / ignition / Nether-entry
        # contracts, plus any future casting workflow) must use
        # ``route == "lava_casting"`` and is allowed regardless of
        # difficulty.
        is_legacy_route_a0 = task.workflow == "route_a_a0"
        if is_legacy_route_a0:
            if task.route != "obsidian_mining" or task.difficulty != 1:
                raise ValueError(
                    "PortalA0 currently supports Route A difficulty 1"
                )
        elif task.route != "lava_casting":
            raise ValueError(
                "PortalA0 currently supports route_a_a0 (obsidian_mining) "
                "and the R6 casting task family (lava_casting)"
            )
        # ``route_a_a0`` predates task-derived inventory slot assignment and
        # its mission/test contract fixes obsidian, flint-and-steel, and dirt
        # to slots 1-3.  Preserve that compatibility contract while deriving
        # the casting workflows' slots from their frozen initial inventory.
        if is_legacy_route_a0:
            self._hotbar_mapping = dict(PORTAL_A0_HOTBAR)
        else:
            self._hotbar_mapping = build_hotbar_mapping(
                task.initial_inventories["agent_1"]
            )
        self._task = task
        raw: Mapping[str, Any] | None = None
        last_error: Exception | None = None
        for attempt in range(1, self._max_reset_attempts + 1):
            if self._env is not None:
                self._env.close()
                self._env = None
            try:
                self._env = self._env_factory(task)
                if hasattr(self._env, "seed"):
                    self._env.seed(task.world_seed)
                raw = self._env.reset()
                for _ in range(self._reset_warmup_steps):
                    raw, _, done, info = self._env.step(self.action_space.no_op())
                    if isinstance(info, Mapping) and "error" in info:
                        raise RuntimeError(
                            f"MineRL reset warm-up failed: {info['error']}"
                        )
                    if done:
                        raise RuntimeError("MineRL terminated during reset warm-up")
                break
            except (OSError, RuntimeError, TypeError) as error:
                last_error = error
                if attempt == self._max_reset_attempts:
                    raise RuntimeError(
                        "MineRL reset failed after "
                        f"{self._max_reset_attempts} attempts: {error}"
                    ) from error
        if raw is None:
            raise RuntimeError("MineRL reset produced no observation") from last_error
        self._step_id = 0
        self._latched = self._fresh_latched_state()
        self._latched["task_reset_step"] = 0
        self._latched["latched_timestamps"][
            "task_reset"
        ] = time.time()
        self._latest_raw = self._validate_raw_observation(raw)
        self._baseline_grid = self._grid_from_raw(self._latest_raw).copy()
        self._latched["grid_world_anchor"] = self._grid_world_anchor(
            self._latest_raw
        )
        # ObservationFromGrid reports ordinary blocks and fluids on one
        # server-side surface. Keep one previous snapshot so a bucket credit
        # is bound only to the exact cell whose world block changed after the
        # accepted action.
        self._latched["baseline_fluid_grid"] = self._baseline_grid.copy()
        self._latched["current_fluid_grid"] = self._baseline_grid.copy()
        self._latched["previous_truth_grid"] = self._baseline_grid.copy()
        return self._public_observations()

    def step(self, actions: Mapping[str, MacroAction]) -> BackendStep:
        task = self._require_task()
        self._assert_owner()
        if set(actions) != {"agent_1"}:
            raise ValueError("actions must contain exactly agent_1")
        action = actions["agent_1"]
        translation = translate_macro_action(
            action,
            self.action_space,
            hotbar_mapping=self._hotbar_mapping,
        )
        if not translation.accepted:
            raise RuntimeError(
                f"MineRL action translation rejected: {translation.error}"
            )
        # A credit is valid for this environment step only. It is added
        # only after strict macro translation succeeds, then consumed or
        # expired by ``_refresh_evaluation_milestones`` against cells
        # first observed in the returned post-action observation.
        accepted_obsidian_placement = (
            translation.accepted
            and isinstance(action, MacroAction)
            and action.action_type == "place_block"
            and action.target == "obsidian"
        )
        accepted_cast_kind = self._cast_credit_kind(action)
        raw: Any = None
        total_reward = 0.0
        done = False
        info: Mapping[str, Any] = {}
        for _ in range(action.duration_ticks):
            raw, reward, done, raw_info = self._env.step(translation.action)
            if isinstance(raw_info, Mapping) and "error" in raw_info:
                raise RuntimeError(f"MineRL step failed: {raw_info['error']}")
            if raw_info is not None and not isinstance(raw_info, Mapping):
                raise TypeError("MineRL step info must be a mapping")
            info = raw_info or {}
            total_reward += float(reward)
            if done:
                break
        if raw is None:
            raise RuntimeError("MineRL action produced no observation")
        self._step_id += 1
        self._latest_raw = self._validate_raw_observation(raw)
        if accepted_obsidian_placement:
            self._latched["pending_place_block_obsidian"] += 1
        if accepted_cast_kind is not None:
            self._record_cast_credit(accepted_cast_kind)
        self._refresh_evaluation_milestones()
        if done:
            self._mark_terminated(
                step_id=self._step_id,
                reason=str(info.get("termination_reason", "mine_done"))
                if isinstance(info, Mapping)
                else "mine_done",
            )
        public_info = {
            "translation_accepted": translation.accepted,
            "translation_error": translation.error,
        }
        if isinstance(info, Mapping):
            public_info["environment_info_keys"] = sorted(str(key) for key in info)
        return BackendStep(
            episode_id=task.task_id,
            step_id=self._step_id,
            observations=self._public_observations(),
            rewards={"agent_1": total_reward},
            terminated=bool(done) or self._forced_termination is not None,
            truncated=False,
            info=public_info,
        )

    def mark_terminated(
        self,
        *,
        step_id: int | None = None,
        reason: str = "driver_terminated",
    ) -> None:
        """Mark the episode as terminated from a non-MineRL driver.

        Phase 2: tests and deterministic drivers use this to signal
        end-of-episode without needing a real MineRL ``done`` flag.
        """
        self._assert_owner()
        self._mark_terminated(
            step_id=step_id if step_id is not None else self._step_id,
            reason=reason,
        )

    def _credit_pending_place_block_for_test(
        self,
        target: str,
        count: int = 1,
    ) -> None:
        """Inject test-only credits for a directly mutated fixture grid.

        Production drivers must submit ``MacroAction`` objects through
        ``step()``. This private hook exists only for geometry fixtures
        that cannot express individual block placement through MineRL.
        """
        self._assert_owner()
        if type(count) is not int or count < 0:
            raise ValueError("count must be a non-negative integer")
        if target == "obsidian":
            self._latched["pending_place_block_obsidian"] += count
        elif target in {"dirt", "flint_and_steel", "other"}:
            # Other items are tracked separately in the future;
            # for now they are no-ops because A0 only attributes
            # obsidian.
            return
        else:
            raise ValueError(f"unsupported place_block target: {target!r}")

    def get_evaluation_state(self) -> EvaluationState:
        task = self._require_task()
        raw = self._require_raw()
        grid = self._grid_from_raw(raw)
        baseline = self._baseline_grid
        if baseline is None:
            raise RuntimeError("evaluation baseline is unavailable")
        detection = detect_portal_frame_from_int_grid(
            grid, BLOCK_ID_TO_NAME, baseline_grid=baseline
        )
        self._refresh_evaluation_milestones(detection=detection, raw=raw)
        return self._build_evaluation_state(detection=detection, raw=raw)

    def get_casting_evaluation_state(
        self, target_cell: tuple[int, int, int]
    ) -> CastingEvaluationState:
        """Return the typed C1 single-cell casting truth for ``target_cell``.

        The cell must fall inside the portal grid. The state is
        built from the raw observation only; the driver or the
        orchestrator never supply it. ``current_block``,
        ``initial_target_block``, ``water_truth``, ``lava_truth``,
        ``target_update_evidence`` and ``relevant_action_steps``
        all derive from the world truth + the per-step cast
        credit history.
        """
        self._require_task()
        return self._typed_casting_truth_for(
            "casting_c1_fixed", (target_cell,)
        )

    def get_continuous_casting_evaluation_state(
        self,
        target_cells: tuple[tuple[int, int, int], ...] = CASTING_C3_TARGET_CELLS,
    ) -> ContinuousCastingEvaluationState:
        """Return the typed C2 three-cell continuous casting truth.

        ``target_cells`` defaults to
        :data:`~obsidianlink.evaluation.continuous_casting.CASTING_C3_TARGET_CELLS`
        which is the frozen C2 contract. Each cell's typed truth
        follows the same per-cell rules as the C1 surface; the
        multi-cell evaluator then reuses the C2 outcome priority.
        """
        task = self._require_task()
        raw = self._require_raw()
        if not target_cells:
            raise ValueError("target_cells must be a non-empty tuple")
        block_grid = self._grid_from_raw(raw)
        baseline_block = self._baseline_grid
        cells: list[ContinuousCastingCellTruth] = []
        for cell in target_cells:
            (
                initial_block,
                current_block,
                water_truth,
                lava_truth,
                transition_evidence,
                relevant,
            ) = self._typed_cell_evidence(
                cell,
                current_grid=block_grid,
                baseline_grid=baseline_block,
            )
            cells.append(
                ContinuousCastingCellTruth(
                    target_cell=cell,
                    initial_block=initial_block,
                    current_block=current_block,
                    water_truth=water_truth,
                    lava_truth=lava_truth,
                    transition_evidence=transition_evidence,
                    relevant_action_steps=relevant,
                )
            )
        max_env_steps = int(task.limits["max_environment_steps"])
        max_game_time = int(task.limits["max_game_time_seconds"])
        terminated = bool(self._latched["episode_terminated"])
        return ContinuousCastingEvaluationState(
            episode_id=task.task_id,
            step_id=self._step_id,
            cells=tuple(cells),
            agent_id="agent_1",
            causality_window_steps=4,
            episode_terminated=terminated,
            terminated_step=self._latched["terminated_step"],
            terminated_reason=self._latched["terminated_reason"],
            current_time_seconds=float(self._step_id),
            max_environment_steps=max_env_steps,
            max_game_time_seconds=max_game_time,
        )

    # ------------------------------------------------------------------
    # R6-C5-LIVE-MINERL-BACKEND-WIRING typed truth surface
    # ------------------------------------------------------------------
    #
    # The five ``get_*_evaluation_state`` methods below build typed
    # evaluator-only state from the raw MineRL observation and the
    # backend's per-step action credit history. They are the single
    # public surface the casting evaluators consume in production;
    # the FakeBackend siblings live on a separate slot but use the
    # same dataclass types. The surface is closed: each method
    # returns the typed state for one casting level, never a
    # dictionary the caller can mutate. Truth is sourced from
    # server-side raw observations only; the backend never
    # derives target-block / fluid verdicts from action intent.
    # ------------------------------------------------------------------

    def _typed_casting_truth_for(
        self,
        workflow: str,
        target_cells: tuple[tuple[int, int, int], ...],
    ) -> CastingEvaluationState:
        """Build a typed :class:`CastingEvaluationState` for one
        single-cell C1-style target.

        The state is composed only from the raw observation,
        the per-step cast credit history, and the typed fluid
        grid. The Agent's intent (which cell the driver is
        *trying* to cast) is irrelevant; the truth comes from
        what the world actually contains.
        """
        task = self._require_task()
        if not target_cells:
            raise ValueError("target_cells must be a non-empty tuple")
        if len(target_cells) != 1:
            raise ValueError(
                "C1 typed truth accepts exactly one target cell, got "
                f"{len(target_cells)}"
            )
        if workflow not in {"casting_c1_fixed"}:
            raise ValueError(
                "C1 typed truth requires workflow=='casting_c1_fixed', "
                f"got {workflow!r}"
            )
        target_cell = target_cells[0]
        raw = self._require_raw()
        block_grid = self._grid_from_raw(raw)
        baseline_block = self._baseline_grid
        cell_index = self._workflow_cell_index(target_cell)
        if cell_index is None:
            raise ValueError(
                f"target cell {target_cell!r} is outside the portal grid"
            )
        (
            initial_block,
            current_block,
            water_truth,
            lava_truth,
            transition_evidence,
            relevant_action_steps,
        ) = self._typed_cell_evidence(
            target_cell,
            current_grid=block_grid,
            baseline_grid=baseline_block,
        )
        terminated = bool(self._latched["episode_terminated"])
        terminated_step = self._latched["terminated_step"]
        terminated_reason = self._latched["terminated_reason"]
        max_env_steps = int(task.limits["max_environment_steps"])
        max_game_time = int(task.limits["max_game_time_seconds"])
        return CastingEvaluationState(
            episode_id=task.task_id,
            step_id=self._step_id,
            agent_id="agent_1",
            target_cell=target_cell,
            initial_target_block=initial_block,
            current_target_block=current_block,
            target_update_evidence=transition_evidence,
            water_truth=water_truth,
            lava_truth=lava_truth,
            relevant_action_steps=relevant_action_steps,
            causality_window_steps=4,
            episode_terminated=terminated,
            terminated_step=terminated_step,
            terminated_reason=terminated_reason,
            current_time_seconds=float(self._step_id),
            max_environment_steps=max_env_steps,
            max_game_time_seconds=max_game_time,
        )

    def _typed_frame_state(
        self,
        target_offsets: tuple[tuple[int, int, int], ...],
    ) -> FrozenFrameEvaluationState:
        """Build the typed C3 frozen-frame truth for ``target_offsets``.

        The C3 frame evaluator expects exactly 14 target cells in
        the public :data:`CASTING_S_C3_FRAME_CELLS` order, plus 6
        interior cells in the matching
        :data:`CASTING_S_C3_INTERIOR_CELLS` order. The state is
        composed from the supported raw block-grid observation and
        world-confirmed, per-cell action latches. The
        ``relevant_action_steps`` and ``action_evidence`` records
        follow the same C1 / C2 attribution contract.
        """
        task = self._require_task()
        if target_offsets != CASTING_S_C3_FRAME_CELLS:
            raise ValueError(
                "casting_s_c3_fixed requires target_offsets to match the "
                "frozen CASTING_S_C3_FRAME_CELLS order"
            )
        raw = self._require_raw()
        block_grid = self._grid_from_raw(raw)
        baseline_block = self._baseline_grid
        cells: list[FrozenFrameCellTruth] = []
        for cell in target_offsets:
            (
                initial_block,
                current_block,
                water_truth,
                lava_truth,
                transition_evidence,
                relevant,
            ) = self._typed_cell_evidence(
                cell,
                current_grid=block_grid,
                baseline_grid=baseline_block,
            )
            water_credit = self._latched[
                "first_water_step_by_offset"
            ].get(cell)
            lava_credit = self._latched[
                "first_lava_step_by_offset"
            ].get(cell)
            # ``action_evidence`` mirrors ``relevant_action_steps``
            # so the C3 frame evaluator can read either surface.
            records: list[FrozenFrameActionEvidence] = []
            for step in relevant:
                kind = (
                    "water_bucket"
                    if (
                        water_credit is not None
                        and step == water_credit
                    )
                    else "lava_bucket"
                )
                records.append(
                    FrozenFrameActionEvidence(
                        episode_id=task.task_id,
                        step_id=step,
                        agent_id=CASTING_S_C3_AGENT_ID,
                        action_type="use_item",
                        item=kind,
                        target_cell=cell,
                    )
                )
            transition_action_step = (
                water_credit if water_credit is not None else (
                    lava_credit if lava_credit is not None else None
                )
            )
            cells.append(
                FrozenFrameCellTruth(
                    target_cell=cell,
                    initial_block=initial_block,
                    current_block=current_block,
                    water_truth=water_truth,
                    lava_truth=lava_truth,
                    transition_evidence=transition_evidence,
                    relevant_action_steps=relevant,
                    action_evidence=tuple(records),
                    transition_action_step=transition_action_step,
                )
            )
        # Interior cells: read the block grid and report air /
        # nether_portal / fire (the closed interior set) or fail
        # closed with ``None`` so the evaluator surfaces a typed
        # truth-missing verdict rather than fabricating a cell.
        interior_cells: list[FrozenFrameInteriorCellTruth] = []
        for cell in CASTING_S_C3_INTERIOR_CELLS:
            current = self._block_name_at(
                block_grid, cell, default="missing"
            )
            if current not in {"air", "nether_portal", "fire"}:
                current = None
            interior_cells.append(
                FrozenFrameInteriorCellTruth(
                    target_cell=cell,
                    current_block=current,
                )
            )
        max_env_steps = int(task.limits["max_environment_steps"])
        max_game_time = int(task.limits["max_game_time_seconds"])
        terminated = bool(self._latched["episode_terminated"])
        return FrozenFrameEvaluationState(
            episode_id=task.task_id,
            step_id=self._step_id,
            cells=tuple(cells),
            interior_cells=tuple(interior_cells),
            agent_id=CASTING_S_C3_AGENT_ID,
            causality_window_steps=4,
            episode_terminated=terminated,
            terminated_step=self._latched["terminated_step"],
            terminated_reason=self._latched["terminated_reason"],
            current_time_seconds=float(self._step_id),
            max_environment_steps=max_env_steps,
            max_game_time_seconds=max_game_time,
        )

    def get_frame_evaluation_state(
        self,
    ) -> FrozenFrameEvaluationState:
        """Return the typed C3 frozen-frame truth surface.

        The frame plan is fixed at :data:`CASTING_S_C3_FRAME_CELLS`
        and :data:`CASTING_S_C3_INTERIOR_CELLS`; the backend
        builds the typed state directly from the raw observation
        + per-step cast credit history. No driver or orchestrator
        input is required.
        """
        self._require_task()
        return self._typed_frame_state(CASTING_S_C3_FRAME_CELLS)

    def get_ignition_evaluation_state(
        self,
    ) -> FrozenIgnitionEvaluationState:
        """Return the typed C4 ignition truth.

        The C4 ignition evaluator reuses the C3 frame truth; the
        backend's latched ``first_ignition_step`` and
        ``first_nether_portal_step`` supply the typed ignition
        action and portal activation evidence. The frame identity
        is built from the public C3 geometry constants; the
        activation's ``latched_frame_identity`` is the same
        instance so the C4 evaluator accepts it.
        """
        task = self._require_task()
        frame_state = self._typed_frame_state(CASTING_S_C3_FRAME_CELLS)
        ignition_step = self._latched["first_ignition_step"]
        activation_step = self._latched["first_nether_portal_step"]
        frame_step = self._latched["typed_frame_complete_step"]
        if ignition_step is None:
            ignition_evidence: IgnitionActionEvidence | None = None
        else:
            ignition_evidence = IgnitionActionEvidence(
                episode_id=task.task_id,
                step_id=ignition_step,
                agent_id=CASTING_S_C4_AGENT_ID,
                action_type="use_item",
                item="flint_and_steel",
                target_cell=(1, 1, 1),
            )
        if activation_step is None:
            activation_evidence: PortalActivationEvidence | None = None
        else:
            identity = build_c4_c3_frame_identity(
                episode_id=task.task_id,
                step_id=frame_step if frame_step is not None else self._step_id,
                agent_id=CASTING_S_C4_AGENT_ID,
                activation_offsets=((1, 1, 1),),
            )
            activation_evidence = PortalActivationEvidence(
                episode_id=task.task_id,
                update_step=activation_step,
                agent_id=CASTING_S_C4_AGENT_ID,
                nether_portal_offset=(1, 1, 1),
                latched_frame_identity=identity,
            )
        if activation_evidence is not None:
            identity = activation_evidence.latched_frame_identity
        else:
            identity = build_c4_c3_frame_identity(
                episode_id=task.task_id,
                step_id=frame_step if frame_step is not None else self._step_id,
                agent_id=CASTING_S_C4_AGENT_ID,
                activation_offsets=(
                    (1, 1, 1) if activation_step is not None else ()
                ),
            )
        max_env_steps = int(task.limits["max_environment_steps"])
        max_game_time = int(task.limits["max_game_time_seconds"])
        terminated = bool(self._latched["episode_terminated"])
        return FrozenIgnitionEvaluationState(
            episode_id=task.task_id,
            step_id=self._step_id,
            frame_state=frame_state,
            latched_frame_identity=identity,
            ignition_action=ignition_evidence,
            activation_evidence=activation_evidence,
            agent_id=CASTING_S_C4_AGENT_ID,
            causality_window_steps=4,
            episode_terminated=terminated,
            terminated_step=self._latched["terminated_step"],
            terminated_reason=self._latched["terminated_reason"],
            current_time_seconds=float(self._step_id),
            max_environment_steps=max_env_steps,
            max_game_time_seconds=max_game_time,
        )

    def get_nether_entry_evaluation_state(
        self,
    ) -> FrozenNetherEntryEvaluationState:
        """Return the typed C5 Nether-entry truth.

        The C5 evaluator reuses the C4 ignition evaluator, so the
        backend returns a frozen state that embeds the C4
        ignition state built by
        :meth:`get_ignition_evaluation_state`. The Nether-entry
        evidence is sourced from the bridge's
        ``portal_transition`` payload; the per-agent
        ``entered_via_episode_portal`` and ``pre_transition_position``
        come from the backend's latched attribution surface.
        """
        task = self._require_task()
        ignition_state = self.get_ignition_evaluation_state()
        agent_id = CASTING_S_C5_AGENT_ID
        transition_step = self._latched["transition_step_by_agent"].get(
            agent_id
        )
        pre_position = self._latched["pre_transition_position_by_agent"].get(
            agent_id
        )
        entered_via_portal = self._latched[
            "entered_via_episode_portal_by_agent"
        ].get(agent_id)
        matched_legacy_identity = self._latched[
            "matched_frame_identity_by_agent"
        ].get(agent_id)
        if (
            transition_step is not None
            and pre_position is not None
            and entered_via_portal is True
            and matched_legacy_identity is not None
            and ignition_state.activation_evidence is not None
        ):
            entry_evidence = NetherEntryEvidence(
                episode_id=task.task_id,
                agent_id=agent_id,
                source_dimension=CASTING_S_C5_SOURCE_DIMENSION,
                target_dimension=CASTING_S_C5_TARGET_DIMENSION,
                transition_step=transition_step,
                pre_transition_position=tuple(float(v) for v in pre_position),
                entered_via_episode_portal=True,
                matched_frame_identity=ignition_state.latched_frame_identity,
            )
        else:
            entry_evidence = None
        agents_in_nether: frozenset[str] = frozenset(
            self._latched["first_nether_step_by_agent"].keys()
        )
        max_env_steps = int(task.limits["max_environment_steps"])
        max_game_time = int(task.limits["max_game_time_seconds"])
        terminated = bool(self._latched["episode_terminated"])
        return FrozenNetherEntryEvaluationState(
            episode_id=task.task_id,
            step_id=self._step_id,
            ignition_state=ignition_state,
            agents_in_nether=agents_in_nether,
            entry_evidence=entry_evidence,
            agent_id=CASTING_S_C5_AGENT_ID,
            episode_terminated=terminated,
            terminated_step=self._latched["terminated_step"],
            terminated_reason=self._latched["terminated_reason"],
            current_time_seconds=float(self._step_id),
            max_environment_steps=max_env_steps,
            max_game_time_seconds=max_game_time,
        )

    def close(self) -> None:
        if not self._opened:
            return
        self._assert_owner()
        try:
            if self._env is not None:
                self._env.close()
        finally:
            self._env = None
            self._task = None
            self._latest_raw = None
            self._baseline_grid = None
            self._latched = self._fresh_latched_state()
            self._step_id = 0
            self._owner_thread = None
            self._forced_termination = None
            self._hotbar_mapping = dict(PORTAL_A0_HOTBAR)
            self._opened = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fresh_latched_state(self) -> dict[str, Any]:
        return {
            "task_reset_step": None,
            "first_obsidian_placed_step": None,
            "build_site_selected_step": None,
            "first_valid_frame_step": None,
            "first_activation_step": None,
            "first_nether_step_by_agent": {},
            "episode_obsidian_offsets": set(),
            "attributed_obsidian_offsets": set(),
            "external_obsidian_offsets": set(),
            "pending_place_block_obsidian": 0,
            "grid_world_anchor": None,
            "max_obsidian_added": 0,
            "portal_activated": False,
            "frame_candidate_observed": False,
            "attribution_failed_observed": False,
            "external_structure_candidate_count": 0,
            "frame_identity": None,
            "latched_activation_offsets": [],
            "latched_timestamps": {},
            "episode_terminated": False,
            "terminated_step": None,
            "terminated_reason": None,
            "pre_transition_position_by_agent": {},
            "last_overworld_position": None,
            "transition_step_by_agent": {},
            "entered_via_episode_portal_by_agent": {},
            "matched_frame_identity_by_agent": {},
            # Evaluator-only typed truth surface (R6-C5-LIVE-MINERL-BACKEND-WIRING).
            # Every entry in ``cast_credit_history`` is recorded only
            # after the macro translator has accepted the action and
            # the low-level MineRL step has produced an observation.
            # The history is bounded to :data:`CAST_CREDIT_HISTORY_MAX`
            # so the list cannot grow without bound across long
            # episodes; once a per-cell attribution commits, the
            # relevant actions are folded into the latched cell
            # state and the entry can be retired.
            "cast_credit_history": [],
            "baseline_fluid_grid": None,
            "current_fluid_grid": None,
            "previous_truth_grid": None,
            # Per-cell first-observed obsidian step. Set when the
            # block grid shows ``obsidian`` at a cell that was not
            # obsidian at baseline; the step is then bound to the
            # most-recent water / lava cast credits to build the
            # per-cell transition evidence and ``relevant_action_steps``.
            "first_obsidian_step_by_offset": {},
            # First-observed ``flint_and_steel`` action step. The
            # backend latches exactly one ignition step per episode
            # because the public C4 / C5 contract binds the ignition
            # to a single ``use_item(flint_and_steel)`` action.
            "first_ignition_step": None,
            # First-observed ``nether_portal`` block on a frame
            # interior cell. Latched once per episode; subsequent
            # appearances are ignored so a stale activation cannot
            # retroactively replace a missing one.
            "first_nether_portal_step": None,
            "typed_frame_complete_step": None,
            # Per-cell first-observed water / lava fluid. The
            # C1 / C2 / C3 evaluators need both water_truth and
            # lava_truth to be present (True) for the same cell
            # in the same typed state. The A0 MineRL bridge
            # cannot supply both fluids at the same time (water
            # + lava react into obsidian on the next tick), so
            # the backend latches the first water and first
            # lava observations independently. The latched
            # vericts expire only at reset / close, never at
            # step boundary, so the C1 evaluator always sees
            # the full causal chain.
            "first_water_step_by_offset": {},
            "first_lava_step_by_offset": {},
        }

    def _mark_terminated(self, *, step_id: int, reason: str) -> None:
        if self._latched["episode_terminated"]:
            return
        self._latched["episode_terminated"] = True
        self._latched["terminated_step"] = step_id
        self._latched["terminated_reason"] = reason
        self._latched["latched_timestamps"]["_terminated"] = time.time()

    @staticmethod
    def _cast_credit_kind(action: MacroAction) -> str | None:
        """Return the typed credit kind for an accepted cast action.

        Returns ``"water"`` for ``use_item(water_bucket)``,
        ``"lava"`` for ``use_item(lava_bucket)``, and
        ``"flint_and_steel"`` for ``use_item(flint_and_steel)``.
        Other action types return ``None`` so the credit history
        only carries evaluator-relevant credits.
        """
        if not isinstance(action, MacroAction):
            return None
        if action.action_type == "use_item":
            if action.target == "water_bucket":
                return "water"
            if action.target == "lava_bucket":
                return "lava"
            if action.target == "flint_and_steel":
                return "flint_and_steel"
        return None

    @staticmethod
    def _cell_index_in_grid(
        cell: tuple[int, int, int],
    ) -> tuple[int, int, int] | None:
        """Return the (x, y, z) index of ``cell`` in the portal grid.

        The portal grid is anchored at :data:`PORTAL_GRID_MIN`; a
        cell that falls outside the closed extent is reported as
        ``None`` so callers can fail closed rather than wrap.
        """
        if (
            not isinstance(cell, tuple)
            or len(cell) != 3
            or any(not isinstance(value, int) for value in cell)
        ):
            return None
        x = cell[0] - PORTAL_GRID_MIN[0]
        y = cell[1] - PORTAL_GRID_MIN[1]
        z = cell[2] - PORTAL_GRID_MIN[2]
        x_size = PORTAL_GRID_MAX[0] - PORTAL_GRID_MIN[0] + 1
        y_size = PORTAL_GRID_MAX[1] - PORTAL_GRID_MIN[1] + 1
        z_size = PORTAL_GRID_MAX[2] - PORTAL_GRID_MIN[2] + 1
        if (
            x < 0
            or x >= x_size
            or y < 0
            or y >= y_size
            or z < 0
            or z >= z_size
        ):
            return None
        return (x, y, z)

    def _workflow_cell_index(
        self,
        cell: tuple[int, int, int],
    ) -> tuple[int, int, int] | None:
        """Map a public task cell to the atSpawn grid coordinate system."""
        task = self._task
        if task is not None and task.workflow == "casting_c1_fixed":
            spawn = task.spawn_positions.get("agent_1")
            if (
                not isinstance(spawn, tuple)
                or len(spawn) != 3
                or any(type(value) is not int for value in spawn)
            ):
                return None
            cell = tuple(cell[index] - spawn[index] for index in range(3))
        return self._cell_index_in_grid(cell)

    def _block_name_at(
        self,
        grid: np.ndarray | None,
        cell: tuple[int, int, int],
        *,
        default: str,
    ) -> str | None:
        """Return the typed block name at ``cell`` in ``grid``."""
        if grid is None:
            return default
        index = self._workflow_cell_index(cell)
        if index is None:
            return default
        block_id = int(grid[index])
        if 0 <= block_id < len(PORTAL_GRID_BLOCKS):
            block = PORTAL_GRID_BLOCKS[block_id]
            if block == "flowing_water":
                return "water"
            if block == "flowing_lava":
                return "lava"
            if block in {"other", "missing"}:
                return default
            return block
        return default

    def _all_workflow_offsets(self) -> tuple[tuple[int, int, int], ...]:
        """Return the union of the workflow's tracked offsets.

        The C1 task tracks its single target cell. The C2
        task tracks the three R5 cells. The C3 / C4 / C5 tasks
        track the public C3 frame plan (14 target + 6
        interior cells). The union is what the per-cell
        fluid latcher iterates over.
        """
        task = self._task
        if task is None:
            return ()
        workflow = task.workflow
        if workflow == "casting_c1_fixed":
            params = task.scenario_parameters
            cell_value = (
                params.get("target_cell")
                if isinstance(params, Mapping)
                else None
            )
            if not isinstance(cell_value, (list, tuple)) or len(cell_value) != 3:
                return ()
            try:
                return (
                    (
                        int(cell_value[0]),
                        int(cell_value[1]),
                        int(cell_value[2]),
                    ),
                )
            except (TypeError, ValueError):
                return ()
        if workflow == "casting_c3_fixed":
            return CASTING_C3_TARGET_CELLS
        if workflow in {"casting_s_c3_fixed", "casting_s_c4_fixed", "casting_s_c5_fixed"}:
            return CASTING_S_C3_FRAME_CELLS + CASTING_S_C3_INTERIOR_CELLS
        return ()

    def _cast_relevant_steps_for_cell(
        self,
        target_cell: tuple[int, int, int],
        *,
        water_credit: int | None,
        lava_credit: int | None,
    ) -> tuple[int, ...]:
        """Return the per-cell relevant action step tuple.

        The C1 / C2 / C3 casting evaluators expect a non-empty
        ``relevant_action_steps`` only when the cell actually
        received a credited cast during the episode. The
        attribution is strict: pre-existing fluids and
        pre-existing obsidian do not earn causal credit. The
        helper honours the casting evaluator's requirement that
        steps be unique and ordered.
        """
        if water_credit is None and lava_credit is None:
            return ()
        # Preserve the driver-relative order ``(lava, water)``
        # so the per-cell record matches the test orchestrator's
        # shape. The casting evaluator only checks disjointness
        # and the causality window; the relative order is part
        # of the public contract.
        ordered: list[int] = []
        if lava_credit is not None:
            ordered.append(lava_credit)
        if water_credit is not None:
            ordered.append(water_credit)
        ordered.sort()
        return tuple(ordered)

    def _typed_cell_evidence(
        self,
        cell: tuple[int, int, int],
        *,
        current_grid: np.ndarray,
        baseline_grid: np.ndarray | None,
    ) -> tuple[
        str | None,
        str | None,
        CastingFluidTruth,
        CastingFluidTruth,
        CastingTransitionEvidence | None,
        tuple[int, ...],
    ]:
        """Build one cell's truth from observed, cell-bound evidence."""
        idx = self._workflow_cell_index(cell)
        if idx is None:
            raise ValueError(f"target cell {cell!r} is outside the portal grid")
        current_block = self._block_name_at(
            current_grid, cell, default="missing"
        )
        initial_block = self._block_name_at(
            baseline_grid, cell, default="missing"
        )
        water_step = self._latched["first_water_step_by_offset"].get(cell)
        lava_step = self._latched["first_lava_step_by_offset"].get(cell)
        update_step = self._latched["first_obsidian_step_by_offset"].get(cell)
        current_id = int(current_grid[idx])
        water_truth = self._fluid_truth_from_id(
            current_id, expected="water", step_id=self._step_id
        )
        lava_truth = self._fluid_truth_from_id(
            current_id, expected="lava", step_id=self._step_id
        )
        if water_step is not None:
            water_truth = CastingFluidTruth(True, water_step)
        if lava_step is not None:
            lava_truth = CastingFluidTruth(True, lava_step)
        relevant = self._cast_relevant_steps_for_cell(
            cell, water_credit=water_step, lava_credit=lava_step
        )
        transition: CastingTransitionEvidence | None = None
        if (
            initial_block not in {None, "missing", "obsidian"}
            and current_block == "obsidian"
            and update_step is not None
        ):
            transition = CastingTransitionEvidence(
                before_block=initial_block,
                after_block="obsidian",
                update_step=update_step,
            )
        return (
            initial_block,
            current_block,
            water_truth,
            lava_truth,
            transition,
            relevant,
        )

    def _record_cast_credit(self, kind: str) -> None:
        """Record an accepted cast credit for the current step.

        The credit is latched after the low-level MineRL step has
        returned, so a translation that fails closed does not earn
        a credit. The history is bounded to
        :data:`CAST_CREDIT_HISTORY_MAX`; once the cap is reached
        the oldest entry is dropped so the attribution remains
        bounded and deterministic.
        """
        if kind not in {"water", "lava", "flint_and_steel"}:
            raise ValueError(f"unknown cast credit kind: {kind!r}")
        history: list[tuple[int, str]] = self._latched["cast_credit_history"]
        history.append((self._step_id, kind))
        if len(history) > CAST_CREDIT_HISTORY_MAX:
            del history[: len(history) - CAST_CREDIT_HISTORY_MAX]
        if kind == "flint_and_steel":
            if self._latched["first_ignition_step"] is None:
                self._latched["first_ignition_step"] = self._step_id

    def _assert_owner(self) -> None:
        if not self._opened:
            raise RuntimeError("backend is not open")
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("MineRL lifecycle must stay on the owner thread")

    def _require_env(self) -> None:
        self._assert_owner()
        if self._env is None:
            raise RuntimeError("backend has not been reset")

    def _require_task(self) -> TaskInstance:
        self._require_env()
        if self._task is None:
            raise RuntimeError("task is unavailable")
        return self._task

    def _require_raw(self) -> dict[str, Any]:
        if self._latest_raw is None:
            raise RuntimeError("raw observation is unavailable")
        return self._latest_raw

    @staticmethod
    def _validate_raw_observation(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise ValueError("MineRL observation must be a dictionary")
        if "pov" not in raw or "inventory" not in raw:
            raise ValueError("MineRL observation must include pov and inventory")
        pov = raw["pov"]
        if not isinstance(pov, np.ndarray) or pov.dtype != np.uint8:
            raise ValueError("pov must be a uint8 numpy array")
        if pov.shape != (360, 640, 3):
            raise ValueError(f"unexpected pov shape: {pov.shape}")
        return raw

    def _public_observations(self) -> Mapping[str, Observation]:
        task = self._require_task()
        raw = self._require_raw()
        inventory = {
            str(item): int(np.asarray(quantity).item())
            for item, quantity in dict(raw["inventory"]).items()
            if int(np.asarray(quantity).item()) > 0
        }
        return {
            "agent_1": Observation(
                episode_id=task.task_id,
                agent_id="agent_1",
                step_id=self._step_id,
                timestamp=time.time(),
                frame=raw["pov"],
                visible_inventory=inventory,
                selected_item=self._selected_item_from_raw(
                    raw,
                    required=task.workflow != "route_a_a0",
                ),
                workflow_stage=task.workflow,
            )
        }

    @staticmethod
    def _grid_from_raw(raw: Mapping[str, Any]) -> np.ndarray:
        """Return the portal grid as a 3D (x, y, z) int32 array."""
        flat = np.asarray(
            raw.get(
                "portal_grid",
                np.full(PORTAL_GRID_SIZE, PORTAL_GRID_MISSING_ID),
            ),
            dtype=np.int32,
        ).reshape(-1)
        if flat.size != PORTAL_GRID_SIZE:
            raise ValueError(f"unexpected portal grid size: {flat.size}")
        x_size = PORTAL_GRID_MAX[0] - PORTAL_GRID_MIN[0] + 1
        y_size = PORTAL_GRID_MAX[1] - PORTAL_GRID_MIN[1] + 1
        z_size = PORTAL_GRID_MAX[2] - PORTAL_GRID_MIN[2] + 1
        grid_3d = flat.reshape((x_size, z_size, y_size), order="F")
        return np.ascontiguousarray(grid_3d.transpose(0, 2, 1))

    @staticmethod
    def _selected_item_from_raw(
        raw: Mapping[str, Any],
        *,
        required: bool = True,
    ) -> str | None:
        """Return the agent's currently selected hotbar item, or ``None``.

        The value comes from MineRL's built-in
        ``EquippedItemObservation`` at ``equipped_items.mainhand.type``.
        Missing or malformed payloads fail closed because this backend claims
        the selected-item capability.
        """
        if not isinstance(raw, Mapping):
            raise ValueError("raw observation must be a mapping")
        equipped = raw.get(PORTAL_SELECTED_ITEM_NAME)
        if not isinstance(equipped, Mapping):
            if not required and equipped is None:
                return None
            raise ValueError("equipped_items observation must be a mapping")
        mainhand = equipped.get("mainhand")
        if not isinstance(mainhand, Mapping):
            raise ValueError("equipped_items.mainhand must be a mapping")
        value = mainhand.get("type")
        if isinstance(value, str) and value in ("none", "empty", "air"):
            return None
        if not isinstance(value, str):
            raise ValueError(
                "equipped_items.mainhand.type must be a string from "
                f"{PORTAL_SELECTABLE_ITEMS!r} or 'empty'/'air', "
                f"got {type(value).__name__}"
            )
        if value not in PORTAL_SELECTABLE_ITEMS:
            raise ValueError(
                f"equipped_items.mainhand.type {value!r} is outside the "
                f"closed Portal selectable items set "
                f"{PORTAL_SELECTABLE_ITEMS!r}"
            )
        return value

    @staticmethod
    def _decode_grid(grid: np.ndarray) -> list[str]:
        flat = np.asarray(grid).reshape(-1)
        result: list[str] = []
        for value in flat:
            index = int(value)
            if 0 <= index < len(PORTAL_GRID_BLOCKS):
                result.append(PORTAL_GRID_BLOCKS[index])
            else:
                result.append("other")
        return result

    @staticmethod
    def _fluid_grid_from_raw(
        raw: Mapping[str, Any],
    ) -> np.ndarray:
        """Return the server block grid used for typed fluid evidence.

        Malmo's supported ``ObservationFromGrid`` already reports water and
        lava block names. A second imaginary observation handler would make
        the mission XML invalid, so fluid truth is a typed projection of the
        same evaluator-only block grid.
        """
        return MineRLEnvironmentBackend._grid_from_raw(raw)

    @staticmethod
    def _fluid_truth_from_id(
        fluid_id: int, *, expected: str, step_id: int
    ) -> CastingFluidTruth:
        """Return a water- or lava-specific verdict for one block id."""
        if expected == "water":
            positive_ids = FLUID_WATER_IDS
        elif expected == "lava":
            positive_ids = FLUID_LAVA_IDS
        else:
            raise ValueError("expected fluid must be 'water' or 'lava'")
        if fluid_id == FLUID_MISSING_ID or fluid_id == FLUID_OTHER_ID:
            return CastingFluidTruth(present=None, evidence_step=None)
        return CastingFluidTruth(
            present=fluid_id in positive_ids,
            evidence_step=step_id,
        )

    @staticmethod
    def _dimension(raw: Mapping[str, Any]) -> str:
        return str(np.asarray(raw.get("portal_dimension", "unknown")).item())

    @staticmethod
    def _position(
        raw: Mapping[str, Any],
    ) -> tuple[float, float, float] | None:
        """Return the agent's world (x, y, z) position from a MineRL
        observation, or ``None`` if the bridge did not supply it.
        """
        location = raw.get("location_stats")
        if not isinstance(location, Mapping):
            return None
        try:
            x = float(np.asarray(location["xpos"]).item())
            y = float(np.asarray(location["ypos"]).item())
            z = float(np.asarray(location["zpos"]).item())
        except (KeyError, TypeError, ValueError):
            return None
        return (x, y, z)

    @staticmethod
    def _grid_world_anchor(
        raw: Mapping[str, Any],
    ) -> tuple[int, int, int] | None:
        value = np.asarray(raw.get("portal_grid_origin", ()))
        if (
            value.shape != (3,)
            or value.dtype.kind not in {"i", "u"}
            or any(
                coordinate < -30_000_000 or coordinate > 30_000_000
                for coordinate in value.tolist()
            )
        ):
            return None
        return tuple(int(coordinate) for coordinate in value.tolist())

    def _world_bounds_from_identity(
        self,
        frame_identity: Mapping[str, Any],
    ) -> tuple[float, float, float, float, float, float] | None:
        interior = frame_identity.get("interior_block_offsets", [])
        if not interior:
            return None
        anchor = self._latched.get("grid_world_anchor")
        if (
            not isinstance(anchor, tuple)
            or len(anchor) != 3
            or any(type(value) is not int for value in anchor)
        ):
            return None
        xs = [anchor[0] + c[0] + PORTAL_GRID_MIN[0] for c in interior]
        ys = [anchor[1] + c[1] + PORTAL_GRID_MIN[1] for c in interior]
        zs = [anchor[2] + c[2] + PORTAL_GRID_MIN[2] for c in interior]
        return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))

    def _position_matches_latched_frame(
        self,
        position: tuple[float, float, float] | None,
        tolerance: float = 1.5,
    ) -> bool:
        """True if ``position`` is within ``tolerance`` blocks of the
        latched episode-built frame's interior bounding box.
        """
        if position is None:
            return False
        identity = self._latched["frame_identity"]
        if identity is None:
            return False
        bounds = self._world_bounds_from_identity(identity)
        if bounds is None:
            return False
        min_x, min_y, min_z, max_x, max_y, max_z = bounds
        px, py, pz = position
        return (
            min_x - tolerance <= px <= max_x + tolerance
            and min_y - tolerance <= py <= max_y + tolerance
            and min_z - tolerance <= pz <= max_z + tolerance
        )

    def _transition_matches_latched_frame(
        self,
        raw: Mapping[str, Any],
        position: tuple[float, float, float] | None,
    ) -> bool | None:
        """Validate explicit bridge transition evidence.

        Position is a necessary sanity check, never sufficient causal
        evidence. Without a typed bridge verdict and the exact latched
        frame interior identity this method returns ``None`` (unknown).
        """
        evidence = raw.get("portal_transition")
        if not isinstance(evidence, Mapping):
            return None
        try:
            present_array = np.asarray(evidence["present"])
            entered_array = np.asarray(evidence["entered_via_portal"])
            sequence_array = np.asarray(evidence["sequence"])
            source_array = np.asarray(
                evidence["source_portal_block_world_position"]
            )
            from_dimension = str(evidence["from_dimension"])
            to_dimension = str(evidence["to_dimension"])
        except (KeyError, TypeError, ValueError):
            return None
        if (
            present_array.shape != ()
            or present_array.dtype.kind != "b"
            or entered_array.shape != ()
            or entered_array.dtype.kind != "b"
            or sequence_array.shape != ()
            or sequence_array.dtype.kind not in {"i", "u"}
        ):
            return None
        present = bool(present_array.item())
        entered_via_portal = bool(entered_array.item())
        sequence = int(sequence_array.item())
        if not present:
            return None
        if sequence < 1 or source_array.shape != (3,):
            return None
        if source_array.dtype.kind not in {"i", "u"}:
            return None
        if (
            from_dimension != "minecraft:overworld"
            or to_dimension != "minecraft:the_nether"
        ):
            return False
        if entered_via_portal is False:
            return False
        identity = self._latched.get("frame_identity")
        if not isinstance(identity, Mapping):
            return None
        bounds = self._world_bounds_from_identity(identity)
        if bounds is None:
            return None
        source = tuple(int(value) for value in source_array.tolist())
        min_x, min_y, min_z, max_x, max_y, max_z = bounds
        source_matches = (
            min_x <= source[0] <= max_x
            and min_y <= source[1] <= max_y
            and min_z <= source[2] <= max_z
        )
        return source_matches and self._position_matches_latched_frame(position)

    def _transition_matches_typed_casting_frame(
        self,
        raw: Mapping[str, Any],
        position: tuple[float, float, float] | None,
    ) -> bool | None:
        """Validate transition evidence against the observed casting frame."""
        if self._latched.get("typed_frame_complete_step") is None:
            return None
        evidence = raw.get("portal_transition")
        anchor = self._latched.get("grid_world_anchor")
        if not isinstance(evidence, Mapping) or not isinstance(anchor, tuple):
            return None
        try:
            present = bool(np.asarray(evidence["present"]).item())
            entered = bool(np.asarray(evidence["entered_via_portal"]).item())
            sequence = int(np.asarray(evidence["sequence"]).item())
            source_array = np.asarray(
                evidence["source_portal_block_world_position"]
            )
            from_dimension = str(evidence["from_dimension"])
            to_dimension = str(evidence["to_dimension"])
        except (KeyError, TypeError, ValueError):
            return None
        if (
            not present
            or sequence < 1
            or source_array.shape != (3,)
            or source_array.dtype.kind not in {"i", "u"}
        ):
            return None
        if (
            not entered
            or from_dimension != CASTING_S_C5_SOURCE_DIMENSION
            or to_dimension != CASTING_S_C5_TARGET_DIMENSION
        ):
            return False
        interior_world = tuple(
            (
                anchor[0] + cell[0],
                anchor[1] + cell[1],
                anchor[2] + cell[2],
            )
            for cell in CASTING_S_C3_INTERIOR_CELLS
        )
        source = tuple(int(value) for value in source_array.tolist())
        if source not in interior_world or position is None:
            return False
        xs = [cell[0] for cell in interior_world]
        ys = [cell[1] for cell in interior_world]
        zs = [cell[2] for cell in interior_world]
        px, py, pz = position
        tolerance = 1.5
        return (
            min(xs) - tolerance <= px <= max(xs) + tolerance
            and min(ys) - tolerance <= py <= max(ys) + tolerance
            and min(zs) - tolerance <= pz <= max(zs) + tolerance
        )

    @staticmethod
    def _stat_value(
        raw: Mapping[str, Any],
        category: str,
        item: str,
    ) -> int:
        values = raw.get(category)
        if not isinstance(values, Mapping) or item not in values:
            return 0
        return int(np.asarray(values[item]).item())

    def _portal_grid_debug(self) -> dict[str, Any]:
        if self._env is None or not hasattr(self._env, "task"):
            return {}
        for observable in getattr(self._env.task, "observables", ()):
            if isinstance(observable, PortalGridObservation):
                return {
                    "portal_grid_payload_present": observable.last_payload_present,
                    "portal_grid_unknown_blocks": list(
                        observable.last_unknown_blocks
                    ),
                    "malmo_info_keys": list(observable.last_hero_keys),
                }
        return {}

    def _refresh_evaluation_milestones(
        self,
        *,
        detection: Any | None = None,
        raw: Mapping[str, Any] | None = None,
    ) -> None:
        if raw is None:
            raw = self._require_raw()
        if detection is None:
            grid = self._grid_from_raw(raw)
            baseline = self._baseline_grid
            if baseline is None:
                return
            detection = detect_portal_frame_from_int_grid(
                grid, BLOCK_ID_TO_NAME, baseline_grid=baseline
            )

        # ------------------------------------------------------------
        # 1. Attribution: split obsidian delta into agent-attributed
        #    vs external. Credits represent accepted obsidian placement
        #    actions for this observation boundary only. Exact-count
        #    fresh deltas are attributed; ambiguous deltas fail closed.
        # ------------------------------------------------------------
        baseline = self._baseline_grid
        if baseline is not None:
            current_obsidian = self._offsets_with_block(OBSIDIAN_ID, raw=raw)
            baseline_obsidian = self._grid_block_set(baseline, OBSIDIAN_ID)
            new_offsets = current_obsidian - baseline_obsidian
            if new_offsets:
                self._latched["episode_obsidian_offsets"].update(new_offsets)
                if self._latched["first_obsidian_placed_step"] is None:
                    self._latched["first_obsidian_placed_step"] = self._step_id
                    self._latched["latched_timestamps"][
                        "first_obsidian_placed"
                    ] = time.time()
            # Only cells never classified before are eligible. A credit
            # is accepted only when the number of fresh cells exactly
            # matches the number of accepted actions for this step.
            # Ambiguous/surplus/delayed deltas fail closed as external.
            new_for_attribution = (
                current_obsidian
                - baseline_obsidian
                - self._latched["attributed_obsidian_offsets"]
                - self._latched["external_obsidian_offsets"]
            )
            pending = self._latched["pending_place_block_obsidian"]
            if new_for_attribution and len(new_for_attribution) == pending:
                self._latched["attributed_obsidian_offsets"].update(
                    new_for_attribution
                )
            else:
                self._latched["external_obsidian_offsets"].update(
                    new_for_attribution
                )
            # Credits never survive an observation boundary.
            self._latched["pending_place_block_obsidian"] = 0

        # Bind an accepted bucket action to world truth only when exactly one
        # tracked cell changes to that fluid in the returned server grid.
        # Merely issuing a bucket action, or an unrelated external mutation,
        # cannot earn per-cell evaluator credit.
        current_truth = self._grid_from_raw(raw).copy()
        previous_truth = self._latched.get("previous_truth_grid")
        baseline_truth = self._baseline_grid
        tracked_offsets = self._all_workflow_offsets()
        current_credit_kind: str | None = None
        history: list[tuple[int, str]] = self._latched["cast_credit_history"]
        if history and history[-1][0] == self._step_id:
            current_credit_kind = history[-1][1]
        if previous_truth is not None and current_credit_kind in {"water", "lava"}:
            expected_ids = (
                FLUID_WATER_IDS
                if current_credit_kind == "water"
                else FLUID_LAVA_IDS
            )
            changed_cells: list[tuple[int, int, int]] = []
            for offset in tracked_offsets:
                idx = self._workflow_cell_index(offset)
                if idx is None:
                    continue
                current_id = int(current_truth[idx])
                previous_id = int(previous_truth[idx])
                baseline_id = (
                    int(baseline_truth[idx])
                    if baseline_truth is not None
                    else FLUID_MISSING_ID
                )
                if (
                    current_id in expected_ids
                    and previous_id not in expected_ids
                    and baseline_id not in expected_ids
                ):
                    changed_cells.append(offset)
            if len(changed_cells) == 1:
                key = (
                    "first_water_step_by_offset"
                    if current_credit_kind == "water"
                    else "first_lava_step_by_offset"
                )
                self._latched[key].setdefault(
                    changed_cells[0], self._step_id
                )

        if previous_truth is not None:
            for offset in tracked_offsets:
                idx = self._workflow_cell_index(offset)
                if idx is None:
                    continue
                if (
                    int(current_truth[idx]) == OBSIDIAN_ID
                    and int(previous_truth[idx]) != OBSIDIAN_ID
                    and (
                        baseline_truth is None
                        or int(baseline_truth[idx]) != OBSIDIAN_ID
                    )
                ):
                    self._latched["first_obsidian_step_by_offset"].setdefault(
                        offset, self._step_id
                    )
        if (
            self._latched["typed_frame_complete_step"] is None
            and all(
                (idx := self._workflow_cell_index(offset)) is not None
                and int(current_truth[idx]) == OBSIDIAN_ID
                and offset in self._latched["first_water_step_by_offset"]
                and offset in self._latched["first_lava_step_by_offset"]
                and offset in self._latched["first_obsidian_step_by_offset"]
                for offset in CASTING_S_C3_FRAME_CELLS
            )
        ):
            self._latched["typed_frame_complete_step"] = self._step_id
        ignition_target = (1, 1, 1)
        ignition_idx = self._workflow_cell_index(ignition_target)
        if (
            current_credit_kind == "flint_and_steel"
            and previous_truth is not None
            and ignition_idx is not None
            and int(current_truth[ignition_idx]) == NETHER_PORTAL_ID
            and int(previous_truth[ignition_idx]) != NETHER_PORTAL_ID
            and self._latched["typed_frame_complete_step"] is not None
            and self._latched["first_nether_portal_step"] is None
        ):
            self._latched["first_nether_portal_step"] = self._step_id
        self._latched["current_fluid_grid"] = current_truth.copy()
        self._latched["previous_truth_grid"] = current_truth

        self._latched["max_obsidian_added"] = max(
            self._latched["max_obsidian_added"],
            len(self._latched["episode_obsidian_offsets"]),
        )

        # ------------------------------------------------------------
        # 2. Build site selected: only via partial-candidate with
        #    structural-continuity rule, and only from episode-added
        #    obsidian (the detector already excludes baseline cells).
        # ------------------------------------------------------------
        if self._latched["build_site_selected_step"] is None:
            for candidate in detection.partial_candidates:
                self._latched["build_site_selected_step"] = self._step_id
                self._latched["latched_timestamps"][
                    "build_site_selected"
                ] = time.time()
                break

        # ------------------------------------------------------------
        # 3. Episode-built frame: latched only when (a) the detector
        #    reports a selected candidate AND (b) every required cell
        #    of that candidate is in the attributed offsets set. If
        #    the bridge cannot attribute the obsidian to allowed
        #    actions, the frame is treated as attribution-failed.
        # ------------------------------------------------------------
        if (
            self._latched["first_valid_frame_step"] is None
            and detection.selected is not None
        ):
            candidate = detection.selected
            required = {
                (cell.x, cell.y, cell.z)
                for cell in candidate.required_frame_blocks
            }
            if required.issubset(self._latched["attributed_obsidian_offsets"]):
                self._latched["first_valid_frame_step"] = self._step_id
                self._latched["latched_timestamps"][
                    "valid_portal_frame"
                ] = time.time()
                self._latched["frame_identity"] = candidate.as_evidence()
            else:
                self._latched["attribution_failed_observed"] = True

        if detection.geometric_valid_candidates:
            self._latched["frame_candidate_observed"] = True
        if detection.attribution_failed_candidates:
            self._latched["attribution_failed_observed"] = True

        # ------------------------------------------------------------
        # 3a. External-structure detection. A geometrically valid
        #     frame whose required cells are not all in the agent
        #     attribution set is an "external structure": the
        #     environment grew the frame without a corresponding
        #     place_block action. The detector itself only knows
        #     about baseline vs current obsidian, so the backend
        #     has to compute this derived count by intersecting
        #     each candidate's required cells with the external
        #     obsidian offsets. Two sub-cases count:
        #
        # 1. Fully external: every required cell is in
        #    ``external_obsidian_offsets`` (the bridge never
        #    produced a place_block action that could have
        #    produced any of the required cells).
        # 2. Mixed: some required cells are attributed and some
        #    are external (the agent placed a few obsidian but
        #    the rest came from the world). The frame is
        #    geometrically valid but is NOT the agent's build.
        # ------------------------------------------------------------
        external_structure_candidate_count = 0
        external_obsidian = self._latched["external_obsidian_offsets"]
        attributed_obsidian = self._latched["attributed_obsidian_offsets"]
        for candidate in detection.geometric_valid_candidates:
            required = {
                (cell.x, cell.y, cell.z)
                for cell in candidate.required_frame_blocks
            }
            if not required:
                continue
            if required.issubset(external_obsidian):
                external_structure_candidate_count += 1
                continue
            if external_obsidian and not required.issubset(
                attributed_obsidian
            ):
                external_structure_candidate_count += 1
        self._latched["external_structure_candidate_count"] = (
            external_structure_candidate_count
        )

        # Activation is bound to the latched frame identity.
        self._maybe_latch_activation(detection=detection, raw=raw)

        # ------------------------------------------------------------
        # 4. Nether transition correlation.
        # ------------------------------------------------------------
        position = self._position(raw)
        if self._dimension(raw) == "minecraft:overworld":
            self._latched["last_overworld_position"] = position
        if self._dimension(raw) == "minecraft:the_nether":
            agent_id = "agent_1"
            nether_latched = self._latched["first_nether_step_by_agent"]
            if agent_id not in nether_latched:
                nether_latched[agent_id] = self._step_id
                self._latched["latched_timestamps"][
                    f"agent_entered_nether:{agent_id}"
                ] = time.time()
            # Record and evaluate the transition exactly once. Later
            # Nether observations cannot retroactively replace a
            # missing/negative first-transition verdict.
            first_transition = agent_id not in self._latched[
                "transition_step_by_agent"
            ]
            if first_transition:
                self._latched["pre_transition_position_by_agent"][agent_id] = (
                    self._latched["last_overworld_position"]
                    or position
                )
                self._latched["transition_step_by_agent"][agent_id] = (
                    self._step_id
                )
                verdict = self._transition_matches_latched_frame(
                    raw,
                    self._latched["pre_transition_position_by_agent"][agent_id],
                )
                if verdict is None:
                    verdict = self._transition_matches_typed_casting_frame(
                        raw,
                        self._latched[
                            "pre_transition_position_by_agent"
                        ][agent_id],
                    )
                if verdict is not None:
                    self._latched[
                        "entered_via_episode_portal_by_agent"
                    ][agent_id] = verdict
                if verdict is True:
                    self._latched["matched_frame_identity_by_agent"][
                        agent_id
                    ] = dict(
                        self._latched["frame_identity"]
                        or {"typed_casting_frame": True}
                    )

    def _maybe_latch_activation(
        self,
        *,
        detection: Any,
        raw: Mapping[str, Any],
    ) -> None:
        """Latch activation only if the latched frame identity's
        interior contains ``nether_portal`` in the current grid.
        """
        if self._latched["portal_activated"]:
            return
        identity = self._latched["frame_identity"]
        if identity is None:
            return
        interior_offsets = identity.get("interior_block_offsets", [])
        if not interior_offsets:
            return
        grid = self._grid_from_raw(raw)
        activated_offsets: list[tuple[int, int, int]] = []
        for raw_offset in interior_offsets:
            x, y, z = raw_offset
            if not (
                0 <= x < grid.shape[0]
                and 0 <= y < grid.shape[1]
                and 0 <= z < grid.shape[2]
            ):
                continue
            if int(grid[x, y, z]) == NETHER_PORTAL_ID:
                activated_offsets.append((x, y, z))
        if activated_offsets:
            self._latched["portal_activated"] = True
            self._latched["first_activation_step"] = self._step_id
            self._latched["latched_activation_offsets"] = activated_offsets
            self._latched["latched_timestamps"][
                "portal_activated"
            ] = time.time()
        # Always record the first-observed nether_portal block on
        # any frame interior cell. The C4 ignition evaluator
        # requires the activation ``update_step`` to be the step at
        # which the portal first appeared in the frame interior;
        # that information is independent from the activation
        # itself and must be latched once per episode. The check
        # is cheap because the interior has at most 6 cells.
        if self._latched["first_nether_portal_step"] is None:
            for raw_offset in interior_offsets:
                x, y, z = raw_offset
                if not (
                    0 <= x < grid.shape[0]
                    and 0 <= y < grid.shape[1]
                    and 0 <= z < grid.shape[2]
                ):
                    continue
                if int(grid[x, y, z]) == NETHER_PORTAL_ID:
                    self._latched["first_nether_portal_step"] = (
                        self._step_id
                    )
                    break

    def _build_evaluation_state(
        self,
        *,
        detection: Any,
        raw: Mapping[str, Any],
    ) -> EvaluationState:
        task = self._require_task()
        current_counts = Counter(self._decode_grid(self._grid_from_raw(raw)))
        baseline_counts = (
            Counter(self._decode_grid(self._baseline_grid))
            if self._baseline_grid is not None
            else Counter()
        )
        obsidian_added = max(
            0, current_counts["obsidian"] - baseline_counts["obsidian"]
        )
        location = raw.get("location_stats")
        position: dict[str, float] = {}
        if isinstance(location, Mapping):
            for key in ("xpos", "ypos", "zpos", "yaw", "pitch"):
                if key in location:
                    position[key] = float(np.asarray(location[key]).item())

        evidence: dict[str, Any] = {
            "portal_grid_size": int(self._grid_from_raw(raw).size),
            "obsidian_added": obsidian_added,
            "max_obsidian_added": self._latched["max_obsidian_added"],
            "nether_portal_blocks": int(current_counts["nether_portal"]),
            "portal_activated_latched": bool(self._latched["portal_activated"]),
            "dimension": self._dimension(raw),
            "use_item_stats": {
                "obsidian": self._stat_value(raw, "use_item", "obsidian"),
                "flint_and_steel": self._stat_value(
                    raw, "use_item", "flint_and_steel"
                ),
            },
            "position": position,
            "grid_world_anchor": self._latched["grid_world_anchor"],
            "portal_grid_counts": {
                key: int(value)
                for key, value in sorted(current_counts.items())
                if value
            },
            "portal_grid_changes": self._grid_changes(
                self._grid_from_raw(raw), self._baseline_grid
            ),
            **self._portal_grid_debug(),
            "frame_candidate_count": len(detection.candidates),
            "partial_candidate_count": len(detection.partial_candidates),
            "geometric_valid_candidate_count": len(
                detection.geometric_valid_candidates
            ),
            "attribution_failed_candidate_count": len(
                detection.attribution_failed_candidates
            ),
            "external_structure_candidate_count": int(
                self._latched["external_structure_candidate_count"]
            ),
            "episode_built_candidate_count": len(
                detection.episode_built_candidates
            ),
            "has_missing_truth_latched": bool(detection.has_missing_truth),
            "missing_frame_cell_count": int(
                detection.missing_frame_cell_count
            ),
            "missing_interior_cell_count": int(
                detection.missing_interior_cell_count
            ),
        }
        latched_identity = self._latched["frame_identity"]
        if latched_identity is not None:
            evidence["frame_selected_evidence"] = dict(latched_identity)
        else:
            evidence["frame_selected_evidence"] = None
        if self._latched["first_activation_step"] is not None:
            evidence["activation_evidence"] = {
                "frame_offsets": (
                    [list(latched_identity["min_corner"])]
                    if latched_identity is not None
                    else []
                ),
                "interior_offsets": [
                    list(offset)
                    for offset in self._latched["latched_activation_offsets"]
                ],
            }
        if self._latched["build_site_selected_step"] is not None:
            evidence["build_site_selected_evidence"] = {
                "latched_step": self._latched["build_site_selected_step"],
            }

        portal_built = self._latched["frame_identity"] is not None
        portal_activated = bool(self._latched["portal_activated"])

        return EvaluationState(
            episode_id=task.task_id,
            step_id=self._step_id,
            portal_built_by_episode=portal_built,
            valid_portal_frame=portal_built,
            portal_activated=portal_activated,
            agents_in_nether=frozenset(
                self._latched["first_nether_step_by_agent"].keys()
            ),
            task_reset_step=self._latched["task_reset_step"],
            first_obsidian_placed_step=self._latched["first_obsidian_placed_step"],
            build_site_selected_step=self._latched["build_site_selected_step"],
            first_valid_frame_step=self._latched["first_valid_frame_step"],
            first_activation_step=self._latched["first_activation_step"],
            first_nether_step_by_agent=dict(
                self._latched["first_nether_step_by_agent"]
            ),
            episode_terminated=bool(self._latched["episode_terminated"]),
            terminated_step=self._latched["terminated_step"],
            terminated_reason=self._latched["terminated_reason"],
            latched_frame_identity=latched_identity,
            latched_activation_offsets=tuple(
                self._latched["latched_activation_offsets"]
            ),
            latched_timestamps=dict(self._latched["latched_timestamps"]),
            attributed_obsidian_offsets=tuple(
                sorted(self._latched["attributed_obsidian_offsets"])
            ),
            external_obsidian_offsets=tuple(
                sorted(self._latched["external_obsidian_offsets"])
            ),
            pending_place_block_obsidian=self._latched[
                "pending_place_block_obsidian"
            ],
            pre_transition_position_by_agent={
                agent_id: tuple(pos)
                for agent_id, pos in self._latched[
                    "pre_transition_position_by_agent"
                ].items()
            },
            transition_step_by_agent=dict(
                self._latched["transition_step_by_agent"]
            ),
            entered_via_episode_portal_by_agent=dict(
                self._latched["entered_via_episode_portal_by_agent"]
            ),
            matched_frame_identity_by_agent={
                agent_id: dict(identity)
                for agent_id, identity in self._latched[
                    "matched_frame_identity_by_agent"
                ].items()
            },
            episode_obsidian_count=len(
                self._latched["episode_obsidian_offsets"]
            ),
            episode_obsidian_offsets=tuple(
                sorted(self._latched["episode_obsidian_offsets"])
            ),
            evidence=evidence,
        )

    def _offset_has_obsidian(
        self,
        offset: Any,
        *,
        raw: Mapping[str, Any] | None = None,
    ) -> bool:
        if raw is None:
            raw = self._require_raw()
        grid = self._grid_from_raw(raw)
        x, y, z = (
            (offset.x, offset.y, offset.z)
            if hasattr(offset, "x")
            else tuple(offset)
        )
        if not (
            0 <= x < grid.shape[0]
            and 0 <= y < grid.shape[1]
            and 0 <= z < grid.shape[2]
        ):
            return False
        return int(grid[x, y, z]) == OBSIDIAN_ID

    def _offsets_with_block(
        self,
        block_id: int,
        *,
        raw: Mapping[str, Any],
    ) -> set[tuple[int, int, int]]:
        grid = self._grid_from_raw(raw)
        offsets: set[tuple[int, int, int]] = set()
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                for z in range(grid.shape[2]):
                    if int(grid[x, y, z]) == block_id:
                        offsets.add((x, y, z))
        return offsets

    def _grid_block_set(
        self,
        grid: np.ndarray,
        block_id: int,
    ) -> set[tuple[int, int, int]]:
        offsets: set[tuple[int, int, int]] = set()
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                for z in range(grid.shape[2]):
                    if int(grid[x, y, z]) == block_id:
                        offsets.add((x, y, z))
        return offsets

    @staticmethod
    def _grid_changes(
        grid: np.ndarray,
        baseline: np.ndarray,
    ) -> list[dict[str, Any]]:
        changes: list[dict[str, Any]] = []
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                for z in range(grid.shape[2]):
                    before_id = int(baseline[x, y, z])
                    after_id = int(grid[x, y, z])
                    if before_id == after_id:
                        continue
                    before = PORTAL_GRID_BLOCKS[
                        before_id if 0 <= before_id < len(PORTAL_GRID_BLOCKS)
                        else PORTAL_GRID_BLOCKS.index("other")
                    ]
                    after = PORTAL_GRID_BLOCKS[
                        after_id if 0 <= after_id < len(PORTAL_GRID_BLOCKS)
                        else PORTAL_GRID_BLOCKS.index("other")
                    ]
                    changes.append(
                        {
                            "offset": [
                                PORTAL_GRID_MIN[0] + x,
                                PORTAL_GRID_MIN[1] + y,
                                PORTAL_GRID_MIN[2] + z,
                            ],
                            "before": before,
                            "after": after,
                        }
                    )
        return changes


__all__ = ["MineRLEnvironmentBackend", "BLOCK_ID_TO_NAME"]

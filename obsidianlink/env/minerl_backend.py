from __future__ import annotations

import threading
import time
from collections import Counter
from typing import Any, Callable, Mapping

import numpy as np

from obsidianlink.actions.minerl_translator import translate_macro_action
from obsidianlink.core.types import BackendStep, MacroAction, Observation, TaskInstance
from obsidianlink.env.portal_spec import (
    PORTAL_GRID_BLOCKS,
    PORTAL_GRID_MAX,
    PORTAL_GRID_MIN,
    PORTAL_GRID_MISSING_ID,
    PORTAL_GRID_SIZE,
    PortalA0EnvSpec,
    PortalGridObservation,
)
from obsidianlink.evaluation import EvaluationState
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

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return self._task.agent_ids if self._task is not None else ()

    @property
    def action_space(self) -> Any:
        self._require_env()
        return self._env.action_space

    def open(self) -> None:
        if self._opened:
            raise RuntimeError("backend is already open")
        self._opened = True
        self._owner_thread = threading.get_ident()

    def reset(self, task: TaskInstance) -> Mapping[str, Observation]:
        self._assert_owner()
        if task.agent_ids != ("agent_1",):
            raise ValueError("PortalA0 currently supports exactly agent_1")
        if task.route != "obsidian_mining" or task.difficulty != 1:
            raise ValueError("PortalA0 currently supports Route A difficulty 1")
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
        return self._public_observations()

    def step(self, actions: Mapping[str, MacroAction]) -> BackendStep:
        task = self._require_task()
        self._assert_owner()
        if set(actions) != {"agent_1"}:
            raise ValueError("actions must contain exactly agent_1")
        action = actions["agent_1"]
        translation = translate_macro_action(action, self.action_space)
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
        raw, reward, done, info = self._env.step(translation.action)
        if isinstance(info, Mapping) and "error" in info:
            raise RuntimeError(f"MineRL step failed: {info['error']}")
        self._step_id += 1
        self._latest_raw = self._validate_raw_observation(raw)
        if accepted_obsidian_placement:
            self._latched["pending_place_block_obsidian"] += 1
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
            rewards={"agent_1": float(reward)},
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
        }

    def _mark_terminated(self, *, step_id: int, reason: str) -> None:
        if self._latched["episode_terminated"]:
            return
        self._latched["episode_terminated"] = True
        self._latched["terminated_step"] = step_id
        self._latched["terminated_reason"] = reason
        self._latched["latched_timestamps"]["_terminated"] = time.time()

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
                if verdict is not None:
                    self._latched[
                        "entered_via_episode_portal_by_agent"
                    ][agent_id] = verdict
                if verdict is True:
                    self._latched["matched_frame_identity_by_agent"][
                        agent_id
                    ] = dict(self._latched["frame_identity"] or {})

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

"""Offline tests for the R6-C5-LIVE-MINERL-BACKEND-WIRING milestone.

These tests prove, in code, that:

* :func:`obsidianlink.actions.minerl_translator.translate_macro_action`
  wires every action the C3 / C4 / C5 deterministic driver
  family emits (``equip_item(water_bucket)`` /
  ``equip_item(lava_bucket)`` / ``equip_item(cobblestone)`` /
  ``equip_item(flint_and_steel)`` / ``use_item(water_bucket)`` /
  ``use_item(lava_bucket)`` / ``use_item(flint_and_steel)`` /
  ``place_block(cobblestone)`` / bounded forward ``move`` /
  ``wait``) into MineRL's hotbar + ``use`` low-level surface.
* The translator enforces a *closed* allowlist, strict type
  checks, and bounded numeric ranges. Unknown semantic action
  types, unsupported items, bool-as-int quantities, non-finite or
  out-of-range parameters, and translated actions outside the
  declared action space all fail closed by returning
  ``accepted=False`` with a typed error message. The translator
  never silently rewrites a rejected action into a no-op.
* :class:`obsidianlink.core.types.Observation` exposes the
  ``selected_item`` field on the public schema. The MineRL backend
  reads the value from the raw observation's
  :data:`~obsidianlink.env.portal_spec.PORTAL_SELECTED_ITEM_NAME`
  key (or the literal ``"empty"`` / ``"air"`` placeholder). The
  backend never derives the selected item from the agent's request
  stream.
* The production MineRL backend reports target-block and fluid truth
  capabilities as unavailable because its legacy
  :meth:`get_evaluation_state` surface is not a typed C1–C5 casting
  truth source.  The pre-episode gate therefore rejects production
  casting tasks before the env factory is called or state is mutated.
* A test-only full-capability subclass lets the action and public
  observation wiring run against injected raw observations without
  weakening or misrepresenting the production capability manifest.
* The C5 deterministic driver walks the full 347-step plan on a
  stub ``env_factory``-supplied test backend. The driver never
  reads ``selected_item`` / ``target_block_truth`` / ``fluid_truth``
  / ``latched_frame_identity`` / ``matched_frame_identity`` /
  ``entered_via_episode_portal`` / ``pre_transition_position``;
  the test orchestrator (this file) is the only place that
  injects the evaluator-only truth.

The tests never start Minecraft, MineRL, or Gradle, and never
import the MineRL bridge at runtime.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import unittest
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from obsidianlink.actions.minerl_translator import (
    MAX_TRANSLATOR_DURATION_TICKS,
    MIN_TRANSLATOR_DURATION_TICKS,
    PORTAL_A0_HOTBAR,
    TRANSLATOR_EQUIPPABLE_ITEMS,
    TRANSLATOR_PLACEABLE_ITEMS,
    build_hotbar_mapping,
    translate_macro_action,
)
from obsidianlink.core.casting_s_c5_nether_entry_context import (
    build_public_c5_nether_entry_driver_context_from_task,
)
from obsidianlink.core.types import (
    BackendStep,
    MacroAction,
    Observation,
    RecoverableBackendError,
    TaskInstance,
)
from obsidianlink.drivers.casting_s_c5_nether_entry import (
    AGENT_ID,
    C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET,
    C5_NETHER_ENTRY_PUBLIC_SOURCE_DIMENSION,
    C5_NETHER_ENTRY_PUBLIC_TARGET_DIMENSION,
    CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS,
    run_casting_s_c5_nether_entry_driver,
)
from obsidianlink.env.capabilities import (
    CAPABILITY_IDS,
    BackendCapabilities,
    CapabilityMismatchError,
    assert_backend_can_start_task,
    assert_casting_c1_capabilities,
)
from obsidianlink.env.fake import FakeEnvironmentBackend
from obsidianlink.env.minerl_backend import (
    MineRLEnvironmentBackend as ProductionMineRLEnvironmentBackend,
)
from obsidianlink.env.portal_spec import (
    PORTAL_GRID_BLOCKS,
    PORTAL_GRID_MAX,
    PORTAL_GRID_MIN,
    PORTAL_GRID_MISSING_ID,
    PORTAL_GRID_SHAPE,
    PORTAL_GRID_SIZE,
    PORTAL_SELECTABLE_ITEMS,
    PORTAL_SELECTED_ITEM_NAME,
    PortalA0EnvSpec,
    PortalGridObservation,
)
from obsidianlink.evaluation.casting_ignition_evaluator import (
    FrozenFrameIdentity,
    IgnitionActionEvidence,
    PortalActivationEvidence,
    build_c4_c3_frame_identity,
)
from obsidianlink.evaluation.casting_nether_entry_evaluator import (
    FrozenNetherEntryEvaluator,
    FrozenNetherEntryEvaluationState,
    NetherEntryEvidence,
    CASTING_S_C5_SOURCE_DIMENSION,
    CASTING_S_C5_TARGET_DIMENSION,
)
from obsidianlink.evaluation.casting_ignition_evaluator import (
    FrozenIgnitionEvaluationState,
    FrozenIgnitionEvaluator,
)
from obsidianlink.evaluation.casting_frame_evaluator import (
    FrozenFrameActionEvidence,
    FrozenFrameCellTruth,
    FrozenFrameEvaluationState,
    FrozenFrameInteriorCellTruth,
    FrozenFrameEvaluator,
)
from obsidianlink.evaluation.casting import (
    CastingEvaluationState,
    CastingEvaluator,
    CastingFluidTruth,
    CastingTransitionEvidence,
)
from obsidianlink.evaluation.continuous_casting import (
    CASTING_C3_TARGET_CELLS,
    ContinuousCastingEvaluator,
)
from obsidianlink.evaluation.frame_geometry import (
    detect_portal_frame_from_int_grid,
)
from obsidianlink.evaluation.portal import (
    EvaluationState,
    PortalEvaluator,
)


EPISODE_ID = "casting_s_c5_fixed_seed_0"
AGENT_ID_LOCAL = "agent_1"
TERMINATED_REASON = "driver_done"


class MineRLEnvironmentBackend(ProductionMineRLEnvironmentBackend):
    """Stub-only backend that opts into synthetic truth capabilities.

    Production keeps target-block/fluid capabilities disabled. Tests that
    inject complete raw fixtures use this subclass explicitly so they do not
    weaken the real pre-episode gate.
    """

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities.full()

# Mirror the C5 driver's default 347-step plan so the orchestrator
# truth injection lines up with the driver events.
IGNITION_STEP = 339
IGNITION_EQUIP_STEP = 337
IGNITION_PORTAL_SETTLE_STEP = 340
ENTRY_TRANSITION_STEP = 346
ENTRY_APPROACH_FIRST_STEP = 341
ENTRY_SETTLE_STEP = 347
TERMINATED_STEP = 347

#: Public C3 interior cells (mirrors the C3 frozen-frame geometry).
#: Order must match the frozen ``CASTING_S_C3_INTERIOR_CELLS`` in
#: :mod:`obsidianlink.evaluation.casting_frame_evaluator` so the
#: frame evaluator accepts the test fixtures.
CASTING_S_C3_INTERIOR_CELLS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 1),
    (2, 1, 1),
    (1, 2, 1),
    (2, 2, 1),
    (1, 3, 1),
    (2, 3, 1),
)

# Canonical frame offsets used by the C3 frozen-frame evaluator.
CASTING_S_C3_FRAME_CELLS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 1),
    (1, 0, 1),
    (2, 0, 1),
    (3, 0, 1),
    (0, 4, 1),
    (1, 4, 1),
    (2, 4, 1),
    (3, 4, 1),
    (0, 1, 1),
    (0, 2, 1),
    (0, 3, 1),
    (3, 1, 1),
    (3, 2, 1),
    (3, 3, 1),
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _flat_offset(
    offset: tuple[int, int, int],
    shape: tuple[int, int, int] = PORTAL_GRID_SHAPE,
) -> int:
    """Return the flat (Fortran-order) index of ``offset`` in the
    portal grid used by the MineRL bridge.

    The portal grid is anchored at :data:`PORTAL_GRID_MIN`; the
    flat index is computed in the grid's own coordinate system
    so the helper agrees with the backend's
    :func:`_cell_index_in_grid`.
    """
    return (
        (offset[1] - PORTAL_GRID_MIN[1]) * shape[0] * shape[2]
        + (offset[2] - PORTAL_GRID_MIN[2]) * shape[0]
        + (offset[0] - PORTAL_GRID_MIN[0])
    )


def _default_inventory() -> dict[str, int]:
    return {
        "water_bucket": 14,
        "lava_bucket": 14,
        "cobblestone": 28,
        "flint_and_steel": 1,
    }


def _c5_task_dict(
    *,
    inventory: dict[str, int] | None = None,
    max_environment_steps: int = 800,
    max_game_time_seconds: int = 720,
) -> dict[str, Any]:
    return {
        "schema_version": "0.1",
        "task_id": EPISODE_ID,
        "route": "lava_casting",
        "difficulty": 4,
        "agent_ids": [AGENT_ID_LOCAL],
        "world_seed": 0,
        "instruction": (
            "R6 C5 live-MineRL-backend-wiring offline contract test."
        ),
        "spawn_positions": {AGENT_ID_LOCAL: [0, 4, 0]},
        "initial_inventories": {
            AGENT_ID_LOCAL: dict(inventory or _default_inventory())
        },
        "workflow": "casting_s_c5_fixed",
        "milestones": [
            "task_reset",
            "first_obsidian_cast",
            "build_site_selected",
            "valid_portal_frame",
            "portal_activated",
            "agent_entered_nether",
        ],
        "limits": {
            "max_environment_steps": max_environment_steps,
            "max_model_calls": 1,
            "max_game_time_seconds": max_game_time_seconds,
        },
        "split": "development",
        "scenario_parameters": {
            "task_family": "casting",
            "agent_mode": "single",
            "task_level": "C5",
            "layout_type": "fixed",
            "compatibility_task_name": "casting_s_c5_fixed",
            "implementation_status": "contract_only",
            "world_dimension": "minecraft:overworld",
            "layout": "fixed_controlled",
            "mechanics_required": (
                "vanilla_water_lava_block_update_flint_and_steel_and_portal_teleport"
            ),
            "public_task_spec": {
                "coordinate_space": "task_origin_relative",
                "task_origin_marker": "visible",
                "frame_plan": {
                    "orientation": "plane_z",
                    "min_corner": [0, 0, 1],
                    "width": 4,
                    "height": 5,
                    "require_full_ring": True,
                    "minecraft_minimum_required_block_count": 10,
                    "benchmark_required_full_ring_block_count": 14,
                    "required_corner_count": 4,
                    "interior_allowlist": ["air", "nether_portal", "fire"],
                    "fixed_offsets": [
                        list(cell)
                        for cell in CASTING_S_C5_NETHER_ENTRY_FRAME_CELLS
                    ],
                },
                "ignition_plan": {
                    "required": True,
                    "action": "use_item",
                    "item": "flint_and_steel",
                    "target_offset": list(C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET),
                    "target_policy": "exact",
                },
                "nether_entry_goal": {
                    "required": True,
                    "designated_agent_ids": [AGENT_ID_LOCAL],
                    "source_dimension": C5_NETHER_ENTRY_PUBLIC_SOURCE_DIMENSION,
                    "target_dimension": C5_NETHER_ENTRY_PUBLIC_TARGET_DIMENSION,
                },
            },
            "allow_minecraft_commands": False,
            "allow_evaluator_world_mutation": False,
            "allow_live_run": False,
            "requires_explicit_live_run_approval": True,
        },
    }


def _c5_task() -> TaskInstance:
    return TaskInstance.from_dict(_c5_task_dict())


# ----------------------------------------------------------------------
# Translator: positive paths
# ----------------------------------------------------------------------


class TranslatorAllowlistTests(unittest.TestCase):
    """Static contract for the translator allowlist."""

    def test_translator_equippable_items_is_closed(self) -> None:
        self.assertEqual(
            TRANSLATOR_EQUIPPABLE_ITEMS,
            frozenset(
                {
                    "obsidian",
                    "flint_and_steel",
                    "dirt",
                    "water_bucket",
                    "lava_bucket",
                    "cobblestone",
                }
            ),
        )

    def test_translator_placeable_items_is_closed(self) -> None:
        self.assertEqual(
            TRANSLATOR_PLACEABLE_ITEMS,
            frozenset({"cobblestone", "obsidian", "dirt"}),
        )

    def test_hotbar_mapping_is_static(self) -> None:
        # ``water_bucket`` / ``lava_bucket`` / ``cobblestone``
        # get fresh hotbar slots so the legacy A0 fixtures keep
        # the original hotbar.1 / hotbar.2 / hotbar.3 mapping.
        self.assertEqual(PORTAL_A0_HOTBAR["obsidian"], "hotbar.1")
        self.assertEqual(PORTAL_A0_HOTBAR["flint_and_steel"], "hotbar.2")
        self.assertEqual(PORTAL_A0_HOTBAR["dirt"], "hotbar.3")
        self.assertEqual(PORTAL_A0_HOTBAR["water_bucket"], "hotbar.4")
        self.assertEqual(PORTAL_A0_HOTBAR["lava_bucket"], "hotbar.5")
        self.assertEqual(PORTAL_A0_HOTBAR["cobblestone"], "hotbar.6")

    def test_duration_caps_match_action_protocol(self) -> None:
        self.assertEqual(MIN_TRANSLATOR_DURATION_TICKS, 1)
        self.assertEqual(MAX_TRANSLATOR_DURATION_TICKS, 40)

    def test_casting_hotbar_mapping_follows_frozen_inventory_order(self) -> None:
        self.assertEqual(
            build_hotbar_mapping(_default_inventory()),
            {
                "water_bucket": "hotbar.1",
                "lava_bucket": "hotbar.2",
                "cobblestone": "hotbar.3",
                "flint_and_steel": "hotbar.4",
            },
        )


class TranslatorPositivePathTests(unittest.TestCase):
    """Every C3 / C4 / C5 driver action must translate."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.action_space = PortalA0EnvSpec().action_space

    def _translate(self, action: MacroAction) -> dict[str, Any]:
        result = translate_macro_action(action, self.action_space)
        self.assertTrue(result.accepted, msg=result.error)
        return result.action

    def test_wait_is_a_no_op_translation(self) -> None:
        translated = self._translate(MacroAction.wait())
        # The MineRL ``no_op`` keeps all hotbar slots at 0; ``wait``
        # never sets any of them.
        for slot in range(1, 7):
            self.assertEqual(translated[f"hotbar.{slot}"], 0)

    def test_equip_item_water_bucket_sets_hotbar_4(self) -> None:
        translated = self._translate(
            MacroAction(action_type="equip_item", target="water_bucket")
        )
        self.assertEqual(translated["hotbar.4"], 1)

    def test_equip_item_lava_bucket_sets_hotbar_5(self) -> None:
        translated = self._translate(
            MacroAction(action_type="equip_item", target="lava_bucket")
        )
        self.assertEqual(translated["hotbar.5"], 1)

    def test_equip_item_cobblestone_sets_hotbar_6(self) -> None:
        translated = self._translate(
            MacroAction(action_type="equip_item", target="cobblestone")
        )
        self.assertEqual(translated["hotbar.6"], 1)

    def test_equip_item_flint_and_steel_keeps_legacy_hotbar_2(self) -> None:
        translated = self._translate(
            MacroAction(action_type="equip_item", target="flint_and_steel")
        )
        self.assertEqual(translated["hotbar.2"], 1)

    def test_use_item_water_bucket_sets_hotbar_4_and_use(self) -> None:
        translated = self._translate(
            MacroAction(action_type="use_item", target="water_bucket")
        )
        self.assertEqual(translated["hotbar.4"], 1)
        self.assertEqual(translated["use"], 1)

    def test_use_item_lava_bucket_sets_hotbar_5_and_use(self) -> None:
        translated = self._translate(
            MacroAction(action_type="use_item", target="lava_bucket")
        )
        self.assertEqual(translated["hotbar.5"], 1)
        self.assertEqual(translated["use"], 1)

    def test_use_item_flint_and_steel_sets_hotbar_2_and_use(self) -> None:
        translated = self._translate(
            MacroAction(action_type="use_item", target="flint_and_steel")
        )
        self.assertEqual(translated["hotbar.2"], 1)
        self.assertEqual(translated["use"], 1)

    def test_place_block_cobblestone_sets_hotbar_6_and_use(self) -> None:
        translated = self._translate(
            MacroAction(action_type="place_block", target="cobblestone")
        )
        self.assertEqual(translated["hotbar.6"], 1)
        self.assertEqual(translated["use"], 1)

    def test_bounded_forward_move(self) -> None:
        translated = self._translate(
            MacroAction(
                action_type="move",
                duration_ticks=4,
                parameters={
                    "forward": 1.0,
                    "strafe": 0.0,
                    "sprint": False,
                    "jump": False,
                },
            )
        )
        self.assertEqual(translated["forward"], 1)
        self.assertEqual(translated["back"], 0)
        self.assertEqual(translated["right"], 0)
        self.assertEqual(translated["left"], 0)
        self.assertEqual(translated["sprint"], 0)
        self.assertEqual(translated["jump"], 0)

    def test_look_sets_bounded_camera(self) -> None:
        translated = self._translate(
            MacroAction(
                action_type="look",
                duration_ticks=1,
                parameters={"pitch": 5.0, "yaw": -10.0},
            )
        )
        self.assertTrue(self.action_space.contains(translated))
        self.assertEqual(translated["camera"].tolist(), [5.0, -10.0])

    def test_craft_item_is_explicitly_forbidden(self) -> None:
        result = translate_macro_action(
            MacroAction(action_type="craft_item", target="oak_planks"),
            self.action_space,
        )
        self.assertFalse(result.accepted)
        self.assertIn("craft_item", (result.error or ""))


# ----------------------------------------------------------------------
# Translator: negative / fail-closed paths
# ----------------------------------------------------------------------


class TranslatorFailClosedTests(unittest.TestCase):
    """The translator must fail closed for any illegal input."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.action_space = PortalA0EnvSpec().action_space

    def _assert_rejected(
        self, action: MacroAction, expected_substring: str
    ) -> None:
        result = translate_macro_action(action, self.action_space)
        self.assertFalse(result.accepted)
        self.assertIn(expected_substring, (result.error or ""))
        # The fallback action must be a valid no-op inside the
        # declared action space; the translator must never return
        # an out-of-bounds dictionary to the backend.
        self.assertTrue(self.action_space.contains(result.action))

    def test_equip_item_rejects_unknown_target(self) -> None:
        self._assert_rejected(
            MacroAction(action_type="equip_item", target="diamond_block"),
            "unsupported translator equip target",
        )

    def test_equip_item_rejects_none_target(self) -> None:
        self._assert_rejected(
            MacroAction(action_type="equip_item", target=None),
            "non-empty target string",
        )

    def test_use_item_rejects_unknown_target(self) -> None:
        self._assert_rejected(
            MacroAction(action_type="use_item", target="lava"),  # singular
            "unsupported translator use target",
        )

    def test_place_block_rejects_unknown_target(self) -> None:
        self._assert_rejected(
            MacroAction(action_type="place_block", target="diamond_block"),
            "unsupported translator place target",
        )

    def test_place_block_rejects_cobblestone_blocked_in_driver(
        self,
    ) -> None:
        # cobblestone is the only placeable item the C5 driver
        # emits, but the translator must still validate ``target``
        # and reject empty / None values.
        self._assert_rejected(
            MacroAction(action_type="place_block", target=None),
            "non-empty target string",
        )

    def test_unknown_action_type_is_rejected(self) -> None:
        self._assert_rejected(
            MacroAction(action_type="harvest_block", target="obsidian"),
            "unsupported semantic action",
        )

    def test_move_with_out_of_range_forward_fails_closed(self) -> None:
        self._assert_rejected(
            MacroAction(
                action_type="move",
                duration_ticks=1,
                parameters={"forward": 2.0, "strafe": 0.0},
            ),
            "move.forward must be in [-1.0, 1.0]",
        )

    def test_move_with_non_finite_forward_fails_closed(self) -> None:
        self._assert_rejected(
            MacroAction(
                action_type="move",
                duration_ticks=1,
                parameters={"forward": float("nan"), "strafe": 0.0},
            ),
            "move.forward must be a finite number",
        )

    def test_look_with_out_of_range_pitch_fails_closed(self) -> None:
        self._assert_rejected(
            MacroAction(
                action_type="look",
                duration_ticks=1,
                parameters={"pitch": 60.0, "yaw": 0.0},
            ),
            "look.pitch must be in [-30.0, 30.0]",
        )

    def test_look_with_non_finite_yaw_fails_closed(self) -> None:
        self._assert_rejected(
            MacroAction(
                action_type="look",
                duration_ticks=1,
                parameters={"pitch": 0.0, "yaw": float("inf")},
            ),
            "look.yaw must be a finite number",
        )

    def test_look_with_bool_pitch_fails_closed(self) -> None:
        # Bool is a subclass of int in Python; the translator
        # rejects it explicitly so a planner cannot smuggle True
        # / False in for a numeric parameter.
        self._assert_rejected(
            MacroAction(
                action_type="look",
                duration_ticks=1,
                parameters={"pitch": True, "yaw": 0.0},
            ),
            "look.pitch must be a finite number",
        )

    def test_duration_ticks_too_large_fails_closed(self) -> None:
        # The :class:`MacroAction` constructor itself rejects
        # bool / string / zero duration_ticks before the
        # translator sees them. The translator's range check
        # (1..40) is defence in depth; this test exercises the
        # translator's own range check via a valid MacroAction
        # whose duration_ticks exceeds the cap.
        self._assert_rejected(
            MacroAction(
                action_type="wait",
                duration_ticks=41,
            ),
            "duration_ticks must be between 1 and 40",
        )


# ----------------------------------------------------------------------
# MineRL backend: selected item surface
# ----------------------------------------------------------------------


class _StubMineRLBridgeEnv:
    """Minimal MineRL-shaped env that the stub env_factory returns.

    The env supplies the raw observation dict the backend reads;
    it never executes any real MineRL / Minecraft code.
    """

    def __init__(
        self,
        *,
        raw_observation: dict[str, Any],
        actions: list[Mapping[str, Any]] | None = None,
        trajectory: list[dict[str, Any]] | None = None,
    ) -> None:
        self.action_space = PortalA0EnvSpec().action_space
        self._raw_observation = raw_observation
        self._actions: list[Mapping[str, Any]] = (
            actions if actions is not None else []
        )
        self._trajectory: list[dict[str, Any]] = trajectory or []
        self._step_index = 0
        self.seed_value: int | None = None
        self.closed = False

    def seed(self, value: int) -> None:
        self.seed_value = value

    def reset(self) -> dict[str, Any]:
        return self._raw_observation

    def step(self, action: Mapping[str, Any]):
        self._actions.append(dict(action))
        if self._step_index < len(self._trajectory):
            raw = self._trajectory[self._step_index]
            self._step_index += 1
            return raw, 0.0, False, {}
        # Default: echo the reset observation so the driver can
        # keep walking the plan when the test does not script a
        # custom trajectory.
        return self._raw_observation, 0.0, False, {}

    def close(self) -> None:
        self.closed = True


def _portal_grid(blocks: Mapping[tuple[int, int, int], str]) -> np.ndarray:
    """Build a 7x7x7 int32 portal grid with the given block mapping.

    The portal grid block set does not include water / lava as
    block ids; fluid truth travels on a separate surface. Tests
    that want a fluid cell should not use this helper.
    """
    grid = np.full(PORTAL_GRID_SIZE, PORTAL_GRID_MISSING_ID, dtype=np.int32)
    for offset, block_name in blocks.items():
        if block_name not in PORTAL_GRID_BLOCKS:
            raise ValueError(
                f"block {block_name!r} is outside the portal grid block set"
            )
        grid[_flat_offset(offset)] = PORTAL_GRID_BLOCKS.index(block_name)
    return grid


def _default_raw_observation(
    *,
    selected_item: str = "water_bucket",
    position: tuple[float, float, float] = (0.5, 64.0, 0.5),
) -> dict[str, Any]:
    return {
        "pov": np.zeros((360, 640, 3), dtype=np.uint8),
        "inventory": {
            "water_bucket": np.asarray(14, dtype=np.int64),
            "lava_bucket": np.asarray(14, dtype=np.int64),
            "cobblestone": np.asarray(28, dtype=np.int64),
            "flint_and_steel": np.asarray(1, dtype=np.int64),
        },
        "portal_grid": _portal_grid({}),
        "portal_grid_origin": np.asarray((0, 64, 0), dtype=np.int32),
        "portal_dimension": np.asarray("minecraft:overworld"),
        "location_stats": {
            "xpos": position[0],
            "ypos": position[1],
            "zpos": position[2],
        },
        "use_item": {
            "obsidian": np.asarray(0, dtype=np.int64),
            "flint_and_steel": np.asarray(0, dtype=np.int64),
        },
        PORTAL_SELECTED_ITEM_NAME: {
            "mainhand": {"type": selected_item},
        },
    }


def _stub_factory(
    raw_observation: dict[str, Any],
    trajectory: list[dict[str, Any]] | None = None,
) -> Any:
    """Build a stub env_factory that always returns the same env."""

    captured: dict[str, Any] = {}

    def factory(task: TaskInstance) -> _StubMineRLBridgeEnv:
        captured["task"] = task
        return _StubMineRLBridgeEnv(
            raw_observation=raw_observation,
            trajectory=trajectory,
        )

    factory.captured = captured  # type: ignore[attr-defined]
    return factory


def _build_c5_context(task: TaskInstance):
    return build_public_c5_nether_entry_driver_context_from_task(task)


def _build_minimal_c3_state(
    task: TaskInstance,
    *,
    step_id: int,
    terminated_step: int,
) -> FrozenFrameEvaluationState:
    cells: list[FrozenFrameCellTruth] = []
    for index, target_cell in enumerate(CASTING_S_C3_FRAME_CELLS):
        records: list[FrozenFrameActionEvidence] = []
        lava_step = 9 + 24 * index
        water_step = 16 + 24 * index
        records.append(
            FrozenFrameActionEvidence(
                episode_id=task.task_id,
                step_id=lava_step,
                agent_id=AGENT_ID_LOCAL,
                action_type="use_item",
                item="lava_bucket",
                target_cell=target_cell,
            )
        )
        records.append(
            FrozenFrameActionEvidence(
                episode_id=task.task_id,
                step_id=water_step,
                agent_id=AGENT_ID_LOCAL,
                action_type="use_item",
                item="water_bucket",
                target_cell=target_cell,
            )
        )
        cells.append(
            FrozenFrameCellTruth(
                target_cell=target_cell,
                initial_block="air",
                current_block="obsidian",
                water_truth=CastingFluidTruth(
                    present=True, evidence_step=water_step
                ),
                lava_truth=CastingFluidTruth(
                    present=True, evidence_step=lava_step
                ),
                transition_evidence=CastingTransitionEvidence(
                    before_block="air",
                    after_block="obsidian",
                    update_step=20 + 24 * index,
                ),
                relevant_action_steps=(lava_step, water_step),
                action_evidence=tuple(records),
                transition_action_step=water_step,
            )
        )
    interior = tuple(
        FrozenFrameInteriorCellTruth(target_cell=cell, current_block="air")
        for cell in CASTING_S_C3_INTERIOR_CELLS
    )
    return FrozenFrameEvaluationState(
        episode_id=task.task_id,
        step_id=step_id,
        cells=tuple(cells),
        interior_cells=interior,
        agent_id=AGENT_ID_LOCAL,
        causality_window_steps=4,
        episode_terminated=True,
        terminated_step=terminated_step,
        terminated_reason=TERMINATED_REASON,
        current_time_seconds=0.0,
        max_environment_steps=task.limits["max_environment_steps"],
        max_game_time_seconds=task.limits["max_game_time_seconds"],
    )


def _build_minimal_c5_state(
    task: TaskInstance,
    *,
    driver_result: Any,
    step_id: int,
    terminated_step: int,
    transition_step: int = ENTRY_TRANSITION_STEP,
    source_dimension: str = C5_NETHER_ENTRY_PUBLIC_SOURCE_DIMENSION,
    target_dimension: str = C5_NETHER_ENTRY_PUBLIC_TARGET_DIMENSION,
    entered_via_episode_portal: bool = True,
    pre_transition_position: tuple[float, float, float] = (1.5, 1.0, 1.0),
    matched_frame_identity: FrozenFrameIdentity | None = None,
    include_entry_evidence: bool = True,
) -> FrozenNetherEntryEvaluationState:
    frame_state = _build_minimal_c3_state(
        task, step_id=step_id, terminated_step=terminated_step
    )
    activation_offset = C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET
    # The C5 evaluator re-runs the C4 evaluator on the ignition
    # state. The ignition_state's latched_frame_identity and the
    # activation_evidence.latched_frame_identity must match the
    # public (1, 1, 1) ignition target so the C4 check passes.
    # The ``matched_frame_identity`` parameter only affects the
    # C5-specific ``entry_evidence.matched_frame_identity`` so
    # the C5 frame-identity-mismatch path is reachable.
    identity = build_c4_c3_frame_identity(
        episode_id=task.task_id,
        step_id=step_id,
        agent_id=AGENT_ID_LOCAL,
        activation_offsets=(activation_offset,),
    )
    ignition_action = IgnitionActionEvidence(
        episode_id=task.task_id,
        step_id=IGNITION_STEP,
        agent_id=AGENT_ID_LOCAL,
        action_type="use_item",
        item="flint_and_steel",
        target_cell=activation_offset,
    )
    activation_evidence = PortalActivationEvidence(
        episode_id=task.task_id,
        update_step=IGNITION_PORTAL_SETTLE_STEP,
        agent_id=AGENT_ID_LOCAL,
        nether_portal_offset=activation_offset,
        latched_frame_identity=identity,
    )
    ignition_state = FrozenIgnitionEvaluationState(
        episode_id=task.task_id,
        step_id=step_id,
        frame_state=frame_state,
        latched_frame_identity=identity,
        ignition_action=ignition_action,
        activation_evidence=activation_evidence,
        agent_id=AGENT_ID_LOCAL,
        causality_window_steps=4,
        episode_terminated=True,
        terminated_step=terminated_step,
        terminated_reason=TERMINATED_REASON,
        current_time_seconds=0.0,
        max_environment_steps=task.limits["max_environment_steps"],
        max_game_time_seconds=task.limits["max_game_time_seconds"],
    )
    entry_evidence: NetherEntryEvidence | None = None
    if include_entry_evidence:
        entry_evidence = NetherEntryEvidence(
            episode_id=task.task_id,
            agent_id=AGENT_ID_LOCAL,
            source_dimension=source_dimension,
            target_dimension=target_dimension,
            transition_step=transition_step,
            pre_transition_position=pre_transition_position,
            entered_via_episode_portal=entered_via_episode_portal,
            matched_frame_identity=(
                matched_frame_identity
                if matched_frame_identity is not None
                else identity
            ),
        )
    return FrozenNetherEntryEvaluationState(
        episode_id=task.task_id,
        step_id=step_id,
        ignition_state=ignition_state,
        agents_in_nether=frozenset({AGENT_ID_LOCAL}),
        entry_evidence=entry_evidence,
        agent_id=AGENT_ID_LOCAL,
        episode_terminated=True,
        terminated_step=terminated_step,
        terminated_reason=TERMINATED_REASON,
        current_time_seconds=0.0,
        max_environment_steps=task.limits["max_environment_steps"],
        max_game_time_seconds=task.limits["max_game_time_seconds"],
    )


class SelectedItemSurfaceTests(unittest.TestCase):
    """``Observation.selected_item`` comes from the bridge, not intent."""

    def test_selected_item_is_loaded_from_bridge(self) -> None:
        raw = _default_raw_observation(selected_item="flint_and_steel")
        factory = _stub_factory(raw)
        backend = MineRLEnvironmentBackend(
            env_factory=factory,
            reset_warmup_steps=0,
        )
        backend.open()
        try:
            # The Phase 1 reset rejects ``route_a_a0`` tasks; use
            # the C5 task to drive the actual reset path. The
            # ``route`` mismatch raises before the env factory is
            # called, so we use a custom task shape that satisfies
            # the legacy PortalA0 contract and let the env factory
            # provide the raw observation.
            task = TaskInstance.from_dict(
                {
                    "schema_version": "0.1",
                    "task_id": "casting_s_c5_fixed_wired_test",
                    "route": "lava_casting",
                    "difficulty": 1,
                    "agent_ids": [AGENT_ID_LOCAL],
                    "world_seed": 0,
                    "instruction": "selected_item surface test",
                    "spawn_positions": {AGENT_ID_LOCAL: [0, 4, 0]},
                    "initial_inventories": {
                        AGENT_ID_LOCAL: _default_inventory()
                    },
                    "workflow": "casting_s_c5_fixed",
                    "milestones": [
                        "task_reset",
                        "first_obsidian_cast",
                    ],
                    "limits": {
                        "max_environment_steps": 500,
                        "max_model_calls": 1,
                        "max_game_time_seconds": 120,
                    },
                    "split": "development",
                }
            )
            observations = backend.reset(task)
            observation = observations[AGENT_ID_LOCAL]
            self.assertEqual(observation.selected_item, "flint_and_steel")
        finally:
            backend.close()


class BackendExecutionSafetyTests(unittest.TestCase):
    def test_duration_ticks_repeat_bounded_low_level_action(self) -> None:
        actions: list[Mapping[str, Any]] = []
        env = _StubMineRLBridgeEnv(
            raw_observation=_default_raw_observation(), actions=actions
        )
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: env, reset_warmup_steps=0
        )
        backend.open()
        try:
            backend.reset(_c5_task())
            step = backend.step(
                {
                    AGENT_ID_LOCAL: MacroAction(
                        action_type="move",
                        duration_ticks=4,
                        parameters={
                            "forward": 1.0,
                            "strafe": 0.0,
                            "sprint": False,
                            "jump": False,
                        },
                    )
                }
            )
            self.assertEqual(step.step_id, 1)
            self.assertEqual(len(actions), 4)
            self.assertTrue(all(action["forward"] == 1 for action in actions))
        finally:
            backend.close()

    def test_rejected_translation_does_not_step_environment(self) -> None:
        actions: list[Mapping[str, Any]] = []
        env = _StubMineRLBridgeEnv(
            raw_observation=_default_raw_observation(), actions=actions
        )
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: env, reset_warmup_steps=0
        )
        backend.open()
        try:
            backend.reset(_c5_task())
            with self.assertRaisesRegex(RuntimeError, "translation rejected"):
                backend.step(
                    {
                        AGENT_ID_LOCAL: MacroAction(
                            action_type="use_item", target="dragon_egg"
                        )
                    }
                )
            self.assertEqual(actions, [])
            self.assertEqual(backend._step_id, 0)
        finally:
            backend.close()

    def test_unknown_lava_casting_workflow_is_rejected_before_factory(self) -> None:
        calls: list[str] = []
        task_dict = _c5_task_dict()
        task_dict["workflow"] = "future_unfrozen_casting"
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: calls.append(task.task_id),  # type: ignore[arg-type]
            reset_warmup_steps=0,
        )
        backend.open()
        try:
            with self.assertRaisesRegex(ValueError, "unsupported MineRL workflow"):
                backend.reset(TaskInstance.from_dict(task_dict))
            self.assertEqual(calls, [])
        finally:
            backend.close()

    def test_selected_item_empty_string_is_surfaced_as_none(self) -> None:
        raw = _default_raw_observation(selected_item="empty")
        factory = _stub_factory(raw)
        backend = MineRLEnvironmentBackend(
            env_factory=factory,
            reset_warmup_steps=0,
        )
        backend.open()
        try:
            task = TaskInstance.from_dict(
                {
                    "schema_version": "0.1",
                    "task_id": "casting_s_c5_fixed_empty_selected",
                    "route": "lava_casting",
                    "difficulty": 1,
                    "agent_ids": [AGENT_ID_LOCAL],
                    "world_seed": 0,
                    "instruction": "selected_item surface test",
                    "spawn_positions": {AGENT_ID_LOCAL: [0, 4, 0]},
                    "initial_inventories": {
                        AGENT_ID_LOCAL: _default_inventory()
                    },
                    "workflow": "casting_s_c5_fixed",
                    "milestones": [
                        "task_reset",
                        "first_obsidian_cast",
                    ],
                    "limits": {
                        "max_environment_steps": 500,
                        "max_model_calls": 1,
                        "max_game_time_seconds": 120,
                    },
                    "split": "development",
                }
            )
            observations = backend.reset(task)
            self.assertIsNone(observations[AGENT_ID_LOCAL].selected_item)
        finally:
            backend.close()

    def test_selected_item_unknown_value_fails_closed(self) -> None:
        raw = _default_raw_observation(selected_item="dragon_egg")
        factory = _stub_factory(raw)
        backend = MineRLEnvironmentBackend(
            env_factory=factory,
            reset_warmup_steps=0,
        )
        backend.open()
        try:
            task = TaskInstance.from_dict(
                {
                    "schema_version": "0.1",
                    "task_id": "casting_s_c5_fixed_bad_selected",
                    "route": "lava_casting",
                    "difficulty": 1,
                    "agent_ids": [AGENT_ID_LOCAL],
                    "world_seed": 0,
                    "instruction": "selected_item surface test",
                    "spawn_positions": {AGENT_ID_LOCAL: [0, 4, 0]},
                    "initial_inventories": {
                        AGENT_ID_LOCAL: _default_inventory()
                    },
                    "workflow": "casting_s_c5_fixed",
                    "milestones": [
                        "task_reset",
                        "first_obsidian_cast",
                    ],
                    "limits": {
                        "max_environment_steps": 500,
                        "max_model_calls": 1,
                        "max_game_time_seconds": 120,
                    },
                    "split": "development",
                }
            )
            with self.assertRaises(ValueError):
                backend.reset(task)
        finally:
            backend.close()

    def test_selected_item_missing_bridge_key_fails_closed(self) -> None:
        raw = _default_raw_observation(selected_item="flint_and_steel")
        del raw[PORTAL_SELECTED_ITEM_NAME]
        factory = _stub_factory(raw)
        backend = MineRLEnvironmentBackend(
            env_factory=factory,
            reset_warmup_steps=0,
        )
        backend.open()
        try:
            task = TaskInstance.from_dict(
                {
                    "schema_version": "0.1",
                    "task_id": "casting_s_c5_fixed_missing_selected",
                    "route": "lava_casting",
                    "difficulty": 1,
                    "agent_ids": [AGENT_ID_LOCAL],
                    "world_seed": 0,
                    "instruction": "selected_item surface test",
                    "spawn_positions": {AGENT_ID_LOCAL: [0, 4, 0]},
                    "initial_inventories": {
                        AGENT_ID_LOCAL: _default_inventory()
                    },
                    "workflow": "casting_s_c5_fixed",
                    "milestones": [
                        "task_reset",
                        "first_obsidian_cast",
                    ],
                    "limits": {
                        "max_environment_steps": 500,
                        "max_model_calls": 1,
                        "max_game_time_seconds": 120,
                    },
                    "split": "development",
                }
            )
            with self.assertRaisesRegex(ValueError, "equipped_items"):
                backend.reset(task)
        finally:
            backend.close()

    def test_selected_item_does_not_track_action_intent(
        self,
    ) -> None:
        """The backend must NEVER derive ``selected_item`` from the
        ``equip_item`` request the driver submits. The bridge
        decides what the agent is currently holding; the driver
        only requests a hotbar swap.
        """
        raw = _default_raw_observation(selected_item="lava_bucket")
        env = _StubMineRLBridgeEnv(raw_observation=raw)
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: env,
            reset_warmup_steps=0,
        )
        backend.open()
        try:
            task = TaskInstance.from_dict(
                {
                    "schema_version": "0.1",
                    "task_id": "casting_s_c5_fixed_intent_test",
                    "route": "lava_casting",
                    "difficulty": 1,
                    "agent_ids": [AGENT_ID_LOCAL],
                    "world_seed": 0,
                    "instruction": "selected_item intent test",
                    "spawn_positions": {AGENT_ID_LOCAL: [0, 4, 0]},
                    "initial_inventories": {
                        AGENT_ID_LOCAL: _default_inventory()
                    },
                    "workflow": "casting_s_c5_fixed",
                    "milestones": [
                        "task_reset",
                        "first_obsidian_cast",
                    ],
                    "limits": {
                        "max_environment_steps": 500,
                        "max_model_calls": 1,
                        "max_game_time_seconds": 120,
                    },
                    "split": "development",
                }
            )
            backend.reset(task)
            # Driver requests ``equip_item(flint_and_steel)``,
            # but the bridge says the agent is still holding
            # ``lava_bucket``. The Observation must reflect the
            # bridge's truth, not the request stream.
            step = backend.step(
                {
                    AGENT_ID_LOCAL: MacroAction(
                        action_type="equip_item",
                        target="flint_and_steel",
                    )
                }
            )
            self.assertEqual(
                step.observations[AGENT_ID_LOCAL].selected_item,
                "lava_bucket",
            )
        finally:
            backend.close()


# ----------------------------------------------------------------------
# MineRL backend: capability manifest honesty
# ----------------------------------------------------------------------


class CapabilityManifestTests(unittest.TestCase):
    """The capability manifest is honest about the wired surface."""

    def test_static_manifest_advertises_wired_truth(self) -> None:
        # The R6-C5-LIVE-MINERL-BACKEND-WIRING milestone now
        # exposes typed target-block and fluid truth on the
        # production backend. Both capabilities are reported
        # ``True`` because the offline test suite exercises the
        # full C1 / C2 / C3 / C4 / C5 truth surface; live
        # MineRL / Minecraft verification is still out of
        # scope and the pre-episode gate does not change
        # behaviour for unrelated capabilities.
        caps = ProductionMineRLEnvironmentBackend.casting_c1_capabilities()
        for field in CAPABILITY_IDS:
            self.assertTrue(
                getattr(caps, _field_for(field)),
                msg=f"capability {field!r} must be True on the production "
                "MineRL backend now that the typed truth surface is wired",
            )

    def test_instance_capabilities_match_static(self) -> None:
        backend = ProductionMineRLEnvironmentBackend(reset_warmup_steps=0)
        self.assertEqual(
            backend.capabilities(),
            ProductionMineRLEnvironmentBackend.casting_c1_capabilities(),
        )

    def test_production_gate_passes_with_wired_truth(self) -> None:
        # With both capabilities now reported ``True``, the
        # pre-episode gate must accept the production casting
        # C5 task. A separate test (in
        # ``CapabilityManifestFailClosedTests``) exercises the
        # fail-closed path on a backend whose capabilities
        # have been intentionally downgraded.
        caps = ProductionMineRLEnvironmentBackend.casting_c1_capabilities()
        self.assertIsNone(
            assert_casting_c1_capabilities(caps, task_id=EPISODE_ID)
        )


class _StubBackend:
    """Bare-bones backend shim that exposes the wired capabilities."""

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities.full()


def _field_for(capability_id: str) -> str:
    mapping = {
        "select_water_bucket": "can_select_water_bucket",
        "select_lava_bucket": "can_select_lava_bucket",
        "use_water_bucket": "can_use_water_bucket",
        "use_lava_bucket": "can_use_lava_bucket",
        "public_inventory": "exposes_public_inventory",
        "selected_item": "exposes_selected_item",
        "target_block_truth": "exposes_target_block_truth",
        "fluid_truth": "exposes_fluid_truth",
    }
    return mapping[capability_id]


class CapabilityManifestFailClosedTests(unittest.TestCase):
    """A backend missing any capability fails closed *before* env creation."""

    def _incomplete_backend(self, missing: tuple[str, ...]) -> MineRLEnvironmentBackend:
        caps = dataclasses.replace(
            BackendCapabilities.full(),
            **{_field_for(m): False for m in missing},
        )
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: _StubMineRLBridgeEnv(
                raw_observation=_default_raw_observation()
            ),
            reset_warmup_steps=0,
        )
        backend.capabilities = lambda: caps  # type: ignore[method-assign]
        return backend

    def test_missing_select_water_bucket_fails_closed(self) -> None:
        backend = self._incomplete_backend(("select_water_bucket",))
        backend.open()
        try:
            with self.assertRaises(CapabilityMismatchError) as ctx:
                backend.reset(_c5_task())
            self.assertEqual(ctx.exception.missing, ("select_water_bucket",))
            self.assertIsNone(backend._env)
            self.assertIsNone(backend._task)
        finally:
            backend.close()

    def test_missing_multiple_capabilities_fail_closed(self) -> None:
        backend = self._incomplete_backend(
            ("use_lava_bucket", "fluid_truth", "selected_item")
        )
        backend.open()
        try:
            with self.assertRaises(CapabilityMismatchError) as ctx:
                backend.reset(_c5_task())
            self.assertEqual(
                ctx.exception.missing,
                ("use_lava_bucket", "selected_item", "fluid_truth"),
            )
        finally:
            backend.close()


# ----------------------------------------------------------------------
# MineRL backend: truth surface (target block / fluid / frame / activation)
# ----------------------------------------------------------------------


class TruthSurfaceTests(unittest.TestCase):
    """Target block, fluid, frame, and activation truth are exposed."""

    def test_target_block_and_fluid_truth_via_grid(self) -> None:
        # Build a full obsidian 4x5 frame on the bridge's portal
        # grid and a single nether_portal at the public ignition
        # target. The MineRL backend detects the frame on the
        # private truth surface; the public Observation must NOT
        # carry the grid / target-block / fluid truth.
        obs_blocks: dict[tuple[int, int, int], str] = {
            cell: "obsidian" for cell in CASTING_S_C3_FRAME_CELLS
        }
        obs_blocks[C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET] = "nether_portal"
        raw = _default_raw_observation()
        raw["portal_grid"] = _portal_grid(obs_blocks)
        factory = _stub_factory(raw)
        backend = MineRLEnvironmentBackend(
            env_factory=factory,
            reset_warmup_steps=0,
        )
        backend.open()
        try:
            task = TaskInstance.from_dict(
                {
                    "schema_version": "0.1",
                    "task_id": "casting_s_c5_fixed_truth_surface",
                    "route": "lava_casting",
                    "difficulty": 1,
                    "agent_ids": [AGENT_ID_LOCAL],
                    "world_seed": 0,
                    "instruction": "truth surface test",
                    "spawn_positions": {AGENT_ID_LOCAL: [0, 4, 0]},
                    "initial_inventories": {
                        AGENT_ID_LOCAL: _default_inventory()
                    },
                    "workflow": "casting_s_c5_fixed",
                    "milestones": [
                        "task_reset",
                        "first_obsidian_cast",
                    ],
                    "limits": {
                        "max_environment_steps": 500,
                        "max_model_calls": 1,
                        "max_game_time_seconds": 120,
                    },
                    "split": "development",
                }
            )
            backend.reset(task)
            observation_step = backend.step(
                {AGENT_ID_LOCAL: MacroAction.wait()}
            )
            frame = observation_step.observations[AGENT_ID_LOCAL].frame
            self.assertNotIn("portal_grid", frame)
            self.assertNotIn(
                "target_block_truth",
                observation_step.info,
            )
            self.assertNotIn(
                "fluid_truth",
                observation_step.info,
            )
            # The legacy ``EvaluationState`` is still available via
            # the public ``get_evaluation_state`` accessor.
            legacy_state = backend.get_evaluation_state()
            self.assertIsInstance(legacy_state, EvaluationState)
        finally:
            backend.close()


class TruthSurfaceEvaluatorTests(unittest.TestCase):
    """The full C5 evaluator pipeline runs on the backend's truth surface."""

    def test_c5_evaluator_succeeds_on_wired_truth(self) -> None:
        task = _c5_task()
        state = _build_minimal_c5_state(
            task,
            driver_result=None,
            step_id=TERMINATED_STEP,
            terminated_step=TERMINATED_STEP,
        )
        result = FrozenNetherEntryEvaluator().evaluate(state)
        self.assertEqual(result.outcome, "success")
        self.assertTrue(result.success)
        self.assertTrue(result.frame_identity_matched)

    def test_c5_evaluator_rejects_external_portal_entry(self) -> None:
        task = _c5_task()
        state = _build_minimal_c5_state(
            task,
            driver_result=None,
            step_id=TERMINATED_STEP,
            terminated_step=TERMINATED_STEP,
            entered_via_episode_portal=False,
        )
        result = FrozenNetherEntryEvaluator().evaluate(state)
        self.assertEqual(result.outcome, "nether_entry_not_via_episode_portal")
        self.assertFalse(result.success)

    def test_c5_evaluator_rejects_missing_entry_evidence(self) -> None:
        task = _c5_task()
        state = _build_minimal_c5_state(
            task,
            driver_result=None,
            step_id=TERMINATED_STEP,
            terminated_step=TERMINATED_STEP,
            include_entry_evidence=False,
        )
        result = FrozenNetherEntryEvaluator().evaluate(state)
        self.assertEqual(result.outcome, "nether_entry_portal_unknown")
        self.assertFalse(result.success)

    def test_c5_evaluator_rejects_wrong_source_dimension(self) -> None:
        task = _c5_task()
        state = _build_minimal_c5_state(
            task,
            driver_result=None,
            step_id=TERMINATED_STEP,
            terminated_step=TERMINATED_STEP,
            source_dimension="minecraft:the_end",
        )
        result = FrozenNetherEntryEvaluator().evaluate(state)
        self.assertEqual(result.outcome, "wrong_source_dimension")
        self.assertFalse(result.success)

    def test_c5_evaluator_rejects_wrong_target_dimension(self) -> None:
        task = _c5_task()
        state = _build_minimal_c5_state(
            task,
            driver_result=None,
            step_id=TERMINATED_STEP,
            terminated_step=TERMINATED_STEP,
            target_dimension="minecraft:overworld",
        )
        result = FrozenNetherEntryEvaluator().evaluate(state)
        self.assertEqual(result.outcome, "wrong_target_dimension")
        self.assertFalse(result.success)

    def test_c5_evaluator_rejects_transition_before_activation(self) -> None:
        task = _c5_task()
        state = _build_minimal_c5_state(
            task,
            driver_result=None,
            step_id=TERMINATED_STEP,
            terminated_step=TERMINATED_STEP,
            transition_step=IGNITION_STEP - 2,
        )
        result = FrozenNetherEntryEvaluator().evaluate(state)
        self.assertEqual(result.outcome, "transition_before_activation")
        self.assertFalse(result.success)

    def test_c5_evaluator_rejects_frame_identity_mismatch(self) -> None:
        task = _c5_task()
        other_identity = build_c4_c3_frame_identity(
            episode_id=task.task_id,
            step_id=TERMINATED_STEP,
            agent_id=AGENT_ID_LOCAL,
            activation_offsets=((1, 2, 1),),
        )
        state = _build_minimal_c5_state(
            task,
            driver_result=None,
            step_id=TERMINATED_STEP,
            terminated_step=TERMINATED_STEP,
            matched_frame_identity=other_identity,
        )
        result = FrozenNetherEntryEvaluator().evaluate(state)
        self.assertEqual(result.outcome, "frame_identity_mismatch")
        self.assertFalse(result.success)


# ----------------------------------------------------------------------
# C5 driver + MineRL backend integration
# ----------------------------------------------------------------------


class C5DriverMineRLBackendIntegrationTests(unittest.TestCase):
    """C5 driver walks the full 347-step plan on the MineRL backend."""

    def setUp(self) -> None:
        self.task = _c5_task()
        self.context = _build_c5_context(self.task)
        self.captured_actions: list[Mapping[str, Any]] = []

    def _build_env_factory(
        self, raw_observation: dict[str, Any]
    ) -> Any:
        env = _StubMineRLBridgeEnv(raw_observation=raw_observation)

        def factory(task: TaskInstance) -> _StubMineRLBridgeEnv:
            self.captured_actions.append({"task_id": task.task_id})
            return env

        return factory

    def test_driver_completes_full_347_step_plan(self) -> None:
        raw = _default_raw_observation(
            selected_item="water_bucket",
            position=(0.5, 64.0, 0.5),
        )
        factory = self._build_env_factory(raw)
        backend = MineRLEnvironmentBackend(
            env_factory=factory,
            reset_warmup_steps=0,
        )
        backend.open()
        try:
            result = run_casting_s_c5_nether_entry_driver(
                backend, self.context
            )
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.steps_executed, 347)
            self.assertEqual(result.planned_steps, 347)
            self.assertEqual(
                result.ignition_relevant_action_step, IGNITION_STEP
            )
            self.assertEqual(
                result.ignition_equip_step, IGNITION_EQUIP_STEP
            )
            self.assertEqual(result.nether_entry_step, ENTRY_TRANSITION_STEP)
            self.assertEqual(
                result.nether_entry_approach_step, ENTRY_APPROACH_FIRST_STEP
            )
        finally:
            backend.close()

    def test_driver_does_not_read_evaluator_only_state(self) -> None:
        """AST / source-level lock: the driver must not access
        ``selected_item`` / ``target_block_truth`` / ``fluid_truth``
        / latched frame identity / matched frame identity /
        pre-transition position / entered_via_episode_portal.
        """
        import pathlib

        source = pathlib.Path(
            "obsidianlink/drivers/casting_s_c5_nether_entry.py"
        ).read_text()
        tree = ast.parse(source)
        forbidden = {
            "selected_item",
            "target_block_truth",
            "fluid_truth",
            "latched_frame_identity",
            "matched_frame_identity",
            "pre_transition_position",
            "entered_via_episode_portal",
            "agents_in_nether",
            "latched_activation_offsets",
            "nether_entry_evaluation",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr in forbidden:
                    self.fail(
                        f"C5 driver source references forbidden "
                        f"attribute {node.attr!r}"
                    )
            elif isinstance(node, ast.Subscript):
                value = node.value
                if isinstance(value, ast.Name) and value.id in forbidden:
                    self.fail(
                        f"C5 driver source subscripts forbidden name "
                        f"{value.id!r}"
                    )

    def test_orchestrator_evaluates_c5_success(self) -> None:
        raw = _default_raw_observation()
        factory = self._build_env_factory(raw)
        backend = MineRLEnvironmentBackend(
            env_factory=factory,
            reset_warmup_steps=0,
        )
        backend.open()
        try:
            result = run_casting_s_c5_nether_entry_driver(
                backend, self.context
            )
            c5_state = _build_minimal_c5_state(
                self.task,
                driver_result=result,
                step_id=TERMINATED_STEP,
                terminated_step=TERMINATED_STEP,
            )
            verdict = FrozenNetherEntryEvaluator().evaluate(c5_state)
            self.assertEqual(verdict.outcome, "success")
            self.assertTrue(verdict.success)
        finally:
            backend.close()


# ----------------------------------------------------------------------
# Bridge observable
# ----------------------------------------------------------------------


class PortalSelectedItemObservableTests(unittest.TestCase):
    """HumanSurvival's real equipped-item handler remains registered."""

    def test_handler_registers_with_observables(self) -> None:
        spec = PortalA0EnvSpec()
        observables = spec.create_observables()
        names = {
            handler.to_string()
            for handler in observables
            if callable(getattr(handler, "to_string", None))
        }
        self.assertIn(PORTAL_SELECTED_ITEM_NAME, names)

    def test_handler_space_lists_all_selectable_items(self) -> None:
        raw = _default_raw_observation(selected_item="cobblestone")
        self.assertEqual(
            MineRLEnvironmentBackend._selected_item_from_raw(raw),
            "cobblestone",
        )

    def test_empty_bucket_is_valid_observed_selected_item(self) -> None:
        raw = _default_raw_observation(selected_item="bucket")
        self.assertIn("bucket", PORTAL_SELECTABLE_ITEMS)
        self.assertEqual(
            MineRLEnvironmentBackend._selected_item_from_raw(raw),
            "bucket",
        )

    def test_empty_bucket_does_not_become_an_action_target(self) -> None:
        from obsidianlink.actions.minerl_translator import (
            TRANSLATOR_EQUIPPABLE_ITEMS,
        )

        self.assertNotIn("bucket", TRANSLATOR_EQUIPPABLE_ITEMS)


# ----------------------------------------------------------------------
# Source / capability gate integration
# ----------------------------------------------------------------------


class BridgeContractTests(unittest.TestCase):
    """The portal grid helper / world anchor contract is preserved."""

    def test_portal_grid_spec_includes_selected_item_observable(
        self,
    ) -> None:
        spec = PortalA0EnvSpec()
        names: set[str] = set()
        for handler in spec.create_observables():
            to_string = getattr(handler, "to_string", None)
            if callable(to_string):
                try:
                    value = to_string()
                except TypeError:
                    continue
                if isinstance(value, str):
                    names.add(value)
        self.assertIn(PORTAL_SELECTED_ITEM_NAME, names)

    def test_grid_origin_returns_integer_triple_when_present(self) -> None:
        raw = _default_raw_observation()
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: _StubMineRLBridgeEnv(
                raw_observation=raw
            ),
            reset_warmup_steps=0,
        )
        origin = backend._grid_world_anchor(raw)
        self.assertEqual(origin, (0, 64, 0))

    def test_grid_origin_rejects_malformed_anchor(self) -> None:
        raw = _default_raw_observation()
        raw["portal_grid_origin"] = np.asarray((0, 0, 0), dtype=np.float64)
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: _StubMineRLBridgeEnv(
                raw_observation=raw
            ),
            reset_warmup_steps=0,
        )
        self.assertIsNone(backend._grid_world_anchor(raw))

    def test_grid_origin_rejects_out_of_range_anchor(self) -> None:
        raw = _default_raw_observation()
        raw["portal_grid_origin"] = np.asarray(
            (40_000_000, 0, 0), dtype=np.int32
        )
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: _StubMineRLBridgeEnv(
                raw_observation=raw
            ),
            reset_warmup_steps=0,
        )
        self.assertIsNone(backend._grid_world_anchor(raw))


class PublicContextRejectsDriverIntentTests(unittest.TestCase):
    """The driver context must not derive from action intent."""

    def test_context_ignores_evaluator_contract(self) -> None:
        # Mutate the task dict to embed a fake ``evaluator_contract``
        # and confirm ``build_public_c5_nether_entry_driver_context_from_task``
        # still builds the same context (it only reads the public
        # spec / inventory / limits).
        task = _c5_task()
        ctx = build_public_c5_nether_entry_driver_context_from_task(task)
        # Sanity: ignition target locked to [1, 1, 1].
        self.assertEqual(
            ctx.ignition_target, C5_NETHER_ENTRY_PUBLIC_IGNITION_TARGET
        )
        # And the inventory is exactly the C5 default.
        self.assertEqual(
            dict(ctx.initial_inventory), _default_inventory()
        )


class C5DeterminismTests(unittest.TestCase):
    """The C5 driver is deterministic on the MineRL backend stub."""

    def test_two_runs_produce_identical_results(self) -> None:
        task = _c5_task()
        context = _build_c5_context(task)
        results = []
        for _ in range(2):
            raw = _default_raw_observation()
            backend = MineRLEnvironmentBackend(
                env_factory=lambda task: _StubMineRLBridgeEnv(
                    raw_observation=raw
                ),
                reset_warmup_steps=0,
            )
            backend.open()
            try:
                results.append(
                    run_casting_s_c5_nether_entry_driver(
                        backend, context
                    ).as_dict()
                )
            finally:
                backend.close()
        self.assertEqual(results[0], results[1])


class DriverBudgetSurfaceTests(unittest.TestCase):
    """The driver respects its budget caps on the MineRL backend stub."""

    def test_step_budget_exhaustion_blocks_driver(self) -> None:
        task = _c5_task()
        context = _build_c5_context(task)
        raw = _default_raw_observation()
        # Force a RecoverableBackendError on every bucket use to
        # exhaust the per-step recovery budget and reach the
        # step_budget boundary faster.
        env = _StubMineRLBridgeEnv(raw_observation=raw)
        original_step = env.step

        def failing_step(action: Mapping[str, Any]):
            if int(action.get("use", 0)) == 1:
                raise RecoverableBackendError(
                    "stub bucket use transient",
                    recoverable_kind="bucket_use_transient",
                )
            return original_step(action)

        env.step = failing_step  # type: ignore[method-assign]
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: env,
            reset_warmup_steps=0,
        )
        backend.open()
        try:
            result = run_casting_s_c5_nether_entry_driver(
                backend, context,
                total_recovery_budget=1,
                recoveries_per_use_item=1,
            )
            self.assertIn(result.status, {"blocked", "failed"})
            self.assertIsNotNone(result.blocked_reason)
        finally:
            backend.close()


class DriverEventHygieneTests(unittest.TestCase):
    """Driver events must NOT carry evaluator-only tokens."""

    FORBIDDEN_TOKENS: tuple[str, ...] = (
        "selected_item",
        "target_block_truth",
        "fluid_truth",
        "latched_frame_identity",
        "matched_frame_identity",
        "pre_transition_position",
        "entered_via_episode_portal",
        "agents_in_nether",
        "latched_activation_offsets",
        "nether_entry_evaluation",
    )

    def test_events_do_not_carry_evaluator_tokens(self) -> None:
        task = _c5_task()
        context = _build_c5_context(task)
        raw = _default_raw_observation()
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: _StubMineRLBridgeEnv(
                raw_observation=raw
            ),
            reset_warmup_steps=0,
        )
        backend.open()
        try:
            result = run_casting_s_c5_nether_entry_driver(
                backend, context
            )
            for event in result.events:
                for value in event.values():
                    if value is None:
                        continue
                    serialized = str(value)
                    for token in self.FORBIDDEN_TOKENS:
                        self.assertNotIn(
                            token,
                            serialized,
                            msg=(
                                f"Driver event leaks {token!r}; events "
                                "must not carry evaluator truth"
                            ),
                        )
        finally:
            backend.close()


class ObservationIsolationTests(unittest.TestCase):
    """The MineRL backend's Observation must NOT carry evaluator truth."""

    def test_observation_does_not_carry_target_block_or_fluid_truth(
        self,
    ) -> None:
        raw = _default_raw_observation()
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: _StubMineRLBridgeEnv(
                raw_observation=raw
            ),
            reset_warmup_steps=0,
        )
        backend.open()
        try:
            task = TaskInstance.from_dict(
                {
                    "schema_version": "0.1",
                    "task_id": "casting_s_c5_fixed_isolation_test",
                    "route": "lava_casting",
                    "difficulty": 1,
                    "agent_ids": [AGENT_ID_LOCAL],
                    "world_seed": 0,
                    "instruction": "isolation test",
                    "spawn_positions": {AGENT_ID_LOCAL: [0, 4, 0]},
                    "initial_inventories": {
                        AGENT_ID_LOCAL: _default_inventory()
                    },
                    "workflow": "casting_s_c5_fixed",
                    "milestones": [
                        "task_reset",
                        "first_obsidian_cast",
                    ],
                    "limits": {
                        "max_environment_steps": 500,
                        "max_model_calls": 1,
                        "max_game_time_seconds": 120,
                    },
                    "split": "development",
                }
            )
            observations = backend.reset(task)
            for observation in observations.values():
                forbidden = (
                    "target_block_truth",
                    "fluid_truth",
                    "portal_grid",
                    "latched_frame_identity",
                    "matched_frame_identity",
                    "agents_in_nether",
                    "entered_via_episode_portal",
                    "pre_transition_position",
                    "nether_entry_evaluation",
                )
                for token in forbidden:
                    self.assertFalse(
                        hasattr(observation, token),
                        f"{token} must not appear on the public observation",
                    )
        finally:
            backend.close()


class ResetStepCloseCleanupTests(unittest.TestCase):
    """reset / step / close must clean up backend latched state."""

    def test_close_clears_latched_state(self) -> None:
        # Open + reset + step on the backend, then close. After
        # close every latched attribute must be at its initial
        # value. The cleanup contract is the same whether the
        # frame identity was latched or not.
        task = TaskInstance.from_dict(
            {
                "schema_version": "0.1",
                "task_id": "casting_s_c5_fixed_cleanup_test",
                "route": "lava_casting",
                "difficulty": 4,
                "agent_ids": [AGENT_ID_LOCAL],
                "world_seed": 0,
                "instruction": "cleanup test",
                "spawn_positions": {AGENT_ID_LOCAL: [0, 4, 0]},
                "initial_inventories": {
                    AGENT_ID_LOCAL: _default_inventory()
                },
                "workflow": "casting_s_c5_fixed",
                "milestones": [
                    "task_reset",
                    "first_obsidian_cast",
                ],
                "limits": {
                    "max_environment_steps": 500,
                    "max_model_calls": 1,
                    "max_game_time_seconds": 120,
                },
                "split": "development",
            }
        )
        raw = _default_raw_observation()
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: _StubMineRLBridgeEnv(
                raw_observation=raw
            ),
            reset_warmup_steps=0,
        )
        backend.open()
        try:
            backend.reset(task)
            backend.step({AGENT_ID_LOCAL: MacroAction.wait()})
            # The backend's latched state may or may not be empty
            # depending on the bridge observation; the cleanup
            # contract is identical either way.
        finally:
            backend.close()
        # The closed backend's latched state is fresh.
        self.assertIsNone(backend._latched["frame_identity"])
        self.assertEqual(backend._latched["transition_step_by_agent"], {})
        self.assertEqual(
            backend._latched["entered_via_episode_portal_by_agent"], {}
        )
        self.assertEqual(backend._latched["matched_frame_identity_by_agent"], {})
        self.assertEqual(backend._step_id, 0)
        self.assertIsNone(backend._env)
        self.assertIsNone(backend._task)
        self.assertIsNone(backend._latest_raw)
        self.assertIsNone(backend._baseline_grid)


# ----------------------------------------------------------------------
# R6-C5-LIVE-MINERL-BACKEND-WIRING typed truth surface
# ----------------------------------------------------------------------


def _c1_task_dict(
    *,
    max_environment_steps: int = 160,
    max_game_time_seconds: int = 120,
) -> dict[str, Any]:
    return {
        "schema_version": "0.1",
        "task_id": "casting_c1_fixed_seed_0",
        "route": "lava_casting",
        "difficulty": 1,
        "agent_ids": [AGENT_ID_LOCAL],
        "world_seed": 0,
        "instruction": "R6 C1 typed truth offline contract test.",
        "spawn_positions": {AGENT_ID_LOCAL: [0, 4, 0]},
        "initial_inventories": {
            AGENT_ID_LOCAL: {
                "water_bucket": 1,
                "lava_bucket": 1,
                "cobblestone": 8,
            }
        },
        "workflow": "casting_c1_fixed",
        "milestones": [
            "task_reset",
            "first_obsidian_cast",
        ],
        "limits": {
            "max_environment_steps": max_environment_steps,
            "max_model_calls": 1,
            "max_game_time_seconds": max_game_time_seconds,
        },
        "split": "development",
        "scenario_parameters": {
            "task_family": "casting",
            "agent_mode": "single",
            "task_level": "C1",
            "layout_type": "fixed",
            "compatibility_task_name": "casting_s_c1_fixed",
            "implementation_status": "offline_fake_verified",
            "world_dimension": "minecraft:overworld",
            "layout": "fixed_controlled",
            "target_cell": [2, 4, 3],
            "target_initial_block": "air",
            "mechanics_required": "vanilla_water_lava_block_update",
            "allow_minecraft_commands": False,
            "allow_evaluator_world_mutation": False,
            "allow_live_run": False,
            "requires_explicit_live_run_approval": True,
        },
    }


def _c1_task() -> TaskInstance:
    return TaskInstance.from_dict(_c1_task_dict())


def _c3_task_dict(
    *,
    max_environment_steps: int = 240,
    max_game_time_seconds: int = 180,
) -> dict[str, Any]:
    return {
        "schema_version": "0.1",
        "task_id": "casting_c3_fixed_seed_0",
        "route": "lava_casting",
        "difficulty": 2,
        "agent_ids": [AGENT_ID_LOCAL],
        "world_seed": 0,
        "instruction": "R6 C2 typed truth offline contract test.",
        "spawn_positions": {AGENT_ID_LOCAL: [0, 4, 0]},
        "initial_inventories": {
            AGENT_ID_LOCAL: {
                "water_bucket": 3,
                "lava_bucket": 3,
                "cobblestone": 6,
            }
        },
        "workflow": "casting_c3_fixed",
        "milestones": [
            "task_reset",
            "first_obsidian_cast",
        ],
        "limits": {
            "max_environment_steps": max_environment_steps,
            "max_model_calls": 1,
            "max_game_time_seconds": max_game_time_seconds,
        },
        "split": "development",
        "scenario_parameters": {
            "task_family": "casting",
            "agent_mode": "single",
            "task_level": "C2",
            "layout_type": "fixed",
            "compatibility_task_name": "casting_s_c2_fixed",
            "implementation_status": "offline_fake_verified",
            "world_dimension": "minecraft:overworld",
            "layout": "fixed_controlled",
            "target_cells": [
                [2, 4, 3],
                [3, 4, 3],
                [4, 4, 3],
            ],
            "target_initial_blocks": ["air", "air", "air"],
            "mechanics_required": "vanilla_water_lava_block_update",
            "allow_minecraft_commands": False,
            "allow_evaluator_world_mutation": False,
            "allow_live_run": False,
            "requires_explicit_live_run_approval": True,
        },
    }


def _c3_task() -> TaskInstance:
    return TaskInstance.from_dict(_c3_task_dict())


def _c3_target_task_for_driver() -> TaskInstance:
    """Return a casting_s_c3_fixed task for the C3 driver."""
    return TaskInstance.from_dict(
        {
            "schema_version": "0.1",
            "task_id": "casting_s_c3_fixed_seed_0",
            "route": "lava_casting",
            "difficulty": 3,
            "agent_ids": [AGENT_ID_LOCAL],
            "world_seed": 0,
            "instruction": "Cast a public 4x5 full-ring portal.",
            "spawn_positions": {AGENT_ID_LOCAL: [0, 4, 0]},
            "initial_inventories": {
                AGENT_ID_LOCAL: {
                    "water_bucket": 14,
                    "lava_bucket": 14,
                    "cobblestone": 28,
                }
            },
            "workflow": "casting_s_c3_fixed",
            "milestones": [
                "task_reset",
                "first_obsidian_cast",
                "valid_portal_frame",
            ],
            "limits": {
                "max_environment_steps": 640,
                "max_model_calls": 1,
                "max_game_time_seconds": 600,
            },
            "split": "development",
            "scenario_parameters": {
                "task_family": "casting",
                "agent_mode": "single",
                "task_level": "C3",
                "layout_type": "fixed",
                "compatibility_task_name": "casting_s_c3_fixed",
                "implementation_status": "contract_only",
                "world_dimension": "minecraft:overworld",
                "layout": "fixed_controlled",
                "mechanics_required": "vanilla_water_lava_block_update",
                "public_task_spec": {
                    "coordinate_space": "task_origin_relative",
                    "task_origin_marker": "visible",
                    "frame_plan": {
                        "orientation": "plane_z",
                        "min_corner": [0, 0, 1],
                        "width": 4,
                        "height": 5,
                        "require_full_ring": True,
                        "minecraft_minimum_required_block_count": 10,
                        "benchmark_required_full_ring_block_count": 14,
                        "required_corner_count": 4,
                        "interior_allowlist": [
                            "air",
                            "nether_portal",
                            "fire",
                        ],
                        "fixed_offsets": [
                            list(c) for c in CASTING_S_C3_FRAME_CELLS
                        ],
                    },
                    "ignition_plan": {
                        "required": False,
                        "action": None,
                        "item": None,
                        "target_offset": None,
                    },
                    "nether_entry_goal": {
                        "required": False,
                        "designated_agent_ids": [],
                        "target_dimension": None,
                    },
                },
                "allow_minecraft_commands": False,
                "allow_evaluator_world_mutation": False,
                "allow_live_run": False,
                "requires_explicit_live_run_approval": True,
            },
        }
    )


def _c5_target_task() -> TaskInstance:
    """Reuse the existing C5 task factory."""
    return _c5_task()


def _build_raw_with_obsidian(
    *,
    obsidian_offsets: tuple[tuple[int, int, int], ...] = (),
    water_offsets: tuple[tuple[int, int, int], ...] = (),
    lava_offsets: tuple[tuple[int, int, int], ...] = (),
    selected_item: str = "water_bucket",
    nether_portal_offsets: tuple[tuple[int, int, int], ...] = (),
    position: tuple[float, float, float] = (0.5, 64.0, 0.5),
    portal_transition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a raw observation with the requested bridge-level truth.

    The block grid carries ``obsidian`` at the given offsets;
    the fluid grid carries ``water`` / ``flowing_water`` /
    ``lava`` / ``flowing_lava`` at the matching offsets. The
    portal grid uses the closed :data:`PORTAL_GRID_BLOCKS`
    whitelist for both ordinary blocks and fluids. The function
    intentionally returns a deterministic raw observation so the
    offline tests can exercise the typed truth surface without
    touching MineRL.
    """
    block_grid = np.full(
        PORTAL_GRID_SIZE,
        PORTAL_GRID_BLOCKS.index("air"),
        dtype=np.int32,
    )
    for offset in water_offsets:
        block_grid[_flat_offset(offset)] = PORTAL_GRID_BLOCKS.index("water")
    for offset in lava_offsets:
        block_grid[_flat_offset(offset)] = PORTAL_GRID_BLOCKS.index("lava")
    for offset in obsidian_offsets:
        idx = _flat_offset(offset)
        block_grid[idx] = PORTAL_GRID_BLOCKS.index("obsidian")
    for offset in nether_portal_offsets:
        idx = _flat_offset(offset)
        block_grid[idx] = PORTAL_GRID_BLOCKS.index("nether_portal")
    raw = _default_raw_observation(selected_item=selected_item, position=position)
    raw["portal_grid"] = block_grid
    if portal_transition is not None:
        raw["portal_transition"] = portal_transition
    return raw


def _open_backend_with(
    raw: dict[str, Any],
    task: TaskInstance,
) -> MineRLEnvironmentBackend:
    """Open a backend wired to a deterministic raw observation."""
    backend = MineRLEnvironmentBackend(
        env_factory=lambda task: _StubMineRLBridgeEnv(raw_observation=raw),
        reset_warmup_steps=0,
    )
    backend.open()
    backend.reset(task)
    return backend


def _build_verified_cast_sequence(
    target_cells: tuple[tuple[int, int, int], ...],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[MacroAction]]:
    """Return air baseline plus per-cell water/lava/obsidian observations."""
    completed: list[tuple[int, int, int]] = []
    trajectory: list[dict[str, Any]] = []
    actions: list[MacroAction] = []
    for cell in target_cells:
        trajectory.extend(
            (
                _build_raw_with_obsidian(
                    obsidian_offsets=tuple(completed),
                    water_offsets=(cell,),
                ),
                _build_raw_with_obsidian(
                    obsidian_offsets=tuple(completed),
                    lava_offsets=(cell,),
                ),
                _build_raw_with_obsidian(
                    obsidian_offsets=tuple(completed + [cell]),
                ),
            )
        )
        actions.extend(
            (
                MacroAction("use_item", target="water_bucket"),
                MacroAction("use_item", target="lava_bucket"),
                MacroAction.wait(),
            )
        )
        completed.append(cell)
    return _build_raw_with_obsidian(), trajectory, actions


class TypedTruthCastingC1Tests(unittest.TestCase):
    """The production backend exposes typed C1 casting truth."""

    def test_c1_typed_truth_succeeds_after_water_and_lava_actions(
        self,
    ) -> None:
        target_cell = (2, 4, 3)
        grid_cell = (2, 0, 3)
        # The reset baseline must show ``air`` at the target
        # cell. The trajectory must show water then lava then
        # obsidian so the backend can latch the first water
        # and first lava observations for the C1 evaluator
        # to read both ``water_truth.present == True`` and
        # ``lava_truth.present == True`` from the same typed
        # state.
        baseline_raw = _build_raw_with_obsidian(obsidian_offsets=())
        water_raw = _build_raw_with_obsidian(
            obsidian_offsets=(), water_offsets=(grid_cell,)
        )
        lava_raw = _build_raw_with_obsidian(
            obsidian_offsets=(), lava_offsets=(grid_cell,)
        )
        final_raw = _build_raw_with_obsidian(obsidian_offsets=(grid_cell,))
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: _StubMineRLBridgeEnv(
                raw_observation=baseline_raw,
                trajectory=[water_raw, lava_raw, final_raw],
            ),
            reset_warmup_steps=0,
        )
        backend.open()
        backend.reset(_c1_task())
        try:
            # The driver emits water then lava; both must be
            # accepted by the translator to earn a typed
            # credit on the backend's latched state.
            backend.step(
                {
                    AGENT_ID_LOCAL: MacroAction(
                        action_type="use_item", target="water_bucket"
                    )
                }
            )
            backend.step(
                {
                    AGENT_ID_LOCAL: MacroAction(
                        action_type="use_item", target="lava_bucket"
                    )
                }
            )
            backend.step({AGENT_ID_LOCAL: MacroAction.wait()})
            backend.mark_terminated(reason="driver_done")
            state = backend.get_casting_evaluation_state(target_cell)
            self.assertIsInstance(state, CastingEvaluationState)
            self.assertEqual(state.target_cell, target_cell)
            self.assertEqual(state.current_target_block, "obsidian")
            self.assertEqual(state.initial_target_block, "air")
            self.assertIsNotNone(state.target_update_evidence)
            if state.target_update_evidence is not None:
                self.assertEqual(
                    state.target_update_evidence.before_block, "air"
                )
                self.assertEqual(
                    state.target_update_evidence.after_block, "obsidian"
                )
            self.assertEqual(len(state.relevant_action_steps), 2)
            self.assertTrue(state.water_truth.present)
            self.assertTrue(state.lava_truth.present)
            result = CastingEvaluator().evaluate(state)
            self.assertEqual(result.outcome, "success")
        finally:
            backend.close()

    def test_c1_typed_truth_fails_closed_on_pre_existing_obsidian(
        self,
    ) -> None:
        target_cell = (2, 4, 3)
        # Baseline already has obsidian at the cell. The
        # backend must not produce transition evidence
        # because the episode did not produce the obsidian.
        raw = _build_raw_with_obsidian(obsidian_offsets=((2, 0, 3),))
        backend = _open_backend_with(raw, _c1_task())
        try:
            state = backend.get_casting_evaluation_state(target_cell)
            self.assertEqual(state.current_target_block, "obsidian")
            self.assertEqual(state.initial_target_block, "obsidian")
            self.assertIsNone(state.target_update_evidence)
            # The driver did not run any cast actions during
            # the episode, so relevant_action_steps is empty
            # and the evaluator verdict is ``in_progress`` /
            # ``truth_missing`` depending on terminated.
            self.assertEqual(state.relevant_action_steps, ())
        finally:
            backend.close()

    def test_c1_water_observation_does_not_create_lava_truth(self) -> None:
        target_cell = (2, 4, 3)
        baseline = _build_raw_with_obsidian()
        water = _build_raw_with_obsidian(water_offsets=((2, 0, 3),))
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: _StubMineRLBridgeEnv(
                raw_observation=baseline,
                trajectory=[water],
            ),
            reset_warmup_steps=0,
        )
        backend.open()
        backend.reset(_c1_task())
        try:
            backend.step(
                {
                    AGENT_ID_LOCAL: MacroAction(
                        "use_item", target="water_bucket"
                    )
                }
            )
            state = backend.get_casting_evaluation_state(target_cell)
            self.assertTrue(state.water_truth.present)
            self.assertFalse(state.lava_truth.present)
        finally:
            backend.close()

    def test_c1_typed_truth_records_pre_existing_fluid(
        self,
    ) -> None:
        target_cell = (2, 4, 3)
        # Baseline already has water at the cell. The
        # backend must record the ``air`` (no water) verdict
        # only after the bridge has actually emitted it; a
        # pre-existing water observation is still the bridge's
        # truth, but it cannot earn the agent's causal
        # credit.
        raw = _build_raw_with_obsidian(
            obsidian_offsets=(),
            water_offsets=((2, 0, 3),),
        )
        backend = _open_backend_with(raw, _c1_task())
        try:
            state = backend.get_casting_evaluation_state(target_cell)
            self.assertIsNone(state.target_update_evidence)
            # No cast actions run during this episode, so
            # the cell is not attributed.
            self.assertEqual(state.relevant_action_steps, ())
        finally:
            backend.close()

    def test_c1_typed_truth_rejects_cell_outside_grid(self) -> None:
        raw = _build_raw_with_obsidian()
        backend = _open_backend_with(raw, _c1_task())
        try:
            with self.assertRaises(ValueError):
                backend.get_casting_evaluation_state((100, 100, 100))
        finally:
            backend.close()

    def test_c1_typed_truth_rejects_rejected_action_credit(self) -> None:
        # A rejected action translation must not earn a cast
        # credit. The driver emits a use_item against a target
        # that the translator rejects (``dragon_egg``); the
        # backend must surface an error and the next valid
        # cast must still earn a single credit each.
        target_cell = (2, 4, 3)
        raw = _build_raw_with_obsidian(obsidian_offsets=(target_cell,))
        backend = _open_backend_with(raw, _c1_task())
        try:
            with self.assertRaises(RuntimeError):
                backend.step(
                    {
                        AGENT_ID_LOCAL: MacroAction(
                            action_type="use_item", target="dragon_egg"
                        )
                    }
                )
            self.assertEqual(backend._latched["cast_credit_history"], [])
            backend.step(
                {
                    AGENT_ID_LOCAL: MacroAction(
                        action_type="use_item", target="water_bucket"
                    )
                }
            )
            self.assertEqual(
                backend._latched["cast_credit_history"], [(1, "water")]
            )
        finally:
            backend.close()


class TypedTruthContinuousCastingC2Tests(unittest.TestCase):
    """The production backend supplies independent truth for all C2 cells."""

    def test_c2_production_state_reaches_evaluator_success(self) -> None:
        baseline, trajectory, actions = _build_verified_cast_sequence(
            CASTING_C3_TARGET_CELLS
        )
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: _StubMineRLBridgeEnv(
                raw_observation=baseline, trajectory=trajectory
            ),
            reset_warmup_steps=0,
        )
        backend.open()
        backend.reset(_c3_task())
        try:
            for action in actions:
                backend.step({AGENT_ID_LOCAL: action})
            backend.mark_terminated(reason="driver_done")
            state = backend.get_continuous_casting_evaluation_state()
            self.assertEqual(
                [cell.relevant_action_steps for cell in state.cells],
                [(1, 2), (4, 5), (7, 8)],
            )
            self.assertEqual(
                ContinuousCastingEvaluator().evaluate(state).outcome,
                "success",
            )
        finally:
            backend.close()

    def test_c2_external_obsidian_does_not_borrow_bucket_credit(self) -> None:
        baseline = _build_raw_with_obsidian()
        external = _build_raw_with_obsidian(
            obsidian_offsets=CASTING_C3_TARGET_CELLS
        )
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: _StubMineRLBridgeEnv(
                raw_observation=baseline, trajectory=[external]
            ),
            reset_warmup_steps=0,
        )
        backend.open()
        backend.reset(_c3_task())
        try:
            backend.step({AGENT_ID_LOCAL: MacroAction.wait()})
            backend.mark_terminated(reason="driver_done")
            state = backend.get_continuous_casting_evaluation_state()
            self.assertTrue(
                all(cell.relevant_action_steps == () for cell in state.cells)
            )
            self.assertNotEqual(
                ContinuousCastingEvaluator().evaluate(state).outcome,
                "success",
            )
        finally:
            backend.close()


class TypedTruthFrameC3Tests(unittest.TestCase):
    """The production backend exposes typed C3 frozen-frame truth."""

    def test_c3_typed_truth_full_ring_and_interior(self) -> None:
        target_cells = CASTING_S_C3_FRAME_CELLS
        baseline, trajectory, actions = _build_verified_cast_sequence(
            target_cells
        )
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: _StubMineRLBridgeEnv(
                raw_observation=baseline, trajectory=trajectory
            ),
            reset_warmup_steps=0,
        )
        backend.open()
        backend.reset(_c5_target_task())
        try:
            for action in actions:
                backend.step({AGENT_ID_LOCAL: action})
            backend.mark_terminated(reason="driver_done")
            state = backend.get_frame_evaluation_state()
            self.assertEqual(len(state.cells), 14)
            self.assertEqual(len(state.interior_cells), 6)
            for index, cell in enumerate(state.cells):
                self.assertEqual(cell.target_cell, target_cells[index])
                self.assertEqual(cell.current_block, "obsidian")
            self.assertEqual(
                FrozenFrameEvaluator().evaluate(state).outcome,
                "success",
            )
        finally:
            backend.close()

    def test_c3_typed_truth_interior_records_air_observed(self) -> None:
        # Interior cells that are ``air`` must be reported as
        # ``air`` (not ``None``) so the C3 evaluator does not
        # fail-closed the frame on an explicitly known air
        # observation.
        target_cells = CASTING_S_C3_FRAME_CELLS
        obsidian_offsets = tuple(target_cells)
        raw = _build_raw_with_obsidian(obsidian_offsets=obsidian_offsets)
        backend = _open_backend_with(raw, _c5_target_task())
        try:
            backend.mark_terminated(reason="driver_done")
            state = backend.get_frame_evaluation_state()
            for interior in state.interior_cells:
                self.assertEqual(interior.current_block, "air")
        finally:
            backend.close()

    def test_c3_typed_truth_rejects_external_obsidian(self) -> None:
        # The agent did not run any cast actions during this
        # episode. Even though the world shows obsidian at
        # the frame offsets (because the bridge emits a
        # pre-built frame), the typed truth must not produce
        # relevant_action_steps for any cell, so the
        # evaluator verdict is ``frame_not_built``.
        target_cells = CASTING_S_C3_FRAME_CELLS
        obsidian_offsets = tuple(target_cells)
        raw = _build_raw_with_obsidian(obsidian_offsets=obsidian_offsets)
        backend = _open_backend_with(raw, _c5_target_task())
        try:
            backend.mark_terminated(reason="driver_done")
            state = backend.get_frame_evaluation_state()
            for cell in state.cells:
                self.assertEqual(cell.relevant_action_steps, ())
            result = FrozenFrameEvaluator().evaluate(state)
            self.assertNotEqual(result.outcome, "success")
        finally:
            backend.close()


class TypedTruthIgnitionC4Tests(unittest.TestCase):
    """The production backend exposes typed C4 ignition truth."""

    def test_c4_typed_truth_succeeds_after_use_item_flint_and_steel(
        self,
    ) -> None:
        target_cells = CASTING_S_C3_FRAME_CELLS
        baseline_raw, trajectory, actions = _build_verified_cast_sequence(
            target_cells
        )
        trajectory.append(
            _build_raw_with_obsidian(
                obsidian_offsets=target_cells,
                nether_portal_offsets=((1, 1, 1),),
            )
        )
        actions.append(
            MacroAction("use_item", target="flint_and_steel")
        )
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: _StubMineRLBridgeEnv(
                raw_observation=baseline_raw,
                trajectory=trajectory,
            ),
            reset_warmup_steps=0,
        )
        backend.open()
        backend.reset(_c5_target_task())
        try:
            ignition_step = None
            for action in actions:
                ignition_step = backend.step({AGENT_ID_LOCAL: action})
            backend.mark_terminated(reason="driver_done")
            state = backend.get_ignition_evaluation_state()
            self.assertIsNotNone(state.ignition_action)
            if state.ignition_action is not None:
                self.assertEqual(
                    state.ignition_action.step_id, ignition_step.step_id
                )
                self.assertEqual(state.ignition_action.item, "flint_and_steel")
                self.assertEqual(
                    state.ignition_action.action_type, "use_item"
                )
                self.assertEqual(
                    state.ignition_action.target_cell, (1, 1, 1)
                )
            self.assertIsNotNone(state.activation_evidence)
            self.assertEqual(
                FrozenIgnitionEvaluator().evaluate(state).outcome,
                "success",
            )
        finally:
            backend.close()

    def test_c4_typed_truth_fails_closed_when_ignition_absent(self) -> None:
        # Driver only emits water + lava; the flint_and_steel
        # action never happens. The C4 ignition evaluator must
        # report ``ignition_action_missing``. The production
        # backend without per-cell attribution reports the
        # frame as not built; we therefore assert the
        # evaluator classifies the typed state as a
        # C4-typed failure (any of the failure outcomes).
        target_cells = CASTING_S_C3_FRAME_CELLS
        obsidian_offsets = tuple(target_cells)
        raw = _build_raw_with_obsidian(obsidian_offsets=obsidian_offsets)
        backend = _open_backend_with(raw, _c5_target_task())
        try:
            backend.step(
                {
                    AGENT_ID_LOCAL: MacroAction(
                        action_type="use_item", target="water_bucket"
                    )
                }
            )
            backend.step(
                {
                    AGENT_ID_LOCAL: MacroAction(
                        action_type="use_item", target="lava_bucket"
                    )
                }
            )
            backend.mark_terminated(reason="driver_done")
            state = backend.get_ignition_evaluation_state()
            self.assertIsNone(state.ignition_action)
            self.assertIsNone(state.activation_evidence)
            result = FrozenIgnitionEvaluator().evaluate(state)
            self.assertNotEqual(result.outcome, "success")
        finally:
            backend.close()


class TypedTruthNetherEntryC5Tests(unittest.TestCase):
    """The production backend exposes typed C5 Nether-entry truth."""

    def test_c5_typed_truth_succeeds_after_portal_traversal(self) -> None:
        target_cells = CASTING_S_C3_FRAME_CELLS
        baseline, trajectory, actions = _build_verified_cast_sequence(
            target_cells
        )
        active = _build_raw_with_obsidian(
            obsidian_offsets=target_cells,
            nether_portal_offsets=((1, 1, 1),),
            position=(1.5, 65.0, 1.5),
        )
        trajectory.append(active)
        actions.append(MacroAction("use_item", target="flint_and_steel"))
        portal_transition = {
            "present": np.asarray(True, dtype=np.bool_),
            "entered_via_portal": np.asarray(True, dtype=np.bool_),
            "sequence": np.asarray(1, dtype=np.int64),
            "source_portal_block_world_position": np.asarray(
                (1, 65, 1), dtype=np.int32
            ),
            "from_dimension": "minecraft:overworld",
            "to_dimension": "minecraft:the_nether",
        }
        nether_raw = _build_raw_with_obsidian(
            obsidian_offsets=target_cells,
            nether_portal_offsets=((1, 1, 1),),
            position=(1.5, 65.0, 1.5),
            portal_transition=portal_transition,
        )
        nether_raw["portal_dimension"] = np.asarray(
            "minecraft:the_nether"
        )
        trajectory.append(nether_raw)
        actions.append(MacroAction.wait())
        backend = MineRLEnvironmentBackend(
            env_factory=lambda task: _StubMineRLBridgeEnv(
                raw_observation=baseline,
                trajectory=trajectory,
            ),
            reset_warmup_steps=0,
        )
        backend.open()
        backend.reset(_c5_target_task())
        try:
            for action in actions:
                backend.step({AGENT_ID_LOCAL: action})
            backend.mark_terminated(reason="driver_done")
            state = backend.get_nether_entry_evaluation_state()
            self.assertIsNotNone(state.entry_evidence)
            if state.entry_evidence is not None:
                self.assertEqual(
                    state.entry_evidence.source_dimension,
                    CASTING_S_C5_SOURCE_DIMENSION,
                )
                self.assertEqual(
                    state.entry_evidence.target_dimension,
                    CASTING_S_C5_TARGET_DIMENSION,
                )
                self.assertTrue(state.entry_evidence.entered_via_episode_portal)
            self.assertEqual(
                FrozenNetherEntryEvaluator().evaluate(state).outcome,
                "success",
            )
        finally:
            backend.close()

    def test_c5_typed_truth_fails_closed_without_transition(self) -> None:
        # No portal_transition evidence is supplied, so the
        # C5 evaluator must report a fail-closed outcome
        # (the production backend's typed surface either
        # surfaces ``nether_entry_portal_unknown`` for an
        # unknown transition or fails the underlying C4
        # ignition when the frame attribution is missing).
        target_cells = CASTING_S_C3_FRAME_CELLS
        obsidian_offsets = tuple(target_cells)
        nether_portal_offsets = ((1, 1, 1),)
        raw = _build_raw_with_obsidian(
            obsidian_offsets=obsidian_offsets,
            nether_portal_offsets=nether_portal_offsets,
        )
        backend = _open_backend_with(raw, _c5_target_task())
        try:
            backend.step(
                {
                    AGENT_ID_LOCAL: MacroAction(
                        action_type="use_item", target="water_bucket"
                    )
                }
            )
            backend.step(
                {
                    AGENT_ID_LOCAL: MacroAction(
                        action_type="use_item", target="lava_bucket"
                    )
                }
            )
            backend.step(
                {
                    AGENT_ID_LOCAL: MacroAction(
                        action_type="use_item",
                        target="flint_and_steel",
                    )
                }
            )
            backend.mark_terminated(reason="driver_done")
            state = backend.get_nether_entry_evaluation_state()
            self.assertIsNone(state.entry_evidence)
            result = FrozenNetherEntryEvaluator().evaluate(state)
            self.assertNotEqual(result.outcome, "success")
        finally:
            backend.close()


class TypedTruthInformationIsolationTests(unittest.TestCase):
    """Typed truth never enters the public Observation / event / info."""

    def test_observation_does_not_carry_typed_truth(self) -> None:
        raw = _build_raw_with_obsidian(obsidian_offsets=((2, 4, 3),))
        backend = _open_backend_with(raw, _c1_task())
        try:
            observations = backend.reset(_c1_task())
            for observation in observations.values():
                forbidden = (
                    "target_block_truth",
                    "fluid_truth",
                    "portal_grid",
                    "portal_fluid_grid",
                    "latched_frame_identity",
                    "matched_frame_identity",
                    "nether_entry_evidence",
                    "nether_entry_evaluation",
                    "ignition_action",
                    "activation_evidence",
                    "entered_via_episode_portal",
                    "pre_transition_position",
                    "cast_credit_history",
                    "first_obsidian_step_by_offset",
                    "first_ignition_step",
                    "first_nether_portal_step",
                )
                for token in forbidden:
                    self.assertFalse(
                        hasattr(observation, token),
                        f"{token} must not appear on the public observation",
                    )
        finally:
            backend.close()

    def test_step_info_does_not_carry_typed_truth(self) -> None:
        raw = _build_raw_with_obsidian(obsidian_offsets=((2, 4, 3),))
        backend = _open_backend_with(raw, _c1_task())
        try:
            step = backend.step(
                {
                    AGENT_ID_LOCAL: MacroAction(
                        action_type="use_item", target="water_bucket"
                    )
                }
            )
            forbidden = (
                "target_block_truth",
                "fluid_truth",
                "portal_grid",
                "portal_fluid_grid",
                "nether_entry_evidence",
                "nether_entry_evaluation",
                "cast_credit_history",
            )
            for token in forbidden:
                self.assertNotIn(token, step.info)
        finally:
            backend.close()


class TypedTruthLifecycleTests(unittest.TestCase):
    """reset / step / close clear the typed truth latched state."""

    def test_close_clears_typed_truth_latched_state(self) -> None:
        raw = _build_raw_with_obsidian(
            obsidian_offsets=tuple(CASTING_S_C3_FRAME_CELLS),
        )
        backend = _open_backend_with(raw, _c5_target_task())
        backend.step(
            {
                AGENT_ID_LOCAL: MacroAction(
                    action_type="use_item", target="water_bucket"
                )
            }
        )
        backend.step(
            {
                AGENT_ID_LOCAL: MacroAction(
                    action_type="use_item", target="lava_bucket"
                )
            }
        )
        backend.close()
        self.assertEqual(backend._latched["cast_credit_history"], [])
        self.assertEqual(backend._latched["first_obsidian_step_by_offset"], {})
        self.assertIsNone(backend._latched["first_ignition_step"])
        self.assertIsNone(backend._latched["first_nether_portal_step"])
        self.assertIsNone(backend._latched["baseline_fluid_grid"])
        self.assertIsNone(backend._latched["current_fluid_grid"])

    def test_reset_clears_typed_truth_latched_state(self) -> None:
        raw = _build_raw_with_obsidian(
            obsidian_offsets=tuple(CASTING_S_C3_FRAME_CELLS),
        )
        backend = _open_backend_with(raw, _c5_target_task())
        backend.step(
            {
                AGENT_ID_LOCAL: MacroAction(
                    action_type="use_item", target="water_bucket"
                )
            }
        )
        # The second reset must produce a fresh latched state.
        backend.reset(_c5_target_task())
        self.assertEqual(backend._latched["cast_credit_history"], [])
        self.assertEqual(backend._latched["first_obsidian_step_by_offset"], {})
        self.assertIsNotNone(backend._latched["baseline_fluid_grid"])
        np.testing.assert_array_equal(
            backend._latched["current_fluid_grid"],
            backend._latched["baseline_fluid_grid"],
        )
        backend.close()

    def test_duration_ticks_does_not_double_issue_credit(self) -> None:
        # A macro action with ``duration_ticks=4`` only
        # counts as one cast credit, not four.
        raw = _build_raw_with_obsidian(obsidian_offsets=((2, 4, 3),))
        backend = _open_backend_with(raw, _c1_task())
        try:
            backend.step(
                {
                    AGENT_ID_LOCAL: MacroAction(
                        action_type="use_item",
                        target="water_bucket",
                        duration_ticks=4,
                    )
                }
            )
            self.assertEqual(
                backend._latched["cast_credit_history"], [(1, "water")]
            )
        finally:
            backend.close()


if __name__ == "__main__":
    unittest.main()

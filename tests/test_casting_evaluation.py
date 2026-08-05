"""Offline tests for the R3 ``casting_c1_fixed`` evaluator.

These tests prove, in code, that:

* :class:`obsidianlink.evaluation.casting.CastingEvaluationState`
  and :class:`obsidianlink.evaluation.casting.CastingEvaluationResult`
  are immutable, type-strict, JSON-serializable evaluators' surface.
* :class:`obsidianlink.evaluation.casting.CastingEvaluator` is a
  pure deterministic object. The same state always produces the
  same result; the priority order is locked.
* Evaluator truth (target cell, block ids, fluid evidence, budget
  numbers, outcome) never leaks into an Agent-visible
  :class:`Observation`.
* :class:`obsidianlink.env.fake.FakeEnvironmentBackend` exposes a
  narrow ``set_casting_evaluation_state`` /
  ``get_casting_evaluation_state`` surface that rejects mismatched
  ``episode_id`` / ``step_id``, refuses reads before
  :meth:`reset`, and never copies casting truth into the
  :class:`Observation`.
* The current real MineRL backend still fails closed for the
  casting-c1 task — the R3 evaluator is an offline contract, not a
  bypass for the R2 capability gap.

The tests never start Minecraft, MineRL, or Gradle, and never
import the MineRL bridge at runtime when checking the manifest.
"""

from __future__ import annotations

import dataclasses
import json
import unittest
from typing import Any, Mapping

from obsidianlink.core.types import MacroAction, Observation
from obsidianlink.env.capabilities import (
    BackendCapabilities,
    CapabilityMismatchError,
    assert_casting_c1_capabilities,
)
from obsidianlink.env.fake import FakeEnvironmentBackend
from obsidianlink.env.minerl_backend import MineRLEnvironmentBackend
from obsidianlink.evaluation.casting import (
    DEFAULT_CAUSALITY_WINDOW_STEPS,
    MAX_CAUSALITY_WINDOW_STEPS,
    NORMAL_TERMINATION_REASONS,
    OUTCOME_ABNORMAL_TERMINATION,
    OUTCOME_CAUSALITY_MISSING,
    OUTCOME_IN_PROGRESS,
    OUTCOME_INVALID_INITIAL_STATE,
    OUTCOME_STEP_BUDGET_EXCEEDED,
    OUTCOME_SUCCESS,
    OUTCOME_TIME_BUDGET_EXCEEDED,
    OUTCOME_TRUTH_MISSING,
    OUTCOME_WRONG_BLOCK,
    OUTCOMES,
    TARGET_BLOCK_IDS,
    CastingEvaluationResult,
    CastingEvaluationState,
    CastingEvaluator,
    CastingFluidTruth,
    CastingTransitionEvidence,
)
from obsidianlink.evaluation.portal import (
    EvaluationState,
    PortalEvaluator,
)
from tests.helpers import casting_c1_task


# Stable target cell used across tests. Mirrors the JSON contract
# (and is therefore identical to ``casting_c1_task.scenario_parameters``).
TARGET_CELL: tuple[int, int, int] = (2, 4, 3)
EPISODE_ID = "casting_c1_fixed_seed_0"


def _success_base_state(
    *,
    episode_id: str = EPISODE_ID,
    step_id: int = 30,
    agent_id: str | None = None,
    target_cell: tuple[int, int, int] = TARGET_CELL,
    initial_target_block: str | None = "air",
    current_target_block: str | None = "obsidian",
    target_update_evidence: CastingTransitionEvidence | None = None,
    water_truth: CastingFluidTruth | None = None,
    lava_truth: CastingFluidTruth | None = None,
    relevant_action_steps: tuple[int, ...] = (20, 24),
    causality_window_steps: int = DEFAULT_CAUSALITY_WINDOW_STEPS,
    episode_terminated: bool = True,
    terminated_step: int | None = 30,
    terminated_reason: str | None = "driver_done",
    current_time_seconds: float = 80.0,
    max_environment_steps: int = 160,
    max_game_time_seconds: float = 120.0,
    evidence: Mapping[str, Any] | None = None,
) -> CastingEvaluationState:
    """Build a fully-populated, *successful* evaluator state.

    The defaults produce ``OUTCOME_SUCCESS`` when the evaluator is
    invoked. Tests then mutate the returned state via
    :func:`dataclasses.replace` to exercise the priority rules.
    The optional evidence defaults are filled in only when the
    caller did not pass them.
    """
    if target_update_evidence is None:
        target_update_evidence = CastingTransitionEvidence(
            before_block=initial_target_block,
            after_block=current_target_block,
            update_step=25,
        )
    if water_truth is None:
        water_truth = CastingFluidTruth(present=True, evidence_step=20)
    if lava_truth is None:
        lava_truth = CastingFluidTruth(present=True, evidence_step=22)
    return CastingEvaluationState(
        episode_id=episode_id,
        step_id=step_id,
        agent_id=agent_id,
        target_cell=target_cell,
        initial_target_block=initial_target_block,
        current_target_block=current_target_block,
        target_update_evidence=target_update_evidence,
        water_truth=water_truth,
        lava_truth=lava_truth,
        relevant_action_steps=relevant_action_steps,
        causality_window_steps=causality_window_steps,
        episode_terminated=episode_terminated,
        terminated_step=terminated_step,
        terminated_reason=terminated_reason,
        current_time_seconds=current_time_seconds,
        max_environment_steps=max_environment_steps,
        max_game_time_seconds=max_game_time_seconds,
        evidence={} if evidence is None else evidence,
    )


def _state_at_step(
    step_id: int, **overrides: Any
) -> CastingEvaluationState:
    """Build a *successful* state whose ``step_id`` matches the
    current backend step. Used by the FakeBackend tests so the
    identity guard does not fire spuriously.
    """
    base: dict[str, Any] = {
        "step_id": step_id,
        "terminated_step": step_id,
        "target_update_evidence": CastingTransitionEvidence(
            before_block="air",
            after_block="obsidian",
            update_step=step_id,
        ),
        "relevant_action_steps": (max(0, step_id - 1),),
        "water_truth": CastingFluidTruth(
            present=True, evidence_step=max(0, step_id - 1)
        ),
        "lava_truth": CastingFluidTruth(
            present=True, evidence_step=max(0, step_id - 1)
        ),
    }
    base.update(overrides)
    return _success_base_state(**base)


class CastingStateImmutabilityTests(unittest.TestCase):
    """The state and result dataclasses must be frozen and serializable."""

    def test_state_is_frozen(self) -> None:
        state = _success_base_state()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            state.episode_id = "other"  # type: ignore[misc]

    def test_state_evidence_is_mapping_proxy(self) -> None:
        state = _success_base_state(
            evidence={"nested": {"steps": [1, 2]}}
        )
        self.assertIsInstance(state.evidence, Mapping)
        # The underlying mapping must be read-only.
        with self.assertRaises(TypeError):
            state.evidence["x"] = 1  # type: ignore[index]
        nested = state.evidence["nested"]
        self.assertIsInstance(nested, Mapping)
        with self.assertRaises(TypeError):
            nested["x"] = 1  # type: ignore[index]
        self.assertEqual(nested["steps"], (1, 2))

    def test_state_relevant_action_steps_is_tuple_of_ints(self) -> None:
        state = _success_base_state()
        self.assertIsInstance(state.relevant_action_steps, tuple)
        for step in state.relevant_action_steps:
            self.assertIs(type(step), int)

    def test_result_is_frozen(self) -> None:
        result = CastingEvaluator().evaluate(_success_base_state())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.success = False  # type: ignore[misc]

    def test_result_evidence_is_mapping_proxy(self) -> None:
        result = CastingEvaluator().evaluate(_success_base_state())
        self.assertIsInstance(result.evidence, Mapping)
        with self.assertRaises(TypeError):
            result.evidence["x"] = 1  # type: ignore[index]
        self.assertIsInstance(result.evidence["target_update_evidence"], Mapping)
        with self.assertRaises(TypeError):
            result.evidence["target_update_evidence"]["after_block"] = "air"
        self.assertEqual(result.evidence["target_cell"], TARGET_CELL)

    def test_result_is_json_serializable(self) -> None:
        result = CastingEvaluator().evaluate(_success_base_state())
        snapshot = result.as_dict()
        payload = json.dumps(snapshot)
        round_tripped = json.loads(payload)
        self.assertEqual(round_tripped["outcome"], OUTCOME_SUCCESS)
        self.assertTrue(round_tripped["success"])
        self.assertEqual(round_tripped["episode_id"], EPISODE_ID)
        snapshot["evidence"]["target_cell"].append(99)
        self.assertEqual(result.evidence["target_cell"], TARGET_CELL)

    def test_repeated_evaluate_is_deterministic(self) -> None:
        state = _success_base_state()
        evaluator = CastingEvaluator()
        first = evaluator.evaluate(state)
        second = evaluator.evaluate(state)
        third = evaluator.evaluate(state)
        self.assertEqual(
            (first.outcome, first.success, first.evidence),
            (second.outcome, second.success, second.evidence),
        )
        self.assertEqual(
            (first.outcome, first.success),
            (third.outcome, third.success),
        )


class CastingStateValidationTests(unittest.TestCase):
    """Invalid state fields must be rejected at construction time."""

    def test_empty_episode_id_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "episode_id"):
            _success_base_state(episode_id="")

    def test_non_int_step_id_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "step_id"):
            dataclasses.replace(_success_base_state(), step_id="0")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "step_id"):
            dataclasses.replace(_success_base_state(), step_id=-1)

    def test_invalid_target_cell_rejected(self) -> None:
        # bools are not allowed as coordinates.
        with self.assertRaisesRegex(ValueError, "target_cell"):
            dataclasses.replace(
                _success_base_state(),
                target_cell=(True, 4, 3),  # type: ignore[arg-type]
            )
        # Wrong length rejected.
        with self.assertRaisesRegex(ValueError, "target_cell"):
            dataclasses.replace(
                _success_base_state(),
                target_cell=(2, 4),  # type: ignore[arg-type]
            )
        # Wrong element type rejected.
        with self.assertRaisesRegex(ValueError, "target_cell"):
            dataclasses.replace(
                _success_base_state(),
                target_cell=(2, 4, "3"),  # type: ignore[arg-type]
            )

    def test_invalid_block_id_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "initial_target_block"):
            dataclasses.replace(
                _success_base_state(), initial_target_block="bedrock"  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "current_target_block"):
            dataclasses.replace(
                _success_base_state(), current_target_block="dirt"  # type: ignore[arg-type]
            )

    def test_invalid_budget_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_environment_steps"):
            dataclasses.replace(_success_base_state(), max_environment_steps=0)
        with self.assertRaisesRegex(ValueError, "max_game_time_seconds"):
            dataclasses.replace(
                _success_base_state(), max_game_time_seconds=float("inf")
            )
        with self.assertRaisesRegex(ValueError, "max_game_time_seconds"):
            dataclasses.replace(
                _success_base_state(), max_game_time_seconds=float("nan")
            )
        with self.assertRaisesRegex(ValueError, "max_game_time_seconds"):
            dataclasses.replace(_success_base_state(), max_game_time_seconds=0)
        with self.assertRaisesRegex(ValueError, "current_time_seconds"):
            dataclasses.replace(
                _success_base_state(), current_time_seconds=float("nan")
            )
        with self.assertRaisesRegex(ValueError, "current_time_seconds"):
            dataclasses.replace(_success_base_state(), current_time_seconds=-1)

    def test_future_evidence_steps_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "relevant_action_steps"):
            _success_base_state(step_id=30, relevant_action_steps=(31,))
        with self.assertRaisesRegex(ValueError, "water_truth"):
            _success_base_state(
                step_id=30,
                water_truth=CastingFluidTruth(present=True, evidence_step=31),
            )
        with self.assertRaisesRegex(ValueError, "target_update_evidence"):
            _success_base_state(
                step_id=30,
                target_update_evidence=CastingTransitionEvidence(
                    before_block="air",
                    after_block="obsidian",
                    update_step=31,
                ),
            )
        with self.assertRaisesRegex(ValueError, "terminated_step"):
            _success_base_state(step_id=30, terminated_step=31)

    def test_causality_window_bounds_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "causality_window_steps"):
            dataclasses.replace(
                _success_base_state(), causality_window_steps=0
            )
        with self.assertRaisesRegex(ValueError, "causality_window_steps"):
            dataclasses.replace(
                _success_base_state(),
                causality_window_steps=MAX_CAUSALITY_WINDOW_STEPS + 1,
            )

    def test_terminated_requires_step_and_consistent_reason(self) -> None:
        with self.assertRaisesRegex(ValueError, "terminated_step"):
            dataclasses.replace(
                _success_base_state(), episode_terminated=True, terminated_step=None
            )
        with self.assertRaisesRegex(ValueError, "terminated"):
            dataclasses.replace(
                _success_base_state(),
                episode_terminated=False,
                terminated_step=10,
            )
        with self.assertRaisesRegex(ValueError, "terminated_reason"):
            dataclasses.replace(
                _success_base_state(),
                episode_terminated=True,
                terminated_step=10,
                terminated_reason="",
            )

    def test_relevant_action_steps_must_be_ints(self) -> None:
        with self.assertRaisesRegex(ValueError, "relevant_action_steps"):
            dataclasses.replace(
                _success_base_state(),
                relevant_action_steps=(1, "2", 3),  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "relevant_action_steps"):
            dataclasses.replace(
                _success_base_state(),
                relevant_action_steps=(-1,),  # type: ignore[arg-type]
            )

    def test_fluid_truth_tri_state_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "present"):
            CastingFluidTruth(present="yes")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "evidence_step"):
            CastingFluidTruth(present=True, evidence_step=-1)


class CastingEvaluatorOutcomeTests(unittest.TestCase):
    """Outcome classification for the casting-c1 priority rules."""

    def test_complete_legal_state_is_success(self) -> None:
        result = CastingEvaluator().evaluate(_success_base_state())
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)
        self.assertEqual(result.failure_type, None)
        self.assertEqual(result.blocking_conditions, ())

    def test_target_already_obsidian_at_reset_is_invalid(self) -> None:
        state = dataclasses.replace(
            _success_base_state(), initial_target_block="obsidian"
        )
        result = CastingEvaluator().evaluate(state)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, OUTCOME_INVALID_INITIAL_STATE)
        self.assertEqual(result.failure_type, OUTCOME_INVALID_INITIAL_STATE)
        self.assertIn(
            "target_already_obsidian_at_reset", result.blocking_conditions
        )

    def test_target_obsidian_but_episode_not_terminated(self) -> None:
        state = dataclasses.replace(
            _success_base_state(
                current_target_block="obsidian",
                target_update_evidence=CastingTransitionEvidence(
                    before_block="air",
                    after_block="obsidian",
                    update_step=25,
                ),
                relevant_action_steps=(20, 24),
                episode_terminated=False,
                terminated_step=None,
                terminated_reason=None,
                step_id=25,
            )
        )
        result = CastingEvaluator().evaluate(state)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, OUTCOME_IN_PROGRESS)
        self.assertEqual(result.failure_type, None)
        self.assertIn("episode_not_terminated", result.blocking_conditions)

    def test_target_wrong_block_cobblestone(self) -> None:
        state = dataclasses.replace(
            _success_base_state(current_target_block="cobblestone")
        )
        result = CastingEvaluator().evaluate(state)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, OUTCOME_WRONG_BLOCK)
        self.assertEqual(result.failure_type, OUTCOME_WRONG_BLOCK)
        self.assertIn(
            "wrong_block:expected_obsidian_got_cobblestone",
            result.blocking_conditions,
        )
        # Even when the block is wrong, the evaluator must not
        # claim causality evidence.
        self.assertIn("actual_block", result.evidence)
        self.assertEqual(result.evidence["actual_block"], "cobblestone")

    def test_target_wrong_block_stone_and_air(self) -> None:
        for block in ("stone", "air", "water", "lava", "missing"):
            state = dataclasses.replace(
                _success_base_state(current_target_block=block)
            )
            result = CastingEvaluator().evaluate(state)
            self.assertFalse(result.success)
            self.assertEqual(result.outcome, OUTCOME_WRONG_BLOCK)
            self.assertIn(
                f"wrong_block:expected_obsidian_got_{block}",
                result.blocking_conditions,
            )

    def test_truth_missing_initial_block(self) -> None:
        state = dataclasses.replace(
            _success_base_state(), initial_target_block=None
        )
        result = CastingEvaluator().evaluate(state)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)
        self.assertIn(
            "missing_truth:initial_target_block", result.blocking_conditions
        )

    def test_truth_missing_current_block(self) -> None:
        state = dataclasses.replace(
            _success_base_state(), current_target_block=None
        )
        result = CastingEvaluator().evaluate(state)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)
        self.assertIn(
            "missing_truth:current_target_block", result.blocking_conditions
        )

    def test_truth_missing_water(self) -> None:
        state = dataclasses.replace(_success_base_state(), water_truth=None)
        result = CastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)
        self.assertIn("missing_truth:water_truth", result.blocking_conditions)

        # Tri-state None on ``present`` also counts as missing.
        state = dataclasses.replace(
            _success_base_state(),
            water_truth=CastingFluidTruth(present=None),
        )
        result = CastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)

    def test_truth_missing_lava(self) -> None:
        state = dataclasses.replace(_success_base_state(), lava_truth=None)
        result = CastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)
        self.assertIn("missing_truth:lava_truth", result.blocking_conditions)

        state = dataclasses.replace(
            _success_base_state(),
            lava_truth=CastingFluidTruth(present=None),
        )
        result = CastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)

    def test_truth_missing_update_evidence(self) -> None:
        # No update evidence at all.
        state = dataclasses.replace(
            _success_base_state(), target_update_evidence=None
        )
        result = CastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)
        self.assertIn(
            "missing_truth:target_update_evidence", result.blocking_conditions
        )

        # Update evidence with missing after_block.
        state = dataclasses.replace(
            _success_base_state(),
            target_update_evidence=CastingTransitionEvidence(
                before_block="air", after_block=None, update_step=25
            ),
        )
        result = CastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)

        # Update evidence with missing update_step.
        state = dataclasses.replace(
            _success_base_state(),
            target_update_evidence=CastingTransitionEvidence(
                before_block="air", after_block="obsidian", update_step=None
            ),
        )
        result = CastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)

        # Update evidence with missing before_block.
        state = dataclasses.replace(
            _success_base_state(),
            target_update_evidence=CastingTransitionEvidence(
                before_block=None, after_block="obsidian", update_step=25
            ),
        )
        result = CastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)

    def test_truth_missing_termination_reason(self) -> None:
        state = dataclasses.replace(
            _success_base_state(), terminated_reason=None
        )
        result = CastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)
        self.assertIn(
            "missing_truth:terminated_reason", result.blocking_conditions
        )

    def test_explicitly_absent_fluid_blocks_success(self) -> None:
        for field_name in ("water_truth", "lava_truth"):
            state = dataclasses.replace(
                _success_base_state(),
                **{
                    field_name: CastingFluidTruth(
                        present=False, evidence_step=20
                    )
                },
            )
            result = CastingEvaluator().evaluate(state)
            self.assertFalse(result.success)
            self.assertEqual(result.outcome, OUTCOME_CAUSALITY_MISSING)
            self.assertIn(
                f"causality_missing:{field_name.removesuffix('_truth')}_not_present",
                result.blocking_conditions,
            )

    def test_transition_must_end_in_obsidian(self) -> None:
        state = dataclasses.replace(
            _success_base_state(),
            target_update_evidence=CastingTransitionEvidence(
                before_block="air",
                after_block="cobblestone",
                update_step=25,
            ),
        )
        result = CastingEvaluator().evaluate(state)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, OUTCOME_CAUSALITY_MISSING)
        self.assertIn(
            "causality_missing:transition_did_not_produce_obsidian",
            result.blocking_conditions,
        )

    def test_truth_missing_relevant_action_steps(self) -> None:
        state = dataclasses.replace(
            _success_base_state(), relevant_action_steps=()
        )
        result = CastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)
        self.assertIn(
            "missing_truth:relevant_action_steps", result.blocking_conditions
        )

    def test_block_change_before_relevant_action(self) -> None:
        # Update happened at step 10, but the only relevant action
        # is at step 20 — the change precedes the action.
        state = dataclasses.replace(
            _success_base_state(
                target_update_evidence=CastingTransitionEvidence(
                    before_block="air", after_block="obsidian", update_step=10
                ),
                relevant_action_steps=(20,),
            )
        )
        result = CastingEvaluator().evaluate(state)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, OUTCOME_CAUSALITY_MISSING)
        self.assertEqual(result.failure_type, OUTCOME_CAUSALITY_MISSING)
        self.assertIn(
            "causality_missing:update_before_any_action",
            result.blocking_conditions,
        )

    def test_block_change_outside_finite_window(self) -> None:
        # The last relevant action is at step 20; the update is at
        # step 25. The default window is 4, so the delta 5 must
        # fail causality.
        state = dataclasses.replace(
            _success_base_state(
                target_update_evidence=CastingTransitionEvidence(
                    before_block="air", after_block="obsidian", update_step=25
                ),
                relevant_action_steps=(20,),
                causality_window_steps=4,
            )
        )
        result = CastingEvaluator().evaluate(state)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, OUTCOME_CAUSALITY_MISSING)
        self.assertIn(
            "causality_missing:outside_window", result.blocking_conditions
        )
        self.assertEqual(result.evidence["causality_delta_steps"], 5)

    def test_block_change_inside_window_is_success(self) -> None:
        # Last action at step 24, update at step 25, window 4 → OK.
        state = dataclasses.replace(
            _success_base_state(
                target_update_evidence=CastingTransitionEvidence(
                    before_block="air", after_block="obsidian", update_step=25
                ),
                relevant_action_steps=(20, 24),
                causality_window_steps=4,
            )
        )
        result = CastingEvaluator().evaluate(state)
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, OUTCOME_SUCCESS)
        self.assertEqual(result.evidence["causality_delta_steps"], 1)
        self.assertEqual(result.evidence["causality_action_step"], 24)

    def test_no_relevant_actions_with_obsidian_target(self) -> None:
        state = dataclasses.replace(
            _success_base_state(relevant_action_steps=())
        )
        # All-truth + obsidian + empty actions → truth_missing
        # already covers this case. The test pins the priority:
        # ``truth_missing`` outranks ``causality_missing``.
        result = CastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)

    def test_step_budget_exceeded(self) -> None:
        # Terminated step > max_environment_steps.
        state = dataclasses.replace(
            _success_base_state(
                step_id=170,
                terminated_step=170,
                max_environment_steps=160,
            )
        )
        result = CastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_STEP_BUDGET_EXCEEDED)
        self.assertEqual(result.failure_type, OUTCOME_STEP_BUDGET_EXCEEDED)
        self.assertIn("step_budget_exceeded", result.blocking_conditions)
        # Budget failure must outrank success / wrong_block.
        self.assertFalse(result.success)

    def test_step_budget_exceeded_in_progress(self) -> None:
        # Not yet terminated, but current step is already past the
        # limit. The evaluator must still report step_budget_exceeded
        # (not in_progress).
        state = dataclasses.replace(
            _success_base_state(
                step_id=200,
                episode_terminated=False,
                terminated_step=None,
                terminated_reason=None,
                max_environment_steps=160,
            )
        )
        result = CastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_STEP_BUDGET_EXCEEDED)

    def test_current_step_cannot_bypass_budget_after_termination(self) -> None:
        state = dataclasses.replace(
            _success_base_state(),
            step_id=200,
            terminated_step=30,
            max_environment_steps=160,
        )
        result = CastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_STEP_BUDGET_EXCEEDED)

    def test_time_budget_exceeded(self) -> None:
        state = dataclasses.replace(
            _success_base_state(
                current_time_seconds=200.0,
                max_game_time_seconds=120.0,
            )
        )
        result = CastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TIME_BUDGET_EXCEEDED)
        self.assertEqual(result.failure_type, OUTCOME_TIME_BUDGET_EXCEEDED)
        self.assertIn("time_budget_exceeded", result.blocking_conditions)

    def test_abnormal_termination(self) -> None:
        state = dataclasses.replace(
            _success_base_state(
                terminated_reason="something_weird",
            )
        )
        result = CastingEvaluator().evaluate(state)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, OUTCOME_ABNORMAL_TERMINATION)
        self.assertEqual(result.failure_type, OUTCOME_ABNORMAL_TERMINATION)
        self.assertIn("abnormal_termination", result.blocking_conditions)

    def test_normal_termination_reasons_are_accepted(self) -> None:
        for reason in NORMAL_TERMINATION_REASONS:
            state = dataclasses.replace(
                _success_base_state(terminated_reason=reason)
            )
            result = CastingEvaluator().evaluate(state)
            self.assertEqual(
                result.outcome,
                OUTCOME_SUCCESS,
                f"reason {reason!r} should not block success",
            )

    def test_priority_is_stable_for_same_input(self) -> None:
        # A state with both budget exceeded AND a wrong block must
        # still report the budget failure (priority is locked).
        state = dataclasses.replace(
            _success_base_state(
                step_id=200,
                current_target_block="cobblestone",
                terminated_step=200,
                max_environment_steps=160,
            )
        )
        result = CastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_STEP_BUDGET_EXCEEDED)

        # A state with budget exceeded AND initial obsidian → still
        # budget failure.
        state = dataclasses.replace(
            _success_base_state(
                step_id=200,
                initial_target_block="obsidian",
                terminated_step=200,
                max_environment_steps=160,
            )
        )
        result = CastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_STEP_BUDGET_EXCEEDED)

        # A state with initial obsidian AND target truth missing →
        # initial state outranks truth_missing? No — invalid_initial_state
        # outranks truth_missing (the initial state is observable, so we
        # can fail on it without needing the rest of the truth).
        state = dataclasses.replace(
            _success_base_state(initial_target_block="obsidian"),
        )
        result = CastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_INVALID_INITIAL_STATE)

    def test_outcome_set_is_closed(self) -> None:
        # The result outcome must always be in the closed OUTCOMES set.
        scenarios = [
            _success_base_state(),
            dataclasses.replace(
                _success_base_state(), initial_target_block="obsidian"
            ),
            dataclasses.replace(
                _success_base_state(), current_target_block="cobblestone"
            ),
            dataclasses.replace(
                _success_base_state(),
                step_id=200,
                terminated_step=200,
                max_environment_steps=160,
            ),
            dataclasses.replace(
                _success_base_state(),
                current_time_seconds=200.0,
                max_game_time_seconds=120.0,
            ),
            dataclasses.replace(
                _success_base_state(terminated_reason="weird_thing")
            ),
            dataclasses.replace(
                _success_base_state(relevant_action_steps=())
            ),
        ]
        for state in scenarios:
            result = CastingEvaluator().evaluate(state)
            self.assertIn(result.outcome, OUTCOMES)


class CastingEvaluatorIsolationTests(unittest.TestCase):
    """The evaluator must read only its typed input state."""

    def test_evaluator_does_not_inspect_observations(self) -> None:
        # Build a state that is *missing* every required truth
        # except termination. The evaluator must return
        # ``truth_missing``; it must not consult any Observation /
        # Agent-visible object.
        state = _success_base_state(
            initial_target_block=None,
            current_target_block=None,
            water_truth=None,
            lava_truth=None,
            target_update_evidence=None,
            relevant_action_steps=(),
        )
        # Build an Observation that contains a *fake* success
        # payload; the evaluator must not look at it.
        observation = Observation(
            episode_id=EPISODE_ID,
            agent_id="agent_1",
            step_id=30,
            timestamp=80.0,
            frame={"target_cell": list(TARGET_CELL), "block": "obsidian"},
            visible_inventory={},
            messages=("task complete",),
            workflow_stage="casting_c1_fixed",
        )
        # The evaluator only takes ``CastingEvaluationState``; it
        # has no way to reach the observation. We assert this by
        # calling it and pinning the outcome.
        result = CastingEvaluator().evaluate(state)
        self.assertEqual(result.outcome, OUTCOME_TRUTH_MISSING)
        # The observation object lives on, untouched.
        self.assertEqual(observation.frame["block"], "obsidian")

    def test_evaluator_signature_only_accepts_state(self) -> None:
        import inspect
        import typing

        sig = inspect.signature(CastingEvaluator.evaluate)
        self.assertEqual(
            list(sig.parameters),
            ["self", "state"],
        )
        # Resolve forward-reference / string annotations via
        # ``get_type_hints`` so the test does not depend on
        # ``from __future__ import annotations`` interaction.
        hints = typing.get_type_hints(CastingEvaluator.evaluate)
        self.assertIn("state", hints)
        self.assertIs(hints["state"], CastingEvaluationState)

    def test_evaluator_source_does_not_import_agents_or_workflows(self) -> (
        None
    ):
        import obsidianlink.evaluation.casting as casting_module

        with open(casting_module.__file__, "r", encoding="utf-8") as handle:
            source = handle.read()
        for forbidden in (
            "obsidianlink.agents",
            "obsidianlink.workflows",
            "obsidianlink.drivers",
            "from obsidianlink.core.types import Observation",
            "MacroAction",
            "VLM",
            "vlm",
            "Qwen",
        ):
            self.assertNotIn(
                forbidden,
                source,
                f"casting module must not reference {forbidden!r}",
            )


class FakeBackendCastingStateTests(unittest.TestCase):
    """FakeBackend evaluator-only casting state surface."""

    def test_set_and_get_casting_state(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(casting_c1_task())
            # ``step_id`` must match the post-reset backend step (0).
            state = _state_at_step(0)
            backend.set_casting_evaluation_state(state)
            self.assertIs(backend.get_casting_evaluation_state(), state)
        finally:
            backend.close()

    def test_get_before_reset_rejected(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            with self.assertRaisesRegex(RuntimeError, "not been reset"):
                backend.get_casting_evaluation_state()
        finally:
            backend.close()

    def test_get_before_open_rejected(self) -> None:
        backend = FakeEnvironmentBackend()
        with self.assertRaisesRegex(RuntimeError, "not open"):
            backend.get_casting_evaluation_state()

    def test_get_without_injection_rejected(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(casting_c1_task())
            with self.assertRaisesRegex(
                RuntimeError, "casting evaluation state is unavailable"
            ):
                backend.get_casting_evaluation_state()
        finally:
            backend.close()

    def test_set_rejects_wrong_episode_id(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(casting_c1_task())
            state = _success_base_state(episode_id="other_episode")
            with self.assertRaisesRegex(ValueError, "episode_id"):
                backend.set_casting_evaluation_state(state)
        finally:
            backend.close()

    def test_set_rejects_wrong_step_id(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(casting_c1_task())
            # ``step_id=1`` does not match the post-reset step (0).
            state = _state_at_step(1)
            with self.assertRaisesRegex(ValueError, "step_id"):
                backend.set_casting_evaluation_state(state)
        finally:
            backend.close()

    def test_set_rejects_non_casting_state(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(casting_c1_task())
            with self.assertRaisesRegex(TypeError, "CastingEvaluationState"):
                backend.set_casting_evaluation_state(  # type: ignore[arg-type]
                    EvaluationState(episode_id=EPISODE_ID, step_id=0)
                )
            with self.assertRaisesRegex(TypeError, "CastingEvaluationState"):
                backend.set_casting_evaluation_state(  # type: ignore[arg-type]
                    {"initial_target_block": "air"}
                )
        finally:
            backend.close()

    def test_step_clears_casting_state(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(casting_c1_task())
            backend.set_casting_evaluation_state(_state_at_step(0))
            # After step, the state is for the previous step and
            # must be cleared so a stale read fails closed.
            backend.step({"agent_1": MacroAction.wait()})
            with self.assertRaisesRegex(
                RuntimeError, "casting evaluation state is unavailable"
            ):
                backend.get_casting_evaluation_state()
        finally:
            backend.close()

    def test_close_clears_casting_state(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(casting_c1_task())
            backend.set_casting_evaluation_state(_state_at_step(0))
        finally:
            backend.close()
        # Reopen and reset; the previous state must not survive.
        backend.open()
        try:
            backend.reset(casting_c1_task())
            with self.assertRaisesRegex(
                RuntimeError, "casting evaluation state is unavailable"
            ):
                backend.get_casting_evaluation_state()
        finally:
            backend.close()

    def test_observation_does_not_leak_casting_truth(self) -> None:
        """Agent-visible observations must not expose casting truth."""
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            observations = backend.reset(casting_c1_task())
            # Inject casting truth for the reset step.
            backend.set_casting_evaluation_state(
                _state_at_step(
                    0,
                    initial_target_block="air",
                    current_target_block="obsidian",
                    target_update_evidence=CastingTransitionEvidence(
                        before_block="air",
                        after_block="obsidian",
                        update_step=0,
                    ),
                    relevant_action_steps=(0,),
                    water_truth=CastingFluidTruth(present=True, evidence_step=0),
                    lava_truth=CastingFluidTruth(present=True, evidence_step=0),
                    episode_terminated=True,
                    terminated_step=0,
                    terminated_reason="driver_done",
                )
            )
            # Run a step (the state slot is cleared because the
            # step id moved). Re-inject for the new step so we
            # also verify the *step's* observation stays clean.
            step = backend.step({"agent_1": MacroAction.wait()})
            backend.set_casting_evaluation_state(
                _state_at_step(
                    1,
                    initial_target_block="air",
                    current_target_block="obsidian",
                    target_update_evidence=CastingTransitionEvidence(
                        before_block="air",
                        after_block="obsidian",
                        update_step=1,
                    ),
                    relevant_action_steps=(0,),
                    episode_terminated=True,
                    terminated_step=1,
                    terminated_reason="driver_done",
                )
            )
            for observation in list(observations.values()) + list(
                step.observations.values()
            ):
                self._assert_observation_is_clean(observation)
        finally:
            backend.close()

    def _assert_observation_is_clean(self, observation: Observation) -> None:
        for forbidden in (
            "target_cell",
            "target_block",
            "target_block_truth",
            "initial_target_block",
            "current_target_block",
            "fluid_truth",
            "water_truth",
            "lava_truth",
            "fluid_evidence",
            "casting_outcome",
            "casting_evaluator",
            "outcome",
            "success",
            "failure_type",
            "blocking_conditions",
        ):
            self.assertFalse(
                hasattr(observation, forbidden),
                f"Observation must not expose {forbidden!r}",
            )
        frame = observation.frame
        if isinstance(frame, dict):
            for forbidden in (
                "target_cell",
                "target_block",
                "initial_target_block",
                "current_target_block",
                "fluid_truth",
                "water_truth",
                "lava_truth",
                "casting_evaluator",
                "casting_outcome",
                "success",
                "blocking_conditions",
            ):
                self.assertNotIn(
                    forbidden,
                    frame,
                    f"Observation.frame must not carry {forbidden!r}",
                )
        if observation.workflow_stage is not None:
            self.assertNotIn(
                "casting",
                observation.workflow_stage.lower().replace("casting_c1", ""),
            )


class CurrentMineRLStateTests(unittest.TestCase):
    """The real MineRL backend must still fail closed for casting-c1."""

    def test_minerl_casting_c1_capabilities_still_gap(self) -> None:
        caps = MineRLEnvironmentBackend.casting_c1_capabilities()
        self.assertIsInstance(caps, BackendCapabilities)
        for field in (
            "can_select_water_bucket",
            "can_select_lava_bucket",
            "can_use_water_bucket",
            "can_use_lava_bucket",
            "exposes_selected_item",
            "exposes_target_block_truth",
            "exposes_fluid_truth",
        ):
            self.assertFalse(
                getattr(caps, field),
                f"MineRL {field} must stay False until wired in",
            )
        # The full gate must therefore still raise.
        with self.assertRaises(CapabilityMismatchError) as ctx:
            assert_casting_c1_capabilities(caps, task_id=EPISODE_ID)
        for expected in (
            "select_water_bucket",
            "select_lava_bucket",
            "use_water_bucket",
            "use_lava_bucket",
            "selected_item",
            "target_block_truth",
            "fluid_truth",
        ):
            self.assertIn(expected, ctx.exception.missing)

    def test_minerl_casting_reset_still_rejected_before_env_creation(self) -> None:
        factory_calls: list[str] = []

        def tracking_factory(task: object) -> object:
            factory_calls.append("called")  # type: ignore[arg-type]
            raise AssertionError(
                "env_factory must not be called when the gate fails"
            )

        backend = MineRLEnvironmentBackend(
            env_factory=tracking_factory,  # type: ignore[arg-type]
            reset_warmup_steps=0,
        )
        backend.open()
        try:
            with self.assertRaises(CapabilityMismatchError):
                backend.reset(casting_c1_task())
            self.assertEqual(factory_calls, [])
            self.assertIsNone(backend._env)
        finally:
            backend.close()


class PortalEvaluatorRegressionTests(unittest.TestCase):
    """R3 must not break the existing Portal evaluator contract."""

    def test_portal_evaluator_does_not_consume_casting_state(self) -> None:
        # The casting truth injected through FakeBackend must not
        # affect the Portal evaluator's verdict. Inject a clearly
        # "successful" casting state and a clearly "failed" Portal
        # state, then assert the Portal evaluator still reports the
        # Portal verdict.
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(casting_c1_task())
            backend.set_casting_evaluation_state(_state_at_step(0))
            backend.set_evaluation_state(
                EvaluationState(
                    episode_id=EPISODE_ID,
                    step_id=0,
                    episode_terminated=True,
                    terminated_step=0,
                )
            )
            result = PortalEvaluator().evaluate(backend.get_evaluation_state())
            self.assertFalse(result.success)
            self.assertEqual(result.failure_type, "frame_never_valid")
        finally:
            backend.close()

    def test_casting_evaluator_does_not_consume_portal_state(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(casting_c1_task())
            # Inject a Portal-only state and confirm the casting
            # evaluator still requires its own truth (truth_missing).
            backend.set_evaluation_state(
                EvaluationState(
                    episode_id=EPISODE_ID,
                    step_id=0,
                    portal_built_by_episode=True,
                    valid_portal_frame=True,
                    portal_activated=True,
                    agents_in_nether=frozenset({"agent_1"}),
                    episode_terminated=True,
                    terminated_step=0,
                )
            )
            with self.assertRaises(RuntimeError):
                # Portal truth is not enough; the casting surface
                # must refuse the read.
                backend.get_casting_evaluation_state()
        finally:
            backend.close()

    def test_legacy_evaluation_state_set_get_still_works(self) -> None:
        backend = FakeEnvironmentBackend()
        backend.open()
        try:
            backend.reset(casting_c1_task())
            state = EvaluationState(
                episode_id=EPISODE_ID,
                step_id=0,
            )
            backend.set_evaluation_state(state)
            self.assertIs(backend.get_evaluation_state(), state)
        finally:
            backend.close()


class BlockIdWhitelistTests(unittest.TestCase):
    """The block id whitelist is the contract between backend and evaluator."""

    def test_block_whitelist_contains_obsidian(self) -> None:
        self.assertIn("obsidian", TARGET_BLOCK_IDS)
        self.assertIn("air", TARGET_BLOCK_IDS)
        self.assertIn("cobblestone", TARGET_BLOCK_IDS)
        self.assertIn("stone", TARGET_BLOCK_IDS)

    def test_block_whitelist_rejects_arbitrary_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "initial_target_block"):
            dataclasses.replace(
                _success_base_state(), initial_target_block="dirt"
            )


if __name__ == "__main__":
    unittest.main()

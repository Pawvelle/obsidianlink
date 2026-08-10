"""Offline tests for the R6 Casting-S-C5 Nether-entry evaluator."""

from __future__ import annotations

import ast
import dataclasses
import json
import unittest
from pathlib import Path

from obsidianlink.core.types import MacroAction, TaskInstance
from obsidianlink.env.fake import FakeEnvironmentBackend
from obsidianlink.evaluation.casting_nether_entry_evaluator import (
    C5_NETHER_ENTRY_OUTCOMES,
    CASTING_S_C5_SOURCE_DIMENSION,
    CASTING_S_C5_TARGET_DIMENSION,
    FrozenNetherEntryEvaluationState,
    FrozenNetherEntryEvaluator,
    NetherEntryEvidence,
    OUTCOME_FRAME_IDENTITY_MISMATCH,
    OUTCOME_FRAME_IDENTITY_MISSING,
    OUTCOME_IGNITION_NOT_COMPLETED,
    OUTCOME_NETHER_ENTRY_NOT_VIA_EPISODE_PORTAL,
    OUTCOME_NETHER_ENTRY_PORTAL_UNKNOWN,
    OUTCOME_NO_AGENT_ENTERED_NETHER,
    OUTCOME_PRE_TRANSITION_POSITION_MISSING,
    OUTCOME_SUCCESS,
    OUTCOME_TRANSITION_BEFORE_ACTIVATION,
    OUTCOME_TRANSITION_STEP_MISSING,
    OUTCOME_WRONG_ENTRY_AGENT,
    OUTCOME_WRONG_SOURCE_DIMENSION,
    OUTCOME_WRONG_TARGET_DIMENSION,
)
from tests.test_r6_casting_c4_ignition_evaluator import (
    AGENT_ID,
    EPISODE_ID,
    _build_state_at_step_zero_for_fakebackend,
    _identity_with,
    _state as _c4_state,
)


MAX_STEPS = 800
MAX_TIME = 720
ROOT = Path(__file__).resolve().parents[1]


def _ignition_state():
    return _c4_state(
        max_environment_steps=MAX_STEPS,
        max_game_time_seconds=MAX_TIME,
    )


def _entry(ignition=None, **changes):
    if ignition is None:
        ignition = _ignition_state()
    values = {
        "episode_id": EPISODE_ID,
        "agent_id": AGENT_ID,
        "source_dimension": CASTING_S_C5_SOURCE_DIMENSION,
        "target_dimension": CASTING_S_C5_TARGET_DIMENSION,
        "transition_step": ignition.step_id,
        "pre_transition_position": (1.5, 1.0, 1.0),
        "entered_via_episode_portal": True,
        "matched_frame_identity": ignition.latched_frame_identity,
    }
    values.update(changes)
    return NetherEntryEvidence(**values)


def _state(*, ignition=None, agents=None, entry_marker=object(), **changes):
    if ignition is None:
        ignition = _ignition_state()
    if agents is None:
        agents = frozenset({AGENT_ID})
    if entry_marker.__class__ is object:
        entry = _entry(ignition)
    else:
        entry = entry_marker
    values = {
        "episode_id": EPISODE_ID,
        "step_id": ignition.step_id,
        "ignition_state": ignition,
        "agents_in_nether": agents,
        "entry_evidence": entry,
        "agent_id": AGENT_ID,
        "episode_terminated": True,
        "terminated_step": ignition.step_id,
        "terminated_reason": "goal_reached",
        "current_time_seconds": 0.0,
        "max_environment_steps": MAX_STEPS,
        "max_game_time_seconds": MAX_TIME,
    }
    # The embedded C4 state must share outer termination metadata.
    if (
        ignition.terminated_reason != values["terminated_reason"]
        or ignition.terminated_step != values["terminated_step"]
    ):
        frame = dataclasses.replace(
            ignition.frame_state,
            terminated_reason=values["terminated_reason"],
            terminated_step=values["terminated_step"],
        )
        ignition = dataclasses.replace(
            ignition,
            frame_state=frame,
            terminated_reason=values["terminated_reason"],
            terminated_step=values["terminated_step"],
        )
        values["ignition_state"] = ignition
        if entry_marker.__class__ is object:
            values["entry_evidence"] = _entry(ignition)
    values.update(changes)
    return FrozenNetherEntryEvaluationState(**values)


class NetherEntryEvaluatorTests(unittest.TestCase):
    def test_success_is_deterministic_and_json_serializable(self) -> None:
        state = _state()
        first = FrozenNetherEntryEvaluator().evaluate(state)
        second = FrozenNetherEntryEvaluator().evaluate(state)
        self.assertEqual(first, second)
        self.assertTrue(first.success)
        self.assertEqual(first.outcome, OUTCOME_SUCCESS)
        self.assertTrue(first.frame_identity_matched)
        json.dumps(first.as_dict())

    def test_contract_outcomes_are_closed(self) -> None:
        self.assertEqual(len(C5_NETHER_ENTRY_OUTCOMES), 17)

    def test_required_failure_paths(self) -> None:
        ignition = _state().ignition_state
        cases = (
            (_state(agents=frozenset()), OUTCOME_NO_AGENT_ENTERED_NETHER),
            (_state(entry_marker=None), OUTCOME_NETHER_ENTRY_PORTAL_UNKNOWN),
            (
                _state(entry_marker=_entry(ignition, agent_id="agent_2")),
                OUTCOME_WRONG_ENTRY_AGENT,
            ),
            (
                _state(entry_marker=_entry(ignition, source_dimension="minecraft:the_end")),
                OUTCOME_WRONG_SOURCE_DIMENSION,
            ),
            (
                _state(entry_marker=_entry(ignition, target_dimension="minecraft:overworld")),
                OUTCOME_WRONG_TARGET_DIMENSION,
            ),
            (
                _state(entry_marker=_entry(ignition, transition_step=None)),
                OUTCOME_TRANSITION_STEP_MISSING,
            ),
            (
                _state(entry_marker=_entry(ignition, pre_transition_position=None)),
                OUTCOME_PRE_TRANSITION_POSITION_MISSING,
            ),
            (
                _state(entry_marker=_entry(ignition, transition_step=601)),
                OUTCOME_TRANSITION_BEFORE_ACTIVATION,
            ),
            (
                _state(entry_marker=_entry(ignition, entered_via_episode_portal=None)),
                OUTCOME_NETHER_ENTRY_PORTAL_UNKNOWN,
            ),
            (
                _state(entry_marker=_entry(ignition, entered_via_episode_portal=False)),
                OUTCOME_NETHER_ENTRY_NOT_VIA_EPISODE_PORTAL,
            ),
            (
                _state(entry_marker=_entry(ignition, matched_frame_identity=None)),
                OUTCOME_FRAME_IDENTITY_MISSING,
            ),
            (
                _state(
                    entry_marker=_entry(
                        ignition,
                        matched_frame_identity=_identity_with(width=5),
                    )
                ),
                OUTCOME_FRAME_IDENTITY_MISMATCH,
            ),
        )
        for state, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    FrozenNetherEntryEvaluator().evaluate(state).outcome,
                    expected,
                )

    def test_c4_failure_cannot_be_overridden_by_entry_claim(self) -> None:
        ignition = _c4_state(
            ignition_action=None,
            max_environment_steps=MAX_STEPS,
            max_game_time_seconds=MAX_TIME,
        )
        result = FrozenNetherEntryEvaluator().evaluate(_state(ignition=ignition))
        self.assertEqual(result.outcome, OUTCOME_IGNITION_NOT_COMPLETED)
        self.assertFalse(result.success)

    def test_structural_validation_is_strict(self) -> None:
        ignition = _state().ignition_state
        with self.assertRaises(ValueError):
            _entry(ignition, transition_step=True)
        with self.assertRaises(ValueError):
            _entry(ignition, entered_via_episode_portal="yes")
        with self.assertRaises(ValueError):
            _entry(ignition, pre_transition_position=(1, float("nan"), 1))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            _entry(ignition).agent_id = "agent_2"  # type: ignore[misc]


class FakeBackendC5SlotTests(unittest.TestCase):
    def _task(self) -> TaskInstance:
        return TaskInstance.from_dict(
            {
                "schema_version": "0.1",
                "task_id": EPISODE_ID,
                "route": "lava_casting",
                "difficulty": 4,
                "agent_ids": [AGENT_ID],
                "world_seed": 0,
                "instruction": "C5 evaluator slot test",
                "spawn_positions": {AGENT_ID: [0, 4, 0]},
                "initial_inventories": {
                    AGENT_ID: {
                        "water_bucket": 14,
                        "lava_bucket": 14,
                        "cobblestone": 28,
                        "flint_and_steel": 1,
                    }
                },
                "workflow": "casting_s_c5_fixed",
                "milestones": ["task_reset", "agent_entered_nether"],
                "limits": {
                    "max_environment_steps": MAX_STEPS,
                    "max_model_calls": 1,
                    "max_game_time_seconds": MAX_TIME,
                },
                "split": "development",
            }
        )

    def test_slot_is_guarded_cleared_and_not_in_observation(self) -> None:
        ignition = _build_state_at_step_zero_for_fakebackend()
        frame = dataclasses.replace(
            ignition.frame_state,
            max_environment_steps=MAX_STEPS,
            max_game_time_seconds=MAX_TIME,
        )
        ignition = dataclasses.replace(
            ignition,
            frame_state=frame,
            max_environment_steps=MAX_STEPS,
            max_game_time_seconds=MAX_TIME,
        )
        state = FrozenNetherEntryEvaluationState(
            episode_id=EPISODE_ID,
            step_id=0,
            ignition_state=ignition,
            max_environment_steps=MAX_STEPS,
            max_game_time_seconds=MAX_TIME,
        )
        backend = FakeEnvironmentBackend()
        backend.open()
        observation = backend.reset(self._task())[AGENT_ID]
        self.assertNotIn("agents_in_nether", repr(observation))
        with self.assertRaises(RuntimeError):
            backend.get_nether_entry_evaluation_state()
        backend.set_nether_entry_evaluation_state(state)
        self.assertIs(backend.get_nether_entry_evaluation_state(), state)
        backend.clear_nether_entry_evaluation_state()
        with self.assertRaises(RuntimeError):
            backend.get_nether_entry_evaluation_state()
        backend.close()

    def test_step_and_reset_clear_stale_c5_truth(self) -> None:
        ignition = _build_state_at_step_zero_for_fakebackend()
        frame = dataclasses.replace(
            ignition.frame_state,
            max_environment_steps=MAX_STEPS,
            max_game_time_seconds=MAX_TIME,
        )
        ignition = dataclasses.replace(
            ignition,
            frame_state=frame,
            max_environment_steps=MAX_STEPS,
            max_game_time_seconds=MAX_TIME,
        )
        state = FrozenNetherEntryEvaluationState(
            episode_id=EPISODE_ID,
            step_id=0,
            ignition_state=ignition,
            max_environment_steps=MAX_STEPS,
            max_game_time_seconds=MAX_TIME,
        )
        backend = FakeEnvironmentBackend()
        backend.open()
        backend.reset(self._task())
        backend.set_nether_entry_evaluation_state(state)
        backend.step({AGENT_ID: MacroAction("wait", duration_ticks=1)})
        with self.assertRaises(RuntimeError):
            backend.get_nether_entry_evaluation_state()
        backend.reset(self._task())
        with self.assertRaises(RuntimeError):
            backend.get_nether_entry_evaluation_state()
        backend.close()

    def test_wrong_workflow_and_wrong_type_fail_closed(self) -> None:
        c4_task = dataclasses.replace(self._task(), workflow="casting_s_c4_fixed")
        backend = FakeEnvironmentBackend()
        backend.open()
        backend.reset(c4_task)
        with self.assertRaises(TypeError):
            backend.set_nether_entry_evaluation_state(object())  # type: ignore[arg-type]
        ignition = _build_state_at_step_zero_for_fakebackend()
        frame = dataclasses.replace(
            ignition.frame_state,
            max_environment_steps=MAX_STEPS,
            max_game_time_seconds=MAX_TIME,
        )
        ignition = dataclasses.replace(
            ignition,
            frame_state=frame,
            max_environment_steps=MAX_STEPS,
            max_game_time_seconds=MAX_TIME,
        )
        state = FrozenNetherEntryEvaluationState(
            episode_id=EPISODE_ID,
            step_id=0,
            ignition_state=ignition,
            max_environment_steps=MAX_STEPS,
            max_game_time_seconds=MAX_TIME,
        )
        with self.assertRaises(ValueError):
            backend.set_nether_entry_evaluation_state(state)
        backend.close()


class InformationIsolationTests(unittest.TestCase):
    def test_evaluator_does_not_import_agent_or_driver_surfaces(self) -> None:
        path = ROOT / "obsidianlink/evaluation/casting_nether_entry_evaluator.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        forbidden = ("obsidianlink.agents", "obsidianlink.drivers", "obsidianlink.workflows")
        self.assertFalse(any(name.startswith(forbidden) for name in imports))


if __name__ == "__main__":
    unittest.main()

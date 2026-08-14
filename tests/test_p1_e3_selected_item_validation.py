from __future__ import annotations

import json
import unittest
from typing import Any

from obsidianlink.env.validation import (
    E0_LIFECYCLE_CASE,
    E1_RGB_CASE,
    E2_INVENTORY_CASE,
    E3_SELECTED_ITEM_CASE,
    EnvironmentValidationResult,
    EnvironmentValidationRunner,
)
from obsidianlink.env.validation.contract import EnvironmentValidationId
from obsidianlink.env.validation.result import UNIT_VERIFIED


EPISODE_ID = "e3-runner-episode"
EXPECTED = "flint_and_steel"


class _Stub:
    def __init__(self, reset_result: object, *, reset_error=None, close_error=None) -> None:
        self.reset_result = reset_result
        self.reset_error = reset_error
        self.close_error = close_error
        self.reset_calls = 0
        self.close_calls = 0
        self.step_calls = 0

    def reset(self) -> object:
        self.reset_calls += 1
        if self.reset_error:
            raise self.reset_error
        return self.reset_result

    def step(self, action: object) -> None:
        self.step_calls += 1
        raise AssertionError("E3 must not step")

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error:
            raise self.close_error


def _state(item: object, **extra: object) -> dict[str, dict[str, object]]:
    payload = {
        "agent_id": "agent_1",
        "episode_id": EPISODE_ID,
        "selected_item": item,
        "step_id": 0,
    }
    payload.update(extra)
    return {"agent_1": payload}


def _run(item: object, *, expected: object = EXPECTED, reset_error=None, close_error=None, **extra: object):
    stub = _Stub(_state(item, **extra), reset_error=reset_error, close_error=close_error)
    result = EnvironmentValidationRunner().run(
        E3_SELECTED_ITEM_CASE,
        lambda: stub,
        episode_id=EPISODE_ID,
        expected_selected_item=expected,  # type: ignore[arg-type]
    )
    return result, stub


class E3SelectedItemRunnerTests(unittest.TestCase):
    def test_exact_match_succeeds_offline_only(self) -> None:
        result, stub = _run(EXPECTED)
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, "selected_item_ok")
        self.assertTrue(result.selected_item_present)
        self.assertEqual(result.observed_selected_item, EXPECTED)
        self.assertEqual(result.expected_selected_item, EXPECTED)
        self.assertTrue(result.selected_item_matches_expected)
        self.assertEqual(result.verification_level, UNIT_VERIFIED)
        self.assertFalse(result.real_execution_performed)
        self.assertFalse(result.integration_verified)
        self.assertEqual((stub.reset_calls, stub.close_calls, stub.step_calls), (1, 1, 0))

    def test_wrong_backend_item_remains_mismatch(self) -> None:
        result, _ = _run("obsidian")
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "selected_item_mismatch")
        self.assertEqual(result.observed_selected_item, "obsidian")
        self.assertEqual(result.expected_selected_item, EXPECTED)
        self.assertFalse(result.selected_item_matches_expected)

    def test_none_missing_malformed_and_leak_fail_closed(self) -> None:
        cases = (
            (None, {}, "selected_item_none"),
            (4, {}, "selected_item_type_invalid"),
            ("", {}, "selected_item_empty"),
            (EXPECTED, {"inventory": {EXPECTED: 1}}, "selected_item_leak"),
        )
        for item, extra, outcome in cases:
            with self.subTest(outcome=outcome):
                result, _ = _run(item, **extra)
                self.assertFalse(result.success)
                self.assertEqual(result.outcome, outcome)

        stub = _Stub({"agent_1": {"agent_id": "agent_1", "episode_id": EPISODE_ID, "step_id": 0}})
        result = EnvironmentValidationRunner().run(
            E3_SELECTED_ITEM_CASE, lambda: stub, episode_id=EPISODE_ID,
            expected_selected_item=EXPECTED,
        )
        self.assertEqual(result.outcome, "selected_item_missing")

    def test_invalid_expected_fails_before_backend_creation(self) -> None:
        for expected in (None, "", 4):
            result, stub = _run(EXPECTED, expected=expected)
            self.assertFalse(result.created)
            self.assertEqual(result.outcome, "runtime_error")
            self.assertEqual(stub.reset_calls, 0)

    def test_reset_close_and_initial_state_failures(self) -> None:
        result, stub = _run(EXPECTED, reset_error=RuntimeError("boom"))
        self.assertEqual(result.outcome, "reset_failed")
        self.assertTrue(result.closed)
        self.assertEqual(stub.close_calls, 1)
        result, _ = _run(EXPECTED, close_error=OSError("close boom"))
        self.assertEqual(result.outcome, "close_failed")
        self.assertFalse(result.success)
        empty = _Stub({})
        result = EnvironmentValidationRunner().run(
            E3_SELECTED_ITEM_CASE, lambda: empty, episode_id=EPISODE_ID,
            expected_selected_item=EXPECTED,
        )
        self.assertEqual(result.outcome, "initial_state_missing")

    def test_result_serialization_is_deterministic_and_e3_only(self) -> None:
        result, _ = _run(EXPECTED)
        payload = result.as_dict()
        self.assertEqual(payload, json.loads(json.dumps(payload)))
        self.assertEqual(payload["observed_selected_item"], EXPECTED)
        self.assertNotIn("observed_inventory", payload)
        self.assertNotIn("rgb_present", payload)

    def test_earlier_case_serialization_has_no_e3_metadata(self) -> None:
        keys = {
            "selected_item_present", "observed_selected_item",
            "expected_selected_item", "selected_item_matches_expected",
        }
        for case, kwargs, state in (
            (E0_LIFECYCLE_CASE, {}, {"agent_1": {"episode_id": EPISODE_ID, "step_id": 0}}),
            (E2_INVENTORY_CASE, {"expected_inventory": {"dirt": 1}}, {"agent_1": {"agent_id": "agent_1", "episode_id": EPISODE_ID, "step_id": 0, "inventory": {"dirt": 1}}}),
        ):
            stub = _Stub(state)
            result = EnvironmentValidationRunner().run(case, lambda: stub, episode_id=EPISODE_ID, **kwargs)
            self.assertTrue(keys.isdisjoint(result.as_dict()))

    def test_result_rejects_false_success_and_claim_promotion(self) -> None:
        kwargs: dict[str, Any] = dict(
            check_id=EnvironmentValidationId.E3,
            name="selected_item",
            episode_id=EPISODE_ID,
            step_id=0,
            success=True,
            outcome="selected_item_ok",
            created=True,
            reset_completed=True,
            initial_state_present=True,
            closed=True,
            selected_item_present=True,
            observed_selected_item=EXPECTED,
            expected_selected_item=EXPECTED,
            selected_item_matches_expected=True,
        )
        with self.assertRaisesRegex(ValueError, "real execution"):
            EnvironmentValidationResult(**kwargs, real_execution_performed=True)
        with self.assertRaisesRegex(ValueError, "integration_verified"):
            EnvironmentValidationResult(**kwargs, integration_verified=True)
        with self.assertRaisesRegex(ValueError, "contradicts"):
            EnvironmentValidationResult(
                **{**kwargs, "success": False, "outcome": "selected_item_mismatch", "observed_selected_item": "dirt", "selected_item_matches_expected": True}
            )


if __name__ == "__main__":
    unittest.main()

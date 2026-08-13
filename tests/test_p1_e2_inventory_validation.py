from __future__ import annotations

import json
import unittest
from typing import Any, Mapping

import numpy as np

from obsidianlink.env.validation import (
    E0_LIFECYCLE_CASE,
    E1_RGB_CASE,
    E2_INVENTORY_CASE,
    EnvironmentValidationResult,
    EnvironmentValidationRunner,
    p1_validation_manifest,
)
from obsidianlink.env.validation.contract import EnvironmentValidationId
from obsidianlink.env.validation.result import UNIT_VERIFIED


EPISODE_ID = "e2-runner-offline-episode"
EXPECTED = {"dirt": 7, "obsidian": 4, "flint_and_steel": 1}


def _state(
    inventory: object,
    **extra: object,
) -> dict[str, dict[str, object]]:
    payload: dict[str, object] = {
        "agent_id": "agent_1",
        "episode_id": EPISODE_ID,
        "inventory": inventory,
        "step_id": 0,
    }
    payload.update(extra)
    return {"agent_1": payload}


class _InventoryStub:
    def __init__(
        self,
        reset_result: object,
        *,
        reset_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.reset_calls = 0
        self.close_calls = 0
        self.step_calls = 0
        self._reset_result = reset_result
        self._reset_error = reset_error
        self._close_error = close_error

    def reset(self) -> object:
        self.reset_calls += 1
        if self._reset_error is not None:
            raise self._reset_error
        return self._reset_result

    def step(self, action: object) -> None:
        self.step_calls += 1
        raise AssertionError("E2 must not execute actions")

    def close(self) -> None:
        self.close_calls += 1
        if self._close_error is not None:
            raise self._close_error


def _run(
    observed: object,
    *,
    expected: object = EXPECTED,
    extra: Mapping[str, object] | None = None,
    reset_error: Exception | None = None,
    close_error: Exception | None = None,
) -> tuple[EnvironmentValidationResult, _InventoryStub]:
    stub = _InventoryStub(
        _state(observed, **dict(extra or {})),
        reset_error=reset_error,
        close_error=close_error,
    )
    result = EnvironmentValidationRunner().run(
        E2_INVENTORY_CASE,
        lambda: stub,
        episode_id=EPISODE_ID,
        expected_inventory=expected,  # type: ignore[arg-type]
    )
    return result, stub


class E2InventoryValidationTests(unittest.TestCase):
    def test_exact_multi_item_match_succeeds_offline_only(self) -> None:
        result, stub = _run(dict(EXPECTED))
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, "inventory_ok")
        self.assertTrue(result.inventory_present)
        self.assertEqual(result.observed_inventory, EXPECTED)
        self.assertEqual(result.expected_inventory, EXPECTED)
        self.assertTrue(result.inventory_matches_expected)
        self.assertEqual(result.check_id, EnvironmentValidationId.E2)
        self.assertEqual(result.verification_level, UNIT_VERIFIED)
        self.assertFalse(result.real_execution_performed)
        self.assertFalse(result.integration_verified)
        self.assertTrue(result.calibration_only)
        self.assertEqual(stub.reset_calls, 1)
        self.assertEqual(stub.close_calls, 1)
        self.assertEqual(stub.step_calls, 0)

    def test_quantity_difference_is_mismatch(self) -> None:
        observed = dict(EXPECTED)
        observed["obsidian"] = 3
        result, _ = _run(observed)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "inventory_mismatch")
        self.assertFalse(result.inventory_matches_expected)

    def test_missing_expected_item_is_mismatch(self) -> None:
        observed = dict(EXPECTED)
        del observed["obsidian"]
        result, _ = _run(observed)
        self.assertEqual(result.outcome, "inventory_mismatch")
        self.assertFalse(result.success)

    def test_extra_observed_item_is_mismatch(self) -> None:
        observed = {**EXPECTED, "cobblestone": 1}
        result, _ = _run(observed)
        self.assertEqual(result.outcome, "inventory_mismatch")
        self.assertFalse(result.success)

    def test_missing_expected_inventory_fails_before_backend_creation(self) -> None:
        result, stub = _run(dict(EXPECTED), expected=None)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "runtime_error")
        self.assertIn("expected_inventory", result.error or "")
        self.assertFalse(result.created)
        self.assertEqual(stub.reset_calls, 0)
        self.assertEqual(stub.close_calls, 0)

    def test_empty_expected_inventory_fails_before_backend_creation(self) -> None:
        result, stub = _run({}, expected={})
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "runtime_error")
        self.assertIn("non-empty", result.error or "")
        self.assertFalse(result.created)
        self.assertEqual(stub.reset_calls, 0)
        self.assertEqual(stub.close_calls, 0)

    def test_invalid_expected_item_name_fails_closed(self) -> None:
        for item in ("", "   ", 1):
            with self.subTest(item=repr(item)):
                result, _ = _run(dict(EXPECTED), expected={item: 1})
                self.assertFalse(result.success)
                self.assertEqual(result.outcome, "runtime_error")

    def test_invalid_expected_quantities_fail_without_coercion(self) -> None:
        for quantity in ("4", 4.0, True, 0, -1):
            with self.subTest(quantity=repr(quantity)):
                result, stub = _run(
                    dict(EXPECTED), expected={"obsidian": quantity}
                )
                self.assertFalse(result.success)
                self.assertEqual(result.outcome, "runtime_error")
                self.assertEqual(stub.reset_calls, 0)

    def test_public_inventory_structure_outcomes_are_preserved(self) -> None:
        cases = (
            (None, "inventory_none"),
            (["obsidian"], "inventory_type_invalid"),
            ({"": 1}, "inventory_item_invalid"),
            ({"obsidian": "4"}, "inventory_quantity_invalid"),
        )
        for observed, outcome in cases:
            with self.subTest(outcome=outcome):
                result, _ = _run(observed)
                self.assertFalse(result.success)
                self.assertEqual(result.outcome, outcome)
                self.assertIsNone(result.inventory_matches_expected)

    def test_empty_observed_inventory_is_a_mismatch_not_structure_error(self) -> None:
        result, _ = _run({}, expected={"obsidian": 4})
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "inventory_mismatch")
        self.assertEqual(result.observed_inventory, {})
        self.assertEqual(result.expected_inventory, {"obsidian": 4})
        self.assertFalse(result.inventory_matches_expected)

    def test_missing_public_inventory_is_preserved(self) -> None:
        stub = _InventoryStub(
            {
                "agent_1": {
                    "agent_id": "agent_1",
                    "episode_id": EPISODE_ID,
                    "step_id": 0,
                }
            }
        )
        result = EnvironmentValidationRunner().run(
            E2_INVENTORY_CASE,
            lambda: stub,
            episode_id=EPISODE_ID,
            expected_inventory=EXPECTED,
        )
        self.assertEqual(result.outcome, "inventory_missing")
        self.assertFalse(result.success)

    def test_public_leak_outcomes_are_preserved(self) -> None:
        leaks = {
            "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
            "selected_item": "obsidian",
            "portal_grid": "evaluator-only",
        }
        for field, value in leaks.items():
            with self.subTest(field=field):
                result, _ = _run(EXPECTED, extra={field: value})
                self.assertFalse(result.success)
                self.assertEqual(result.outcome, "inventory_leak")

    def test_reset_exception_fails_closed_and_closes(self) -> None:
        result, stub = _run(EXPECTED, reset_error=RuntimeError("reset exploded"))
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "reset_failed")
        self.assertTrue(result.closed)
        self.assertEqual(stub.close_calls, 1)

    def test_close_exception_invalidates_an_exact_match(self) -> None:
        result, stub = _run(EXPECTED, close_error=OSError("close exploded"))
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "close_failed")
        self.assertFalse(result.closed)
        self.assertTrue(result.inventory_matches_expected)
        self.assertEqual(result.observed_inventory, EXPECTED)
        self.assertEqual(stub.close_calls, 1)

    def test_expected_and_observed_inventory_are_detached(self) -> None:
        expected = dict(EXPECTED)
        observed = dict(EXPECTED)
        result, _ = _run(observed, expected=expected)
        expected["obsidian"] = 99
        observed["obsidian"] = 88
        self.assertEqual(result.expected_inventory, EXPECTED)
        self.assertEqual(result.observed_inventory, EXPECTED)

        first = result.as_dict()
        first_expected = first["expected_inventory"]
        first_observed = first["observed_inventory"]
        assert isinstance(first_expected, dict)
        assert isinstance(first_observed, dict)
        first_expected["obsidian"] = 77
        first_observed["obsidian"] = 66
        second = result.as_dict()
        self.assertEqual(second["expected_inventory"], EXPECTED)
        self.assertEqual(second["observed_inventory"], EXPECTED)

    def test_serialized_e2_result_has_inventory_metadata(self) -> None:
        result, _ = _run(EXPECTED)
        payload = result.as_dict()
        self.assertEqual(payload["expected_inventory"], EXPECTED)
        self.assertEqual(payload["observed_inventory"], EXPECTED)
        self.assertTrue(payload["inventory_present"])
        self.assertTrue(payload["inventory_matches_expected"])
        self.assertEqual(payload, json.loads(json.dumps(payload)))

    def test_e0_and_e1_serialization_have_no_inventory_payload(self) -> None:
        e0_stub = _InventoryStub(
            {"agent_1": {"episode_id": EPISODE_ID, "step_id": 0}}
        )
        e0 = EnvironmentValidationRunner().run(
            E0_LIFECYCLE_CASE,
            lambda: e0_stub,
            episode_id=EPISODE_ID,
        )
        self.assertTrue(e0.success)

        e1_stub = _InventoryStub(
            {
                "agent_1": {
                    "agent_id": "agent_1",
                    "episode_id": EPISODE_ID,
                    "rgb": np.zeros((2, 3, 3), dtype=np.uint8),
                    "step_id": 0,
                }
            }
        )
        e1 = EnvironmentValidationRunner().run(
            E1_RGB_CASE,
            lambda: e1_stub,
            episode_id=EPISODE_ID,
        )
        self.assertTrue(e1.success)

        inventory_keys = {
            "expected_inventory",
            "inventory_matches_expected",
            "inventory_present",
            "observed_inventory",
        }
        self.assertTrue(inventory_keys.isdisjoint(e0.as_dict()))
        self.assertTrue(inventory_keys.isdisjoint(e1.as_dict()))

    def test_result_rejects_e2_real_or_integration_claims(self) -> None:
        kwargs: dict[str, Any] = dict(
            check_id=EnvironmentValidationId.E2,
            name="inventory_observation",
            episode_id=EPISODE_ID,
            step_id=0,
            success=True,
            outcome="inventory_ok",
            created=True,
            reset_completed=True,
            initial_state_present=True,
            closed=True,
            inventory_present=True,
            observed_inventory=EXPECTED,
            expected_inventory=EXPECTED,
            inventory_matches_expected=True,
        )
        with self.assertRaisesRegex(ValueError, "real execution"):
            EnvironmentValidationResult(**kwargs, real_execution_performed=True)
        with self.assertRaisesRegex(ValueError, "integration_verified"):
            EnvironmentValidationResult(**kwargs, integration_verified=True)

    def test_result_rejects_false_e2_success_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact observed/expected"):
            EnvironmentValidationResult(
                check_id=EnvironmentValidationId.E2,
                name="inventory_observation",
                episode_id=EPISODE_ID,
                step_id=0,
                success=True,
                outcome="inventory_ok",
                created=True,
                reset_completed=True,
                initial_state_present=True,
                closed=True,
                inventory_present=True,
                observed_inventory={"obsidian": 3},
                expected_inventory={"obsidian": 4},
                inventory_matches_expected=False,
            )

    def test_result_rejects_empty_expected_inventory_success(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty expected"):
            EnvironmentValidationResult(
                check_id=EnvironmentValidationId.E2,
                name="inventory_observation",
                episode_id=EPISODE_ID,
                step_id=0,
                success=True,
                outcome="inventory_ok",
                created=True,
                reset_completed=True,
                initial_state_present=True,
                closed=True,
                inventory_present=True,
                observed_inventory={},
                expected_inventory={},
                inventory_matches_expected=True,
            )

    def test_e2_case_and_manifest_remain_calibration_only_not_run(self) -> None:
        self.assertIs(E2_INVENTORY_CASE.check_id, EnvironmentValidationId.E2)
        self.assertEqual(E2_INVENTORY_CASE.name, "inventory_observation")
        self.assertFalse(E2_INVENTORY_CASE.requires_server_truth)
        self.assertTrue(E2_INVENTORY_CASE.calibration_only)
        e2_manifest = p1_validation_manifest()[2]
        self.assertEqual(e2_manifest["status"], "not_run")
        self.assertFalse(e2_manifest["requires_server_truth"])


if __name__ == "__main__":
    unittest.main()

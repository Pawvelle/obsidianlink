from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from obsidianlink.env.validation import (
    E1_RGB_CASE,
    EnvironmentValidationRecorder,
    EnvironmentValidationResult,
    EnvironmentValidationRunner,
    P1_VALIDATION_CASES,
    PublicRGBObservation,
    inspect_public_rgb,
    inspect_rgb_array,
    p1_validation_manifest,
)
from obsidianlink.env.validation.contract import EnvironmentValidationId
from obsidianlink.env.validation.result import UNIT_VERIFIED


EPISODE_ID = "e1-offline-episode"


class _RGBStub:
    def __init__(self, reset_result: object) -> None:
        self.reset_calls = 0
        self.close_calls = 0
        self._reset_result = reset_result

    def reset(self) -> object:
        self.reset_calls += 1
        return self._reset_result

    def close(self) -> None:
        self.close_calls += 1


def _valid_rgb(*, height: int = 360, width: int = 640) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


_RGB_UNSET = object()


def _public_state(
    rgb: object = _RGB_UNSET, **extra: object
) -> dict[str, dict[str, object]]:
    payload: dict[str, object] = {
        "agent_id": "agent_1",
        "episode_id": EPISODE_ID,
        "step_id": 0,
    }
    if rgb is not _RGB_UNSET:
        payload["rgb"] = rgb
    payload.update(extra)
    return {"agent_1": payload}


def _run(reset_result: object) -> EnvironmentValidationResult:
    stub = _RGBStub(reset_result)
    return EnvironmentValidationRunner().run(
        E1_RGB_CASE,
        lambda: stub,
        episode_id=EPISODE_ID,
    )


class E1RGBContractTests(unittest.TestCase):
    def test_valid_hxw3_uint8_succeeds(self) -> None:
        rgb = _valid_rgb()
        inspection = inspect_rgb_array(rgb)
        self.assertTrue(inspection.valid)
        self.assertEqual(inspection.outcome, "rgb_ok")
        self.assertEqual(inspection.height, 360)
        self.assertEqual(inspection.width, 640)
        self.assertEqual(inspection.channels, 3)
        self.assertEqual(inspection.dtype, "uint8")
        result = _run(_public_state(rgb))
        self.assertTrue(result.success)
        self.assertEqual(result.outcome, "rgb_ok")
        self.assertTrue(result.rgb_present)
        self.assertEqual(result.rgb_height, 360)
        self.assertEqual(result.rgb_width, 640)
        self.assertEqual(result.rgb_channels, 3)
        self.assertEqual(result.rgb_dtype, "uint8")
        self.assertEqual(result.check_id, EnvironmentValidationId.E1)
        self.assertEqual(result.verification_level, UNIT_VERIFIED)
        self.assertFalse(result.real_execution_performed)
        self.assertFalse(result.integration_verified)

    def test_missing_rgb_fails_closed(self) -> None:
        result = _run(_public_state())
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "rgb_missing")
        self.assertTrue(result.closed)
        self.assertFalse(result.rgb_present)
        inspection = inspect_public_rgb(_public_state(), episode_id=EPISODE_ID)
        self.assertEqual(inspection.outcome, "rgb_missing")

    def test_none_rgb_fails_closed(self) -> None:
        result = _run(_public_state(rgb=None))
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "rgb_none")
        self.assertTrue(result.closed)
        self.assertFalse(result.rgb_present)

    def test_wrong_shape_fails_closed(self) -> None:
        cases = (
            np.zeros((360, 640), dtype=np.uint8),
            np.zeros((360, 640, 1), dtype=np.uint8),
            np.zeros((0, 640, 3), dtype=np.uint8),
            np.zeros((360, 0, 3), dtype=np.uint8),
        )
        for rgb in cases:
            with self.subTest(shape=rgb.shape):
                result = _run(_public_state(rgb))
                self.assertFalse(result.success)
                self.assertEqual(result.outcome, "rgb_shape_invalid")
                self.assertTrue(result.closed)

    def test_illegal_dtype_fails_closed(self) -> None:
        cases = (
            np.zeros((16, 16, 3), dtype=np.float32),
            np.zeros((16, 16, 3), dtype=np.uint16),
            np.zeros((16, 16, 3), dtype=np.int32),
        )
        for rgb in cases:
            with self.subTest(dtype=rgb.dtype):
                result = _run(_public_state(rgb))
                self.assertFalse(result.success)
                self.assertEqual(result.outcome, "rgb_dtype_invalid")
                self.assertTrue(result.closed)

    def test_public_payload_rejects_inventory_selected_item_and_truth(self) -> None:
        rgb = _valid_rgb(height=8, width=8)
        leaked = _public_state(
            rgb,
            visible_inventory={"dirt": 1},
            selected_item="dirt",
            portal_grid="evaluator-only",
        )
        inspection = inspect_public_rgb(leaked, episode_id=EPISODE_ID)
        self.assertEqual(inspection.outcome, "rgb_leak")
        result = _run(leaked)
        self.assertFalse(result.success)
        self.assertEqual(result.outcome, "rgb_leak")
        snapshot = json.dumps(result.as_dict())
        self.assertNotIn("evaluator-only", snapshot)
        self.assertNotIn('"dirt"', snapshot)

    def test_public_rgb_observation_type_is_not_legacy_observation(self) -> None:
        observation = PublicRGBObservation(
            episode_id=EPISODE_ID,
            agent_id="agent_1",
            step_id=0,
            rgb=_valid_rgb(height=4, width=4),
        )
        self.assertEqual(set(observation.as_public_dict()), {"agent_id", "episode_id", "rgb", "step_id"})
        self.assertFalse(hasattr(observation, "visible_inventory"))
        self.assertFalse(hasattr(observation, "selected_item"))
        self.assertFalse(hasattr(observation, "workflow_stage"))
        result = _run({"agent_1": observation})
        self.assertTrue(result.success)

    def test_offline_e1_success_never_marks_integration_verified(self) -> None:
        result = _run(_public_state(_valid_rgb(height=4, width=6)))
        self.assertTrue(result.success)
        self.assertFalse(result.integration_verified)
        self.assertFalse(result.real_execution_performed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "e1" / "validation_result.json"
            payload = json.loads(
                EnvironmentValidationRecorder().record(result, path).read_text(
                    encoding="utf-8"
                )
            )
        self.assertFalse(payload["integration_verified"])
        self.assertFalse(payload["real_execution_performed"])
        self.assertEqual(payload["verification_level"], UNIT_VERIFIED)
        self.assertNotIn("pixels", json.dumps(payload))
        manifest = p1_validation_manifest()
        self.assertTrue(all(item["status"] == "not_run" for item in manifest))
        self.assertEqual(manifest[1]["check_id"], "E1")
        self.assertEqual(manifest[1]["name"], "rgb_observation")

    def test_e1_case_matches_manifest(self) -> None:
        self.assertIs(E1_RGB_CASE.check_id, EnvironmentValidationId.E1)
        self.assertEqual(E1_RGB_CASE.name, "rgb_observation")
        self.assertIs(E1_RGB_CASE, P1_VALIDATION_CASES[1])
        self.assertFalse(E1_RGB_CASE.requires_server_truth)

    def test_result_rejects_e1_integration_claims(self) -> None:
        kwargs: dict[str, Any] = dict(
            check_id=EnvironmentValidationId.E1,
            name="rgb_observation",
            episode_id=EPISODE_ID,
            step_id=0,
            success=True,
            outcome="rgb_ok",
            created=True,
            reset_completed=True,
            initial_state_present=True,
            closed=True,
            rgb_present=True,
            rgb_height=360,
            rgb_width=640,
            rgb_channels=3,
            rgb_dtype="uint8",
        )
        with self.assertRaisesRegex(ValueError, "integration_verified"):
            EnvironmentValidationResult(**kwargs, integration_verified=True)
        with self.assertRaisesRegex(ValueError, "real execution"):
            EnvironmentValidationResult(**kwargs, real_execution_performed=True)


if __name__ == "__main__":
    unittest.main()

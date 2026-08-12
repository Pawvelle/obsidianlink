from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout

from obsidianlink.cli import main


class CliTests(unittest.TestCase):
    def test_v2_offline_contract_check(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["--check"])
        self.assertEqual(result, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["phase"], "P1-REAL-MINERL-ENVIRONMENT-VALIDATION")
        self.assertIsNone(payload["active_benchmark_task_id"])
        self.assertEqual(payload["verification_level"], "unit_verified")
        self.assertEqual(payload["task_catalog_version"], "2026-08-12-v2")
        self.assertEqual(payload["task_catalog_entries"], 7)
        self.assertEqual(payload["legacy_entries"], 5)
        self.assertEqual(payload["calibration_entries"], 2)
        self.assertEqual(payload["benchmark_visible_entries"], 0)
        self.assertFalse(payload["live_run_allowed"])
        self.assertTrue(payload["p1_validation"]["contract_ready"])
        self.assertFalse(payload["p1_validation"]["real_execution_performed"])
        self.assertFalse(payload["p1_validation"]["integration_verified"])
        self.assertEqual(payload["v2_taxonomy_example"]["level"], "L1")
        self.assertEqual(
            [case["check_id"] for case in payload["p1_validation"]["cases"]],
            [f"E{index}" for index in range(13)],
        )
        self.assertIn("No MineRL/Minecraft", payload["note"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout

from obsidianlink.cli import main


class CliTests(unittest.TestCase):
    def test_offline_contract_check(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["--check"])
        self.assertEqual(result, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(
            payload["phase"], "reset_5_continuous_casting"
        )
        self.assertEqual(payload["active_task"], "casting_c3_fixed")
        self.assertFalse(payload["live_run_allowed"])
        # R4 single-cell contract is still part of the check.
        self.assertTrue(payload["r4"]["action_parser_accepted"])
        self.assertTrue(payload["r4"]["portal_evaluator_success"])
        self.assertEqual(
            payload["r4"]["casting_evaluator_outcome"], "truth_missing"
        )
        self.assertEqual(payload["r4"]["driver_status"], "completed")
        self.assertEqual(payload["r4"]["driver_success_outcome"], "success")
        # R5 multi-cell contract.
        self.assertEqual(payload["r5"]["c3_driver_status"], "completed")
        self.assertEqual(payload["r5"]["c3_evaluator_outcome"], "success")
        self.assertEqual(payload["r5"]["c3_evaluator_completed_cells"], 3)
        self.assertEqual(payload["r5"]["c3_evaluator_total_cells"], 3)
        self.assertTrue(payload["r5"]["c3_evaluator_success"])
        self.assertIn("no real MineRL", payload["note"])


if __name__ == "__main__":
    unittest.main()

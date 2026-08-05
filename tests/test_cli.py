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
        self.assertEqual(payload["phase"], "reset_2_capability_manifest")
        self.assertEqual(payload["active_task"], "casting_c1_fixed")
        self.assertFalse(payload["live_run_allowed"])
        self.assertTrue(payload["action_parser_accepted"])
        self.assertTrue(payload["portal_evaluator_success"])
        self.assertIn("no real MineRL", payload["note"])


if __name__ == "__main__":
    unittest.main()

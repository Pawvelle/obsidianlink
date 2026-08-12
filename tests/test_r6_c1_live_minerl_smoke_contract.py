"""Legacy C1 smoke compatibility invariants under the v2 scope reset.

This module never starts MineRL, Gradle, or a model API.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from obsidianlink.cli import _offline_contract_check
from obsidianlink.core.task_catalog import load_task_catalog


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "benchmark/catalog/tasks.json"
SMOKE_RUNBOOK = ROOT / "docs/runbooks/C1_LIVE_MINERL_SMOKE.md"


class C1LiveSmokeLegacyContractTests(unittest.TestCase):
    def test_cli_reports_p1_without_running_legacy_driver(self) -> None:
        payload = _offline_contract_check()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["phase"], "P1-REAL-MINERL-ENVIRONMENT-VALIDATION")
        self.assertFalse(payload["live_run_allowed"])
        self.assertEqual(payload["benchmark_visible_entries"], 0)
        self.assertIn("deterministic legacy drivers were not executed", payload["note"])

    def test_c1_and_c5_are_quarantined_legacy_entries(self) -> None:
        catalog = load_task_catalog(CATALOG_PATH)
        for compatibility_id in ("casting_c1_fixed", "casting_s_c5_fixed"):
            entry = catalog.entry_for_compatibility_id(compatibility_id)
            self.assertEqual(entry.kind, "legacy")
            self.assertEqual(entry.implementation_status, "legacy_regression")
            self.assertEqual(entry.verification_level, "unit_verified")
            self.assertFalse(entry.benchmark_visible)
            self.assertFalse(entry.live_run_allowed)

    def test_historical_c1_identity_and_safety_contract_remain_available(self) -> None:
        c1 = load_task_catalog(CATALOG_PATH).entry_for_compatibility_id(
            "casting_c1_fixed"
        )
        assert c1.taxonomy is not None
        self.assertEqual(c1.taxonomy.task_family, "casting")
        self.assertEqual(c1.taxonomy.task_level, "C1")
        runbook = SMOKE_RUNBOOK.read_text(encoding="utf-8")
        self.assertIn("Legacy", runbook)
        self.assertIn("不允许预置或直接放置 `obsidian`", runbook)
        self.assertIn("每次真实 MineRL/Minecraft 运行都需用户单独批准", runbook)


if __name__ == "__main__":
    unittest.main()

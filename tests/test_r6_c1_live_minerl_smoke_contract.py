"""Offline invariants for R6-C1 live MineRL smoke runner wiring.

Checks that:
* backend wiring remains done;
* current phase is C1 smoke runner wiring done;
* C5 remains contract_only / live_run_allowed=false;
* C1 smoke identity stays explicit;
* public docs do not claim live MineRL is verified.

It does not start MineRL, Gradle, or any model API.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from obsidianlink.cli import _offline_contract_check
from obsidianlink.core.task_catalog import load_task_catalog


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "benchmark/catalog/tasks.json"
SMOKE_RUNBOOK = ROOT / "docs/runbooks/C1_LIVE_MINERL_SMOKE.md"
PROJECT_STATUS = ROOT / "PROJECT_STATUS.md"
README = ROOT / "README.md"

EXPECTED_PHASE = "r6_c1_live_minerl_smoke_runner_wiring_done"
CURRENT_TASK = "R6-C1-LIVE-MINERL-SMOKE-RUNNER-WIRING"
FORBIDDEN_LIVE_CLAIMS = (
    "真实 MineRL 已验证",
    "真实 MineRL/Minecraft 已验证",
    "live MineRL verified",
    "真实浇筑已验证",
)


class C1LiveSmokeContractFreezeTests(unittest.TestCase):
    def test_cli_phase_is_runner_wiring_done(self) -> None:
        payload = _offline_contract_check()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["phase"], EXPECTED_PHASE)
        self.assertNotEqual(
            payload["phase"], "r6_c5_live_minerl_backend_wiring_done"
        )
        self.assertFalse(payload["live_run_allowed"])
        note = str(payload["note"])
        self.assertIn(CURRENT_TASK, note)
        self.assertIn("offline_stub", note)
        self.assertIn("no real MineRL", note)

    def test_check_environment_phase_matches(self) -> None:
        text = (ROOT / "scripts/check_environment.py").read_text(encoding="utf-8")
        self.assertIn(f'"{EXPECTED_PHASE}"', text)
        self.assertNotIn(
            '"r6_c5_live_minerl_backend_wiring_done"',
            text,
        )

    def test_c5_remains_contract_only_and_live_run_disallowed(self) -> None:
        catalog = load_task_catalog(CATALOG_PATH)
        c5 = next(
            entry
            for entry in catalog.entries
            if entry.compatibility_id == "casting_s_c5_fixed"
        )
        assert c5.taxonomy is not None
        self.assertEqual(c5.implementation_status, "contract_only")
        self.assertFalse(c5.live_run_allowed)
        self.assertEqual(c5.taxonomy.task_family, "casting")
        self.assertEqual(c5.taxonomy.agent_mode, "single")
        self.assertEqual(c5.taxonomy.task_level, "C5")
        self.assertEqual(c5.taxonomy.layout_type, "fixed")

    def test_c1_smoke_identity_is_explicit(self) -> None:
        catalog = load_task_catalog(CATALOG_PATH)
        c1 = next(
            entry
            for entry in catalog.entries
            if entry.compatibility_id == "casting_c1_fixed"
        )
        assert c1.taxonomy is not None
        self.assertEqual(c1.taxonomy.task_family, "casting")
        self.assertEqual(c1.taxonomy.agent_mode, "single")
        self.assertEqual(c1.taxonomy.task_level, "C1")
        self.assertEqual(c1.taxonomy.layout_type, "fixed")
        self.assertEqual(c1.canonical_name, "casting_s_c1_fixed")
        self.assertFalse(c1.live_run_allowed)

        runbook = SMOKE_RUNBOOK.read_text(encoding="utf-8")
        self.assertIn("family | `casting`", runbook)
        self.assertIn("mode | `single`", runbook)
        self.assertIn("level | `C1`", runbook)
        self.assertIn("layout | `fixed`", runbook)
        self.assertIn("`casting_c1_fixed`", runbook)
        self.assertIn("`agent_1`", runbook)
        self.assertIn("不允许预置或直接放置 `obsidian`", runbook)
        self.assertIn("live_run_allowed", runbook)
        self.assertIn(CURRENT_TASK, runbook)
        self.assertIn("offline_stub", runbook)
        self.assertIn(
            "下一步必须是用户单独授权的一次 C1 真实 MineRL smoke run",
            runbook,
        )

        status = PROJECT_STATUS.read_text(encoding="utf-8")
        self.assertIn(CURRENT_TASK, status)
        self.assertIn("compatibility task | `casting_c1_fixed`", status)
        self.assertIn("designated agent | `agent_1`", status)

        readme = README.read_text(encoding="utf-8")
        self.assertIn(CURRENT_TASK, readme)

    def test_docs_do_not_claim_live_minerl_verified(self) -> None:
        for path in (README, PROJECT_STATUS, SMOKE_RUNBOOK):
            text = path.read_text(encoding="utf-8")
            for phrase in FORBIDDEN_LIVE_CLAIMS:
                self.assertNotIn(
                    phrase,
                    text,
                    msg=f"{path.name} must not claim {phrase!r}",
                )
            self.assertIn("尚未验证", text)

    def test_active_implementation_remains_c2_compatibility_id(self) -> None:
        catalog = load_task_catalog(CATALOG_PATH)
        self.assertEqual(catalog.active_compatibility_id, "casting_c3_fixed")
        payload = json.loads((ROOT / "benchmark/catalog/tasks.json").read_text())
        self.assertEqual(payload["active_compatibility_id"], "casting_c3_fixed")


if __name__ == "__main__":
    unittest.main()

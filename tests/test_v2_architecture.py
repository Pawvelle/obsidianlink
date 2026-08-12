from __future__ import annotations

import ast
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from obsidianlink.benchmark.evidence import (
    EvidenceChannel,
    EvidenceIdentity,
    EvidenceRecord,
    VerificationLevel,
)
from obsidianlink.benchmark.task import (
    BenchmarkSuite,
    ExecutionMode,
    LayoutType,
    TaskIdentity,
)
from obsidianlink.env.validation import P1_VALIDATION_CASES, p1_validation_manifest
from obsidianlink.multi_agent.protocols import AgentMessage


ROOT = Path(__file__).resolve().parents[1]


class V2ArchitectureTests(unittest.TestCase):
    def test_v2_benchmark_modules_do_not_import_legacy_drivers(self) -> None:
        sources = tuple((ROOT / "obsidianlink/benchmark").glob("*.py"))
        self.assertTrue(sources)
        for source in sources:
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(
                        node.module.startswith(("obsidianlink.drivers", "obsidianlink.runners")),
                        f"{source.name} imports legacy solution module {node.module}",
                    )

    def test_task_identity_freezes_family_suite_mode_level_and_layout(self) -> None:
        identity = TaskIdentity(
            task_instance_id="future_p1_seed_0",
            suite=BenchmarkSuite.END_TO_END,
            mode=ExecutionMode.SINGLE,
            level="P1",
            layout=LayoutType.CONTROLLED,
        )
        self.assertEqual(identity.family, "nether_portal_construction")
        self.assertIn("end_to_end_p1_controlled", identity.canonical_name)
        with self.assertRaises(FrozenInstanceError):
            identity.level = "P2"  # type: ignore[misc]
        with self.assertRaisesRegex(ValueError, "invalid"):
            TaskIdentity(
                task_instance_id="bad",
                suite=BenchmarkSuite.DIAGNOSTIC,
                mode=ExecutionMode.SINGLE,
                level="P1",
                layout=LayoutType.CONTROLLED,
            )

    def test_verification_vocabulary_is_closed(self) -> None:
        self.assertEqual(
            {item.value for item in VerificationLevel},
            {"unit_verified", "integration_verified", "benchmark_evaluated"},
        )

    def test_p1_manifest_is_e0_through_e12_and_not_run(self) -> None:
        self.assertEqual(
            [case.check_id.value for case in P1_VALIDATION_CASES],
            [f"E{index}" for index in range(13)],
        )
        manifest = p1_validation_manifest()
        self.assertTrue(all(item["status"] == "not_run" for item in manifest))
        self.assertTrue(all(item["calibration_only"] for item in manifest))
        self.assertTrue(manifest[10]["requires_server_truth"])

    def test_evidence_channels_are_explicit_and_immutable(self) -> None:
        identity = EvidenceIdentity("episode-1", 2, "agent_1")
        record = EvidenceRecord(
            identity=identity,
            channel=EvidenceChannel.EVALUATOR_ONLY,
            event_type="server_block_truth",
            payload={"block": "obsidian", "history": [{"step": 2}]},
        )
        self.assertEqual(record.channel, EvidenceChannel.EVALUATOR_ONLY)
        with self.assertRaises(TypeError):
            record.payload["block"] = "air"  # type: ignore[index]
        with self.assertRaises(TypeError):
            record.payload["history"][0]["step"] = 3  # type: ignore[index]

    def test_multi_agent_messages_are_explicit(self) -> None:
        message = AgentMessage("episode-1", 3, "agent_1", "agent_2", "water found")
        self.assertEqual(message.sender_id, "agent_1")
        with self.assertRaisesRegex(ValueError, "must differ"):
            AgentMessage("episode-1", 3, "agent_1", "agent_1", "hidden share")


if __name__ == "__main__":
    unittest.main()

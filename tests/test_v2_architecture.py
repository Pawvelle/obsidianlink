from __future__ import annotations

import ast
import importlib.util
import json
import re
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import obsidianlink
import obsidianlink.benchmark as benchmark_api
from obsidianlink import LegacyTaskInstance, TaskInstance
from obsidianlink.benchmark.evidence import (
    EvidenceChannel,
    EvidenceIdentity,
    EvidenceRecord,
    VerificationLevel,
)
from obsidianlink.benchmark.task import (
    BenchmarkSuite,
    DIAGNOSTIC_LEVELS,
    ExecutionMode,
    LayoutType,
    PORTAL_LEVELS,
    TaskIdentity,
)
from obsidianlink.env.validation import P1_VALIDATION_CASES, p1_validation_manifest
from obsidianlink.multi_agent.protocols import AgentMessage
from obsidianlink.tasks.portal_construction import PortalConstructionLevel


ROOT = Path(__file__).resolve().parents[1]
LEGACY_SOLUTION_PREFIXES = (
    "obsidianlink.drivers",
    "obsidianlink.runners",
)
LEGACY_EVALUATION_PREFIXES = ("obsidianlink.evaluation",)
AGENT_LAYER_PREFIXES = ("obsidianlink.agents",)
LEGACY_TASK_TYPE_PREFIXES = ("obsidianlink.core.types",)


def _python_sources(package: str) -> tuple[Path, ...]:
    return tuple(sorted((ROOT / package).rglob("*.py")))


def _imported_modules(source: Path) -> tuple[str, ...]:
    modules: list[str] = []
    tree = ast.parse(source.read_text(encoding="utf-8"))
    relative = source.relative_to(ROOT).with_suffix("")
    package_parts = relative.parts if source.name == "__init__.py" else relative.parts[:-1]
    package = ".".join(package_parts)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative_name = "." * node.level + (node.module or "")
                modules.append(importlib.util.resolve_name(relative_name, package))
            elif node.module:
                modules.append(node.module)
    return tuple(modules)


def _matches_prefix(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(f"{prefix}.")


class V2ArchitectureTests(unittest.TestCase):
    def assert_package_avoids_imports(
        self, package: str, banned_prefixes: tuple[str, ...]
    ) -> None:
        sources = _python_sources(package)
        self.assertTrue(sources)
        for source in sources:
            for module in _imported_modules(source):
                for prefix in banned_prefixes:
                    self.assertFalse(
                        _matches_prefix(module, prefix),
                        f"{source.relative_to(ROOT)} imports forbidden module {module}",
                    )

    def test_v2_benchmark_is_solver_and_model_independent(self) -> None:
        self.assert_package_avoids_imports(
            "obsidianlink/benchmark",
            LEGACY_SOLUTION_PREFIXES
            + LEGACY_EVALUATION_PREFIXES
            + AGENT_LAYER_PREFIXES
            + LEGACY_TASK_TYPE_PREFIXES,
        )

    def test_v2_tasks_do_not_import_legacy_solution_or_evaluator_modules(self) -> None:
        self.assert_package_avoids_imports(
            "obsidianlink/tasks",
            LEGACY_SOLUTION_PREFIXES
            + LEGACY_EVALUATION_PREFIXES
            + LEGACY_TASK_TYPE_PREFIXES,
        )

    def test_p1_validation_does_not_import_legacy_or_benchmark_evaluator_modules(
        self,
    ) -> None:
        self.assert_package_avoids_imports(
            "obsidianlink/env/validation",
            LEGACY_SOLUTION_PREFIXES
            + LEGACY_EVALUATION_PREFIXES
            + AGENT_LAYER_PREFIXES
            + (
                "obsidianlink.benchmark.evaluator",
                "obsidianlink.env.fake",
                "obsidianlink.env.minerl_backend",
            ),
        )

    def test_v2_packages_do_not_import_legacy_task_instance(self) -> None:
        for package in ("obsidianlink/benchmark", "obsidianlink/tasks"):
            for source in _python_sources(package):
                tree = ast.parse(source.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        imported_names = {alias.name for alias in node.names}
                        self.assertTrue(
                            imported_names.isdisjoint(
                                {"TaskInstance", "LegacyTaskInstance"}
                            ),
                            f"{source.relative_to(ROOT)} imports legacy TaskInstance",
                        )

    def test_v2_tasks_do_not_embed_v1_route_taxonomy(self) -> None:
        legacy_route_values = {"obsidian_mining", "lava_casting"}
        for source in _python_sources("obsidianlink/tasks"):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            constants = {
                node.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            }
            self.assertTrue(
                constants.isdisjoint(legacy_route_values),
                f"{source.relative_to(ROOT)} embeds v1 route taxonomy",
            )

    def test_task_identity_freezes_family_suite_mode_level_and_layout(self) -> None:
        identity = TaskIdentity(
            task_instance_id="future_l1_seed_0",
            suite=BenchmarkSuite.END_TO_END,
            mode=ExecutionMode.SINGLE,
            level="L1",
            layout=LayoutType.CONTROLLED,
        )
        self.assertEqual(identity.family, "nether_portal_construction")
        self.assertEqual(
            identity.canonical_name,
            "nether_portal_construction_s_end_to_end_l1_controlled",
        )
        with self.assertRaises(FrozenInstanceError):
            identity.level = "L2"  # type: ignore[misc]

    def test_task_identity_level_namespaces_fail_closed(self) -> None:
        self.assertEqual(DIAGNOSTIC_LEVELS, {f"D{index}" for index in range(1, 7)})
        self.assertEqual(PORTAL_LEVELS, {f"L{index}" for index in range(1, 5)})
        for level in sorted(DIAGNOSTIC_LEVELS):
            TaskIdentity(
                task_instance_id=f"diagnostic_{level.lower()}",
                suite=BenchmarkSuite.DIAGNOSTIC,
                mode=ExecutionMode.SINGLE,
                level=level,
                layout=LayoutType.CONTROLLED,
            )
        for suite in (
            BenchmarkSuite.END_TO_END,
            BenchmarkSuite.GENERALIZATION_RECOVERY,
        ):
            for level in sorted(PORTAL_LEVELS):
                TaskIdentity(
                    task_instance_id=f"{suite.value}_{level.lower()}",
                    suite=suite,
                    mode=ExecutionMode.SINGLE,
                    level=level,
                    layout=LayoutType.CONTROLLED,
                )
        with self.assertRaisesRegex(ValueError, "invalid"):
            TaskIdentity(
                task_instance_id="old_conflicting_level",
                suite=BenchmarkSuite.END_TO_END,
                mode=ExecutionMode.SINGLE,
                level="P1",
                layout=LayoutType.CONTROLLED,
            )
        with self.assertRaisesRegex(ValueError, "invalid"):
            TaskIdentity(
                task_instance_id="bad",
                suite=BenchmarkSuite.DIAGNOSTIC,
                mode=ExecutionMode.SINGLE,
                level="L1",
                layout=LayoutType.CONTROLLED,
            )

    def test_schema_uses_diagnostic_d_and_end_to_end_l_levels(self) -> None:
        schema = json.loads(
            (ROOT / "benchmark/schemas/v2_task_identity.schema.json").read_text(
                encoding="utf-8"
            )
        )
        level_pattern = schema["properties"]["level"]["pattern"]
        end_to_end_pattern = schema["allOf"][0]["else"]["properties"]["level"][
            "pattern"
        ]
        self.assertIsNotNone(re.fullmatch(level_pattern, "D1"))
        self.assertIsNotNone(re.fullmatch(level_pattern, "L4"))
        self.assertIsNone(re.fullmatch(level_pattern, "P1"))
        self.assertIsNotNone(re.fullmatch(end_to_end_pattern, "L1"))
        self.assertIsNone(re.fullmatch(end_to_end_pattern, "P1"))

    def test_roadmap_phase_p1_and_task_level_l1_are_distinct(self) -> None:
        roadmap = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
        phase_ids = re.findall(r"^## (P[0-8]) —", roadmap, flags=re.MULTILINE)
        self.assertEqual(phase_ids, [f"P{index}" for index in range(9)])
        self.assertIn("## P1 — Real Environment Validation", roadmap)
        self.assertEqual(
            {level.value for level in PortalConstructionLevel},
            {f"L{index}" for index in range(1, 5)},
        )

    def test_legacy_task_instance_alias_is_explicit_but_not_v2_api(self) -> None:
        self.assertIs(LegacyTaskInstance, TaskInstance)
        self.assertIn("LegacyTaskInstance", obsidianlink.__all__)
        self.assertIn("v1 task instance", TaskInstance.__doc__.lower())
        self.assertIn("TaskIdentity", TaskInstance.__doc__)
        self.assertFalse(hasattr(benchmark_api, "TaskInstance"))

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

from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_DOCS = (
    ROOT / "README.md",
    ROOT / "PROJECT_STATUS.md",
    ROOT / "AGENTS.md",
    ROOT / "ROADMAP.md",
    ROOT / "BENCHMARK_SPEC.md",
    ROOT / "DATASET_CARD.md",
    ROOT / "docs/benchmark/TASK_TAXONOMY.md",
    ROOT / "docs/architecture/TASK_REGISTRY.md",
    ROOT / "docs/architecture/V2_ARCHITECTURE.md",
    ROOT / "docs/architecture/P1_ENVIRONMENT_VALIDATION.md",
    ROOT / "docs/legacy/v1/README.md",
)
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
ACTIVE_ROOT_DOCS = (
    "README.md",
    "PROJECT_STATUS.md",
    "ROADMAP.md",
    "BENCHMARK_SPEC.md",
    "DATASET_CARD.md",
)


class ProjectDocumentationTests(unittest.TestCase):
    def test_core_documents_exist_and_are_not_empty(self) -> None:
        for path in CORE_DOCS:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertTrue(path.is_file())
                self.assertGreater(len(path.read_text(encoding="utf-8")), 100)

    def test_project_status_has_only_p1_as_active_phase(self) -> None:
        text = (ROOT / "PROJECT_STATUS.md").read_text(encoding="utf-8")
        self.assertIn("P1-REAL-MINERL-ENVIRONMENT-VALIDATION", text)
        self.assertIn("当前唯一 active task", text)
        self.assertIn("P1 real MineRL environment validation", text)
        self.assertLess(len(text), 10000)

    def test_active_docs_use_unified_v2_scope(self) -> None:
        for relative in ACTIVE_ROOT_DOCS:
            text = (ROOT / relative).read_text(encoding="utf-8")
            with self.subTest(relative=relative):
                self.assertIn("Nether Portal", text)
                self.assertIn("unit_verified", text)
                self.assertNotIn("## Suite B — Ruined Portal", text)
                self.assertNotIn("## Suite C — Adaptive Routing", text)
                self.assertNotIn("Casting-S/M", text)

    def test_docs_do_not_equate_legacy_milestones_with_end_to_end_success(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        spec = (ROOT / "BENCHMARK_SPEC.md").read_text(encoding="utf-8")
        for text in (readme, spec):
            self.assertIn("driver `completed`", text)
            self.assertIn("Nether entry", text)
            self.assertIn("FakeBackend success", text)

    def test_active_developer_docs_freeze_mc_agent_conda_environment(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        status = (ROOT / "PROJECT_STATUS.md").read_text(encoding="utf-8")
        for text in (readme, agents, status):
            self.assertIn("mc-agent", text)
            self.assertIn("environment.yml", text)
        self.assertIn("conda run -n mc-agent python", readme)
        self.assertIn("conda run -n mc-agent python", agents)
        self.assertIn("不得退回 `/usr/bin/python3`", agents)

    def test_v1_authoritative_docs_are_archived(self) -> None:
        for relative in (
            "PROJECT_STATUS_V1.md",
            "BENCHMARK_SPEC_V1.md",
            "ROADMAP_V1.md",
            "TASK_TAXONOMY_V1.md",
            "TASK_REGISTRY_V1.md",
        ):
            self.assertTrue((ROOT / "docs/legacy/v1" / relative).is_file())

    def test_local_markdown_links_resolve(self) -> None:
        for path in CORE_DOCS:
            text = path.read_text(encoding="utf-8")
            for target in MARKDOWN_LINK.findall(text):
                target = target.strip().strip("<>")
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                file_part = target.split("#", 1)[0]
                if not file_part:
                    continue
                resolved = (path.parent / file_part).resolve()
                with self.subTest(document=path.relative_to(ROOT), target=target):
                    self.assertTrue(resolved.exists(), resolved)


if __name__ == "__main__":
    unittest.main()

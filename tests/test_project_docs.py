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
    ROOT / "docs/runbooks/FIRST_OBSIDIAN_BLOCK.md",
)
MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


class ProjectDocumentationTests(unittest.TestCase):
    def test_core_documents_exist_and_are_not_empty(self) -> None:
        for path in CORE_DOCS:
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertTrue(path.is_file())
                self.assertGreater(len(path.read_text(encoding="utf-8")), 100)

    def test_current_status_names_one_offline_next_task(self) -> None:
        text = (ROOT / "PROJECT_STATUS.md").read_text(encoding="utf-8")
        self.assertIn("R2-CAPABILITY-MANIFEST", text)
        self.assertIn("当前唯一目标", text)
        self.assertIn("禁止真实 MineRL、Gradle 和模型调用", text)

    def test_current_task_is_the_only_named_work_item(self) -> None:
        status = (ROOT / "PROJECT_STATUS.md").read_text(encoding="utf-8")
        self.assertIn("R2-CAPABILITY-MANIFEST", status)
        self.assertIn("BackendCapabilities", status)
        self.assertIn("R3-CASTING-EVALUATOR", status)

    def test_root_docs_use_the_real_casting_mainline(self) -> None:
        for name in (
            "README.md",
            "PROJECT_STATUS.md",
            "ROADMAP.md",
            "BENCHMARK_SPEC.md",
            "DATASET_CARD.md",
        ):
            with self.subTest(name=name):
                text = (ROOT / name).read_text(encoding="utf-8")
                self.assertIn("casting_c1_fixed", text)

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
                with self.subTest(
                    document=path.relative_to(ROOT),
                    target=target,
                ):
                    self.assertTrue(resolved.exists(), resolved)


if __name__ == "__main__":
    unittest.main()

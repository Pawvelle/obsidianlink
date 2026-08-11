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
    ROOT / "docs/tasks/casting/casting_c1_fixed.md",
    ROOT / "docs/tasks/casting/casting_c3_fixed.md",
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
        # The current task line is the one immediately under
        # ``当前唯一目标``. The project must declare exactly one
        # in-flight phase, and the safety reminder must stay
        # present. After offline engineering milestones complete,
        # the next step may be an authorized live smoke run rather
        # than another ``R*-...`` offline task id.
        self.assertIn("当前唯一目标", text)
        self.assertIn("禁止真实 MineRL、Gradle 和模型调用", text)
        self.assertTrue(
            re.search(r"下一任务：`(R[3-9]-[A-Z0-9-]+)`", text) is not None
            or (
                "下一任务：用户单独授权的一次" in text
                and "C1" in text
                and "真实 MineRL smoke run" in text
            ),
            msg="PROJECT_STATUS must name either an R*-task or an authorized C1 live smoke next step",
        )

    def test_current_task_is_the_only_named_work_item(self) -> None:
        status = (ROOT / "PROJECT_STATUS.md").read_text(encoding="utf-8")
        # Both R2 (the completed capability-manifest phase) and
        # the current casting-c1 evaluator phase must remain
        # referenced so the document doubles as a project history.
        self.assertIn("R2-CAPABILITY-MANIFEST", status)
        self.assertIn("BackendCapabilities", status)
        self.assertIn("CastingEvaluationState", status)
        self.assertIn("R4-DETERMINISTIC-CASTING-DRIVER", status)

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
                self.assertIn("casting_c3_fixed", text)

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

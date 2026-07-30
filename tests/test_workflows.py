from __future__ import annotations

import unittest

from obsidianlink.workflows.model import WorkflowDefinition, WorkflowStage


class WorkflowDefinitionTests(unittest.TestCase):
    def test_dependencies_unlock_in_order(self) -> None:
        workflow = WorkflowDefinition(
            (
                WorkflowStage("check_resources"),
                WorkflowStage("build_frame", ("check_resources",)),
                WorkflowStage("activate", ("build_frame",)),
            )
        )
        self.assertEqual(
            [stage.name for stage in workflow.available(())],
            ["check_resources"],
        )
        self.assertEqual(
            [stage.name for stage in workflow.available(("check_resources",))],
            ["build_frame"],
        )
        self.assertTrue(
            workflow.is_complete(("check_resources", "build_frame", "activate"))
        )

    def test_forward_dependency_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unavailable"):
            WorkflowDefinition(
                (
                    WorkflowStage("activate", ("build_frame",)),
                    WorkflowStage("build_frame"),
                )
            )


if __name__ == "__main__":
    unittest.main()

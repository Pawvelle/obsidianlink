from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCHES = ROOT / "patches" / "minerl"
ENVSERVER_PATCH = PATCHES / "obsidianlink-envserver.patch"
DRAW_PATCH = PATCHES / "e10-drawing-decorator.patch"
E11_PATCH = PATCHES / "e11-drawing-decorator-obsidian.patch"
MARSHAL_PATCH = PATCHES / "e11-server-thread-marshal.patch"
DIAG_PATCH = PATCHES / "e11-portal-activation-diagnostic.patch"
SITE_MCP = Path(
    "/opt/anaconda3/envs/mc-agent/lib/python3.10/site-packages/minerl/MCP-Reborn"
)
FORBIDDEN_RUNTIME = (
    "entered_via_portal",
    "PortalTransition",
    "portal_transition",
    "netherEntry",
    "changeDimension",
    "Blocks.NETHER_PORTAL",
    "BlockType.PORTAL.getDefaultState",
    "placePortalBlocks",
    "func_242967_a",
)


def _apply(patch: Path, cwd: Path) -> None:
    completed = subprocess.run(
        ["patch", "-p1", "--forward", "--no-backup-if-mismatch", "-i", str(patch)],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in (0, 1):
        raise AssertionError(completed.stdout + completed.stderr)


class E11ServerThreadMarshalPatchTests(unittest.TestCase):
    def test_patch_is_narrow_envserver_action_boundary(self) -> None:
        text = MARSHAL_PATCH.read_text(encoding="utf-8")
        self.assertIn("src/main/java/com/minerl/multiagent/env/EnvServer.java", text)
        self.assertNotIn("SoundEngine.java", text)
        self.assertNotIn("launchClient.sh", text)
        self.assertNotIn("build.gradle", text)
        self.assertNotIn("AbstractFireBlock.java", text)
        self.assertNotIn("PortalSize.java", text)
        self.assertNotIn("NetherPortalBlock.java", text)
        self.assertNotIn("FlintAndSteelItem.java", text)
        self.assertIn("marshalFlintAndSteelUseToServerThread", text)
        self.assertIn("server.execute(() -> {", text)
        self.assertIn("func_219441_a", text)
        self.assertIn("Items.FLINT_AND_STEEL", text)
        self.assertIn("stripUseMouseButton", text)
        self.assertIn("[E11-MARSHAL]", text)
        self.assertIn("Thread.currentThread().getName()", text)
        self.assertIn("player.pick(4.5D, 1.0F, false)", text)
        added = "\n".join(
            line[1:]
            for line in text.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        self.assertIn("LOGGER.info", added)
        self.assertNotIn("setBlockState", added)
        for marker in FORBIDDEN_RUNTIME:
            self.assertNotIn(marker, text, marker)
        digest = hashlib.sha256(MARSHAL_PATCH.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "45bf4af8c323806f0905e5851d236ba77f28c925f5ee39f87b50a15b7f2f9be8",
        )

    def test_diagnostic_patch_is_not_part_of_the_marshal_fix(self) -> None:
        self.assertTrue(DIAG_PATCH.is_file())
        marshal = MARSHAL_PATCH.read_text(encoding="utf-8")
        diagnostic = DIAG_PATCH.read_text(encoding="utf-8")
        self.assertNotIn("placePortalBlocks SET", marshal)
        self.assertIn("placePortalBlocks SET", diagnostic)
        self.assertNotIn("marshalFlintAndSteelUseToServerThread", diagnostic)

    def test_reconstructed_envserver_marshals_flint_and_steel_only(self) -> None:
        source = SITE_MCP / "src/main/java/com/minerl/multiagent/env/EnvServer.java"
        self.assertTrue(source.is_file(), "P1 baseline EnvServer is missing")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_dir = root / "src/main/java/com/minerl/multiagent/env"
            env_dir.mkdir(parents=True)
            shutil.copy(source, env_dir / "EnvServer.java")
            _apply(ENVSERVER_PATCH, root)
            _apply(DRAW_PATCH, root)
            _apply(E11_PATCH, root)
            before = (env_dir / "EnvServer.java").read_text(encoding="utf-8")
            self.assertNotIn("marshalFlintAndSteelUseToServerThread", before)
            _apply(MARSHAL_PATCH, root)
            env_text = (env_dir / "EnvServer.java").read_text(encoding="utf-8")
            self.assertIn("marshalFlintAndSteelUseToServerThread", env_text)
            self.assertIn("stripUseMouseButton", env_text)
            self.assertIn("Items.FLINT_AND_STEEL", env_text)
            self.assertIn("func_219441_a", env_text)
            self.assertIn("must not pre-place portal or fire", env_text)
            self.assertIn("Blocks.OBSIDIAN.getDefaultState()", env_text)
            self.assertNotIn("Blocks.NETHER_PORTAL", env_text)
            self.assertIn(
                "if (marshalFlintAndSteelUseToServerThread(actions))",
                env_text,
            )
            exec_start = env_text.index("public static void execActions")
            exec_end = env_text.index(
                "private static KeyboardListener.State constructKeyboardState"
            )
            chunk = env_text[exec_start:exec_end]
            self.assertEqual(chunk.count("{"), chunk.count("}"))
            for marker in FORBIDDEN_RUNTIME:
                self.assertNotIn(marker, env_text, marker)


class RecordedE11ServerThreadMarshalLiveTests(unittest.TestCase):
    HISTORY = ROOT / "runs" / "history" / "p1-e11-live-20260817-001"

    def test_recorded_marshal_live_timed_out_before_flint_and_steel(self) -> None:
        payload = json.loads((self.HISTORY / "result.json").read_text(encoding="utf-8"))
        review = json.loads((self.HISTORY / "run_review.json").read_text(encoding="utf-8"))
        identity = json.loads(
            (self.HISTORY / "runtime_identity.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["episode_id"], "p1-e11-live-002")
        self.assertFalse(payload["success"])
        self.assertEqual(payload["outcome"], "truth_identity_mismatch")
        self.assertEqual(payload["tested_action_count"], 0)
        self.assertIsNone(payload["translated_action_accepted"])
        self.assertIsNone(payload["after_block_truth"])
        self.assertIsNone(payload["truth_missing_count"])
        self.assertTrue(payload["reset_completed"])
        self.assertTrue(payload["opened"])
        self.assertTrue(payload["real_execution_performed"])
        self.assertFalse(payload["integration_verified"])
        self.assertEqual(payload["before_dimension"], "minecraft:overworld")
        before = payload["before_block_truth"]
        interior = {
            (0, 4, 1),
            (1, 4, 1),
            (0, 5, 1),
            (1, 5, 1),
            (0, 6, 1),
            (1, 6, 1),
        }
        controls = {(0, 8, 1), (0, 4, 3)}
        self.assertEqual(
            sum(
                1
                for cell in before
                if tuple(cell["world_cell"]) not in interior | controls
                and cell["block"] == "obsidian"
            ),
            14,
        )
        self.assertEqual(
            sum(1 for cell in before if tuple(cell["world_cell"]) in interior and cell["block"] == "air"),
            6,
        )
        self.assertEqual(review["failure_stage"], "flint_and_steel_server_thread_marshal_timeout")
        self.assertEqual(review["nether_portal_count_after"], None)
        self.assertFalse(review["verification"]["e11_real_activation_reviewed_success"])
        self.assertFalse(review["verification"]["e11_integration_verified"])
        self.assertFalse(review["verification"]["e12_started"])
        self.assertFalse(review["verification"]["second_patch_round"])
        self.assertEqual(
            identity["run_jar_sha256"],
            "c69fd49e030d501b4c0b1cca9f58b47a2722baf0af9695d679d84188a8499196",
        )
        self.assertEqual(
            identity["production_jar_restored_after_failure"],
            "836cb5ac6f89edca3cec255dd895e791212b04794d3349eb13a1b2b313416b6f",
        )

    def test_jvm_log_proves_marshal_wait_deadlocked(self) -> None:
        text = (self.HISTORY / "e11_marshal_jvm.log").read_text(encoding="utf-8")
        self.assertIn("flint_and_steel server-thread marshal timed out", text)
        self.assertIn("Saving and pausing game", text)
        self.assertNotIn("[E11-MARSHAL]", text)
        self.assertIn("awaitMarshaledFlintAndSteelUse", text)

    def test_live_001_evaluator_replay_is_not_rewritten(self) -> None:
        from obsidianlink.env.integration.e11_diagnostics import (
            load_recorded_result,
            replay_recorded_evaluator,
        )
        from obsidianlink.env.validation.truth import PORTAL_ACTIVATION_NOT_OBSERVED

        live = load_recorded_result()
        self.assertEqual(live["episode_id"], "p1-e11-live-001")
        inspection = replay_recorded_evaluator(live)
        self.assertEqual(inspection.outcome, PORTAL_ACTIVATION_NOT_OBSERVED)
        self.assertEqual(inspection.after_portal_block_count, 0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE_MCP = Path(
    "/opt/anaconda3/envs/mc-agent/lib/python3.10/site-packages/minerl/MCP-Reborn"
)
SCRIPT = ROOT / "scripts" / "build_p1_canonical_runtime.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_p1_canonical_runtime", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("canonical runtime builder could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class P1CanonicalRuntimeTests(unittest.TestCase):
    def test_patch_chain_excludes_old_e11_and_e12_changes(self) -> None:
        builder = _load_builder()
        self.assertEqual(
            builder.CANONICAL_PATCHES,
            (
                "obsidianlink-envserver.patch",
                "p1-canonical-audio-source.patch",
                "e10-drawing-decorator.patch",
                "e11-drawing-decorator-obsidian.patch",
                "p1-env-integrated-server-unpaused.patch",
            ),
        )
        names = "\n".join(builder.CANONICAL_PATCHES)
        for forbidden in ("marshal", "diagnostic", "paused-executor", "e12"):
            self.assertNotIn(forbidden, names)
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('["./gradlew", "--no-daemon", "shadowJar"]', script)

    def test_environment_runtime_keeps_integrated_server_unpaused(self) -> None:
        text = (
            ROOT
            / "patches/minerl/p1-env-integrated-server-unpaused.patch"
        ).read_text(encoding="utf-8")
        self.assertIn("IntegratedServer.java", text)
        self.assertIn("this.mc.gameSettings.envPort == 0", text)
        self.assertIn("Minecraft.getInstance().isGamePaused()", text)
        self.assertNotIn("canRun(", text)
        self.assertNotIn("TickDelayedTask", text)
        self.assertNotIn("ObsidianLinkE11Task", text)
        self.assertNotIn("processRightClickBlock", text)
        self.assertNotIn("setBlockState", text)
        self.assertNotIn("NETHER_PORTAL", text)

    def test_staged_runtime_is_clean_and_preserves_unrelated_sources(self) -> None:
        builder = _load_builder()
        relative_files = tuple(builder.FROZEN_SOURCE_SHA256) + (
            "src/main/java/net/minecraft/entity/Entity.java",
            "src/main/java/net/minecraft/entity/player/ServerPlayerEntity.java",
        )
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            source = temporary / "source"
            output = temporary / "canonical"
            for relative in relative_files:
                source_path = SITE_MCP / relative
                destination = source / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, destination)
            entity_before = (
                source / "src/main/java/net/minecraft/entity/Entity.java"
            ).read_bytes()
            player_before = (
                source
                / "src/main/java/net/minecraft/entity/player/ServerPlayerEntity.java"
            ).read_bytes()

            manifest = builder.stage_canonical_runtime(source, output)

            env_server = (
                output
                / "src/main/java/com/minerl/multiagent/env/EnvServer.java"
            ).read_text(encoding="utf-8")
            integrated_server = (
                output
                / "src/main/java/net/minecraft/server/integrated/IntegratedServer.java"
            ).read_text(encoding="utf-8")
            self.assertIn("Blocks.LAVA.getDefaultState()", env_server)
            self.assertIn("Blocks.OBSIDIAN.getDefaultState()", env_server)
            self.assertNotIn("Blocks.NETHER_PORTAL", env_server)
            self.assertIn("this.mc.gameSettings.envPort == 0", integrated_server)
            self.assertNotIn("ObsidianLinkE11Task", integrated_server)
            self.assertEqual(
                (
                    output / "src/main/java/net/minecraft/entity/Entity.java"
                ).read_bytes(),
                entity_before,
            )
            self.assertEqual(
                (
                    output
                    / "src/main/java/net/minecraft/entity/player/ServerPlayerEntity.java"
                ).read_bytes(),
                player_before,
            )
            self.assertIn("mcp_patch.diff", manifest["excluded_patch_families"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCHES = ROOT / "patches" / "minerl"
ENVSERVER_PATCH = PATCHES / "obsidianlink-envserver.patch"
AUDIO_PATCH = PATCHES / "disable-client-audio.patch"
DRAW_PATCH = PATCHES / "e10-drawing-decorator.patch"
SITE_MCP = Path(
    "/opt/anaconda3/envs/mc-agent/lib/python3.10/site-packages/minerl/MCP-Reborn"
)
FORBIDDEN_RUNTIME = (
    "entered_via_portal",
    "PortalTransition",
    "portal_transition",
    "netherEntry",
    "ServerPlayerEntity portal",
)


def _apply(patch: Path, cwd: Path, extra: list[str] | None = None) -> None:
    command = ["patch", "-p1", "--forward", "--no-backup-if-mismatch"]
    if extra:
        command.extend(extra)
    completed = subprocess.run(
        command + ["-i", str(patch)],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in (0, 1):
        raise AssertionError(completed.stdout + completed.stderr)


class E10DrawingDecoratorRuntimePatchTests(unittest.TestCase):
    def test_patch_is_narrow_fail_closed_and_ordered_after_platform(self) -> None:
        text = DRAW_PATCH.read_text(encoding="utf-8")
        self.assertIn("src/main/java/com/minerl/multiagent/env/EnvServer.java", text)
        self.assertNotIn("SoundEngine.java", text)
        self.assertNotIn("launchClient.sh", text)
        self.assertNotIn("build.gradle", text)
        self.assertIn("prepareControlledBuildArea(mc, missionInit);", text)
        self.assertIn("applyMissionDrawingDecorator(mc, missionInit);", text)
        platform = text.index("prepareControlledBuildArea(mc, missionInit);")
        draw = text.index("applyMissionDrawingDecorator(mc, missionInit);")
        inventory = text.index("setAgentInventory(mc.player, missionInit)")
        skip = text.index("waitForNextObservation();")
        await_done = text.index("awaitMissionDrawingDecorator(drawingDone);")
        self.assertLess(platform, draw)
        self.assertLess(draw, inventory)
        self.assertLess(inventory, skip)
        self.assertLess(skip, await_done)
        self.assertIn("return done;", text)
        self.assertIn("drawingDone.get(30L, TimeUnit.SECONDS)", text)
        self.assertIn("BlockType.LAVA", text)
        self.assertIn("Blocks.LAVA.getDefaultState()", text)
        self.assertNotIn("FLOWING_LAVA", text)
        self.assertIn("BlockType.OBSIDIAN", text)
        self.assertIn("must not pre-place obsidian", text)
        self.assertIn("DrawBlock type not allowed", text)
        self.assertIn("new BlockPos(x, y, z)", text)
        self.assertNotIn("startPos.getX() +", text)
        self.assertIn("decorators.isEmpty()", text)
        self.assertIn("drawingDone.get(30L, TimeUnit.SECONDS)", text)
        self.assertIn("server.execute", text)
        self.assertIn("y < 0 || y > 255", text)
        for marker in FORBIDDEN_RUNTIME:
            self.assertNotIn(marker, text)

    def test_audio_mitigation_patch_is_unchanged(self) -> None:
        text = AUDIO_PATCH.read_text(encoding="utf-8")
        self.assertEqual(text.count("obsidianlink.disableClientAudio"), 2)
        self.assertIn("SoundEngine.java", text)
        self.assertIn("launchClient.sh", text)
        self.assertNotIn("DrawingDecorator", text)

    def test_reconstructed_envserver_applies_patches_without_future_code(self) -> None:
        source = SITE_MCP / "src/main/java/com/minerl/multiagent/env/EnvServer.java"
        sound = SITE_MCP / "src/main/java/net/minecraft/client/audio/SoundEngine.java"
        launcher = SITE_MCP / "launchClient.sh"
        self.assertTrue(source.is_file(), "P1 baseline EnvServer is missing")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_dir = root / "src/main/java/com/minerl/multiagent/env"
            sound_dir = root / "src/main/java/net/minecraft/client/audio"
            env_dir.mkdir(parents=True)
            sound_dir.mkdir(parents=True)
            shutil.copy(source, env_dir / "EnvServer.java")
            shutil.copy(sound, sound_dir / "SoundEngine.java")
            shutil.copy(launcher, root / "launchClient.sh")
            _apply(ENVSERVER_PATCH, root)
            _apply(AUDIO_PATCH, root)
            _apply(DRAW_PATCH, root)
            env_text = (env_dir / "EnvServer.java").read_text(encoding="utf-8")
            sound_text = (sound_dir / "SoundEngine.java").read_text(encoding="utf-8")
            launch_text = (root / "launchClient.sh").read_text(encoding="utf-8")
            call = (
                "applyServerInitialConditions(mc, missionInit);\n"
                "        prepareControlledBuildArea(mc, missionInit);\n"
                "        CompletableFuture<Void> drawingDone =\n"
                "                applyMissionDrawingDecorator(mc, missionInit);\n"
                "        mc.execute(() -> setAgentInventory(mc.player, missionInit));"
            )
            self.assertIn(call, env_text)
            self.assertIn("awaitMissionDrawingDecorator(drawingDone);", env_text)
            self.assertLess(
                env_text.index("waitForNextObservation();"),
                env_text.index("awaitMissionDrawingDecorator(drawingDone);"),
            )
            self.assertIn("must not pre-place obsidian", env_text)
            self.assertIn("BlockType.LAVA", env_text)
            self.assertIn("DrawBlock type not allowed", env_text)
            self.assertIn("new BlockPos(x, y, z)", env_text)
            self.assertIn("if (decorators == null || decorators.isEmpty())", env_text)
            self.assertIn("Boolean.getBoolean(\"obsidianlink.disableClientAudio\")", sound_text)
            self.assertIn("-Dobsidianlink.disableClientAudio=true", launch_text)
            for marker in FORBIDDEN_RUNTIME:
                self.assertNotIn(marker, env_text)
                self.assertNotIn(marker, sound_text)
            self.assertNotIn("obsidianlink-envserver.patch", env_text)

    def test_default_envspec_still_omits_drawing_decorator(self) -> None:
        from obsidianlink.env.portal_spec import PortalA0EnvSpec, parse_mission_draw_blocks

        self.assertEqual(parse_mission_draw_blocks(PortalA0EnvSpec().to_xml()), ())
        self.assertNotIn("DrawingDecorator", PortalA0EnvSpec().to_xml())


if __name__ == "__main__":
    unittest.main()

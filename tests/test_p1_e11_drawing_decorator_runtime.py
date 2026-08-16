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
E11_PATCH = PATCHES / "e11-drawing-decorator-obsidian.patch"
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


class E11DrawingDecoratorRuntimePatchTests(unittest.TestCase):
    def test_patch_is_narrow_fail_closed_obsidian_fixture(self) -> None:
        text = E11_PATCH.read_text(encoding="utf-8")
        self.assertIn("src/main/java/com/minerl/multiagent/env/EnvServer.java", text)
        self.assertNotIn("SoundEngine.java", text)
        self.assertNotIn("launchClient.sh", text)
        self.assertNotIn("build.gradle", text)
        self.assertIn("BlockType.OBSIDIAN", text)
        self.assertIn("Blocks.OBSIDIAN.getDefaultState()", text)
        self.assertIn("BlockType.LAVA", text)
        self.assertIn("Blocks.LAVA.getDefaultState()", text)
        self.assertIn("BlockType.PORTAL", text)
        self.assertIn("BlockType.FIRE", text)
        self.assertIn("must not pre-place portal or fire", text)
        self.assertIn("DrawBlock type not allowed", text)
        self.assertNotIn("FLOWING_LAVA", text)
        for marker in FORBIDDEN_RUNTIME:
            self.assertNotIn(marker, text)

    def test_e10_patch_still_rejects_obsidian_until_e11_is_applied(self) -> None:
        text = DRAW_PATCH.read_text(encoding="utf-8")
        self.assertIn("must not pre-place obsidian", text)
        self.assertNotIn("Blocks.OBSIDIAN.getDefaultState()", text)

    def test_reconstructed_envserver_allows_obsidian_after_e11_patch(self) -> None:
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
            before = (env_dir / "EnvServer.java").read_text(encoding="utf-8")
            self.assertIn("must not pre-place obsidian", before)
            _apply(E11_PATCH, root)
            env_text = (env_dir / "EnvServer.java").read_text(encoding="utf-8")
            self.assertNotIn("must not pre-place obsidian", env_text)
            self.assertIn("must not pre-place portal or fire", env_text)
            self.assertIn("Blocks.OBSIDIAN.getDefaultState()", env_text)
            self.assertIn("Blocks.LAVA.getDefaultState()", env_text)
            self.assertIn("BlockType.PORTAL", env_text)
            self.assertIn("BlockType.FIRE", env_text)
            self.assertIn("DrawBlock type not allowed", env_text)
            for marker in FORBIDDEN_RUNTIME:
                self.assertNotIn(marker, env_text)


if __name__ == "__main__":
    unittest.main()

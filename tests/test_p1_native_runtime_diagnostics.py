from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from obsidianlink.env.integration.native_runtime import (
    inspect_fatjar,
    parse_loaded_native_paths,
)


class NativeRuntimeDiagnosticsTests(unittest.TestCase):
    def test_fatjar_native_versions_and_architectures_are_static(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            jar = Path(directory) / "runtime.jar"
            with zipfile.ZipFile(jar, "w") as archive:
                archive.writestr(
                    "macos/arm64/org/lwjgl/stb/liblwjgl_stb.dylib", b"arm"
                )
                archive.writestr(
                    "macos/x64/org/lwjgl/stb/liblwjgl_stb.dylib", b"intel"
                )
                archive.writestr("unrelated.txt", b"ignored")
            details = inspect_fatjar(jar)
        resources = details["native_resources"]
        self.assertEqual(len(resources), 2)
        self.assertEqual(
            {item["architecture_classifier"] for item in resources},
            {"arm64", "x64"},
        )
        self.assertTrue(all(item["name"] == "liblwjgl_stb.dylib" for item in resources))

    def test_hs_err_loaded_native_paths_are_deduplicated(self) -> None:
        text = """
0x1 /tmp/lwjgluser/3.3.1-SNAPSHOT/liblwjgl_stb.dylib
0x2 /tmp/lwjgluser/3.3.1-SNAPSHOT/libopenal.dylib
0x3 /tmp/lwjgluser/3.3.1-SNAPSHOT/liblwjgl_stb.dylib
0x4 /tmp/unrelated.dylib
"""
        paths = parse_loaded_native_paths(text)
        self.assertEqual([path.name for path in paths], ["liblwjgl_stb.dylib", "libopenal.dylib"])

    def test_audio_mitigation_is_property_gated_without_dependency_change(self) -> None:
        patch_path = (
            Path(__file__).resolve().parents[1]
            / "patches/minerl/disable-client-audio.patch"
        )
        text = patch_path.read_text(encoding="utf-8")
        self.assertEqual(text.count("obsidianlink.disableClientAudio"), 2)
        self.assertIn("SoundEngine.java", text)
        self.assertIn("launchClient.sh", text)
        self.assertNotIn("build.gradle", text)
        self.assertNotIn("version: '3.", text)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_p1_e11_packet_diagnostic_runtime.py"
PATCH = ROOT / "patches" / "minerl" / "e11-packet-chain-diagnostic.patch"
CANONICAL_RUNTIME = Path("/private/tmp/obsidianlink-p1-canonical-e11-20260817")


def _load():
    spec = importlib.util.spec_from_file_location("packet_diagnostic", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("packet diagnostic builder could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class E11PacketDiagnosticRuntimeTests(unittest.TestCase):
    def test_patch_is_logging_only_and_covers_required_chain(self) -> None:
        text = PATCH.read_text(encoding="utf-8")
        for marker in (
            "client_send packet=CPlayerTryUseItemOnBlockPacket",
            "server_received handler=processTryUseItemOnBlock",
            "server_player_interaction flint_enter",
            "flint_onItemUse light_fire",
            "fire_onBlockAdded canLightPortal=",
            "portal_place_enter",
        ):
            self.assertIn(marker, text)
        added_lines = [line for line in text.splitlines() if line.startswith("+") and not line.startswith("+++")]
        self.assertFalse(any("Blocks.NETHER_PORTAL" in line for line in added_lines))
        self.assertFalse(any("world.setBlockState(blockpos1, Blocks.NETHER_PORTAL" in line for line in added_lines))

    def test_staging_applies_diagnostic_patch_to_verified_canonical_runtime(self) -> None:
        module = _load()
        self.assertEqual(module.DIAGNOSTIC_PATCH, "e11-packet-chain-diagnostic.patch")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "runtime"
            manifest = module.stage_packet_diagnostic_runtime(CANONICAL_RUNTIME, output)
            self.assertTrue(manifest["diagnostic_only"])
            self.assertEqual(manifest["canonical_runtime_root"], str(CANONICAL_RUNTIME))
            self.assertEqual(manifest["diagnostic_patch"]["name"], module.DIAGNOSTIC_PATCH)
            server = (output / "src/main/java/net/minecraft/network/play/ServerPlayNetHandler.java").read_text(encoding="utf-8")
            self.assertIn("server_received handler=processTryUseItemOnBlock", server)


if __name__ == "__main__":
    unittest.main()

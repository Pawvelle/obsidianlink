from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "scripts" / "build_p1_e11_completion_barrier_runtime.py"
CHECK = ROOT / "scripts" / "check_p1_e11_completion_barrier_semantic_diff.py"
PATCH = ROOT / "patches" / "minerl" / "p1-e11-action-completion-barrier.patch"
CANONICAL = Path("/private/tmp/obsidianlink-p1-canonical-e11-20260817")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class E11CompletionBarrierRuntimeTests(unittest.TestCase):
    def test_patch_uses_server_completion_not_a_tick_delay(self) -> None:
        text = PATCH.read_text(encoding="utf-8")
        self.assertIn("awaitE11FlintAndSteelCompletionBarrier", text)
        self.assertIn("completeE11FlintAndSteelCompletionBarrier", text)
        self.assertIn("isE11FlintAndSteelCompletionBarrierPending", text)
        self.assertIn("TimeUnit.NANOSECONDS.timedWait", text)
        self.assertNotIn("Blocks.NETHER_PORTAL", text)
        self.assertNotIn("PortalSize", text)
        self.assertNotIn("FlintAndSteelItem", text)
        self.assertNotIn("ServerPlayerEntity", text)
        self.assertNotIn("Entity.java", text)

    def test_staging_preserves_canonical_boundary(self) -> None:
        module = _load(BUILD, "completion_builder")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runtime"
            manifest = module.stage_completion_barrier_runtime(CANONICAL, output)
            self.assertEqual(manifest["baseline_jar_sha256"], module.CANONICAL_JAR_SHA256)
            source = (output / "src/main/java/com/minerl/multiagent/env/EnvServer.java").read_text(encoding="utf-8")
            self.assertIn("E11_ACTION_COMPLETION_MONITOR", source)
            self.assertNotIn("Blocks.NETHER_PORTAL", source)

    def test_semantic_checker_rejects_entity_mutation(self) -> None:
        module = _load(CHECK, "completion_checker")
        self.assertIn("com/minerl/multiagent/env/EnvServer$1.class", module.EXPECTED)
        self.assertIn("net/minecraft/client/ReplaySender$Mode.class", module.EXPECTED)
        self.assertNotIn("net/minecraft/entity/Entity.class", module.EXPECTED)
        self.assertNotIn("net/minecraft/entity/player/ServerPlayerEntity.class", module.EXPECTED)


if __name__ == "__main__":
    unittest.main()

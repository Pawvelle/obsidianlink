from __future__ import annotations

import importlib.util
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_p1_e11_packet_diagnostic_semantic_diff.py"


def _load():
    spec = importlib.util.spec_from_file_location("packet_diff", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("packet diagnostic diff checker could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class E11PacketDiagnosticSemanticDiffTests(unittest.TestCase):
    def test_accepts_only_expected_logging_classes(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.jar"
            candidate = root / "candidate.jar"
            entries = {name: b"base" for name in module.EXPECTED_CHANGED_ENTRIES}
            with zipfile.ZipFile(baseline, "w") as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)
            with zipfile.ZipFile(candidate, "w") as archive:
                for name in entries:
                    data = module.REQUIRED_LOG_MARKERS.get(name, b"changed")
                    archive.writestr(name, data)
            result = module.compare_packet_diagnostic_jars(baseline, candidate)
        self.assertTrue(result["semantic_diff_clean"])

    def test_allows_the_compiler_generated_server_handler_helper(self) -> None:
        module = _load()
        self.assertIn(
            "net/minecraft/network/play/ServerPlayNetHandler$1.class",
            module.EXPECTED_CHANGED_ENTRIES,
        )

    def test_rejects_an_unexpected_gameplay_class(self) -> None:
        module = _load()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.jar"
            candidate = root / "candidate.jar"
            with zipfile.ZipFile(baseline, "w") as archive:
                for name in module.EXPECTED_CHANGED_ENTRIES:
                    archive.writestr(name, b"base")
                archive.writestr("net/minecraft/entity/Entity.class", b"base")
            with zipfile.ZipFile(candidate, "w") as archive:
                for name in module.EXPECTED_CHANGED_ENTRIES:
                    archive.writestr(name, module.REQUIRED_LOG_MARKERS.get(name, b"changed"))
                archive.writestr("net/minecraft/entity/Entity.class", b"changed")
            with self.assertRaises(RuntimeError):
                module.compare_packet_diagnostic_jars(baseline, candidate)


if __name__ == "__main__":
    unittest.main()

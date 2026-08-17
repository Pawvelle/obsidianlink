#!/usr/bin/env python3
"""Fail closed unless the E11 completion barrier changes only its execution path."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Sequence


EXPECTED = {
    # Recompiling the enclosing source updates only LineNumberTable metadata in
    # these nested artifacts; their executable bytecode is unchanged.
    "com/minerl/multiagent/env/EnvServer$1.class",
    "com/minerl/multiagent/env/EnvServer.class",
    "net/minecraft/client/ReplaySender$Mode.class",
    "net/minecraft/client/ReplaySender.class",
    "net/minecraft/network/play/ServerPlayNetHandler.class",
    "net/minecraft/network/play/ServerPlayNetHandler$1.class",
    "version.properties",
}
REQUIRED = {
    "com/minerl/multiagent/env/EnvServer.class": b"awaitE11FlintAndSteelCompletionBarrier",
    "net/minecraft/client/ReplaySender.class": b"isE11FlintAndSteelCompletionBarrierPending",
    "net/minecraft/network/play/ServerPlayNetHandler.class": b"completeE11FlintAndSteelCompletionBarrier",
}
FORBIDDEN = (b"ObsidianLinkE11Task", b"E11-MARSHAL", b"Blocks.NETHER_PORTAL", b"portal_transition", b"entered_via_portal")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(baseline: Path, candidate: Path) -> dict[str, object]:
    with zipfile.ZipFile(baseline) as left, zipfile.ZipFile(candidate) as right:
        left_info = {item.filename: item for item in left.infolist()}
        right_info = {item.filename: item for item in right.infolist()}
        changed = {
            name for name in set(left_info) | set(right_info)
            if name not in left_info or name not in right_info
            or (left_info[name].CRC, left_info[name].file_size) != (right_info[name].CRC, right_info[name].file_size)
        }
        semantic = sorted(name for name in changed if name == "version.properties" or name.startswith(("com/minerl/", "net/minecraft/")))
        unexpected = sorted(set(semantic) - EXPECTED)
        missing = sorted(EXPECTED - set(semantic))
        missing_markers = sorted(name for name, marker in REQUIRED.items() if marker not in right.read(name))
        forbidden = {
            name: [marker.decode("ascii") for marker in FORBIDDEN if marker in right.read(name)]
            for name in semantic if name.endswith(".class") and any(marker in right.read(name) for marker in FORBIDDEN)
        }
    result = {
        "baseline_jar_sha256": _sha256(baseline),
        "candidate_jar_sha256": _sha256(candidate),
        "changed_semantic_entries": semantic,
        "expected_changed_entries": sorted(EXPECTED),
        "unexpected_changed_entries": unexpected,
        "missing_expected_changed_entries": missing,
        "missing_required_markers": missing_markers,
        "forbidden_marker_hits": forbidden,
        "semantic_diff_clean": not unexpected and not missing and not missing_markers and not forbidden,
    }
    if not result["semantic_diff_clean"]:
        raise RuntimeError(json.dumps(result, indent=2, sort_keys=True))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = compare(args.baseline.resolve(), args.candidate.resolve())
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.resolve().write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Print the offline PortalSize audit of recorded p1-e11-live-001.

Does not start MineRL or Minecraft.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obsidianlink.env.integration.e11_diagnostics import (  # noqa: E402
    diagnose_recorded_live_failure,
    diagnosis_as_dict,
)


def main() -> int:
    diagnosis = diagnose_recorded_live_failure()
    json.dump(diagnosis_as_dict(diagnosis), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

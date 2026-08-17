#!/usr/bin/env python3
"""CLI wrapper for the authorized P1 E0-E12 validation suite.

``--check`` and ``--preflight-only`` are offline-safe. A live pilot still
requires the exact authorization flags and must not be started without
separate user approval.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obsidianlink.env.integration.p1_suite import main


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Print a static MineRL/LWJGL runtime inventory without launching Minecraft."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from obsidianlink.env.integration.native_runtime import inspect_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hs-err",
        type=Path,
        help="optional captured hs_err_pid log used to identify actually loaded natives",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional new JSON evidence path; existing files are never overwritten",
    )
    args = parser.parse_args()
    payload = inspect_runtime(hs_err_path=args.hs_err)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

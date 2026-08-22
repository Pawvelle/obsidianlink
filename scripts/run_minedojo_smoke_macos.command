#!/bin/zsh
# Start MineDojo from a real Terminal.app session on macOS.
#
# MineDojo's macOS launcher starts Minecraft in a second Terminal.app window
# and ties it to this Python process.  Do not run this through a short-lived
# IDE task runner; closing this terminal intentionally stops Minecraft.

set -e

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"

exec /opt/anaconda3/bin/conda run --no-capture-output -n mc-agent \
  python -m obsidianlink.experiments.run_minedojo_smoke \
  --summary-path logs/minedojo_smoke_latest.json "$@"

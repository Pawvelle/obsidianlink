#!/bin/zsh
# Run one real local-Qwen + MineDojo task from Terminal.app on macOS.

set -e

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"

exec /opt/anaconda3/bin/conda run --no-capture-output -n mc-agent \
  python -m obsidianlink.experiments.run_minedojo_harvest_log \
  --backend qwen \
  --max-steps 240 \
  --max-planning-cycles 10 \
  --output-dir logs/qwen_harvest_latest

#!/bin/zsh
# Run the smallest real MiniMax + MineDojo harvesting trial from Terminal.app.
# The API key is deliberately requested at runtime and is never saved here.

set -e

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"

if [[ -z "${MINIMAX_API_KEY:-}" ]]; then
  read -rs "MINIMAX_API_KEY?MiniMax API key: "
  echo
  export MINIMAX_API_KEY
fi

exec /opt/anaconda3/bin/conda run --no-capture-output -n mc-agent \
  python -m obsidianlink.experiments.run_minedojo_harvest_log \
  --backend minimax \
  --max-steps 80 \
  --max-planning-cycles 5 \
  --output-dir logs/minimax_harvest_latest

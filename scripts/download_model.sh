#!/usr/bin/env bash
set -euo pipefail

REPO_ID="Qwen/Qwen3-VL-2B-Instruct"
REVISION="89644892e4d85e24eaac8bacfd4f463576704203"
LOCAL_DIR="models/Qwen3-VL-2B-Instruct"
EXPECTED_SHA256="7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"

hf download "$REPO_ID" \
  --revision "$REVISION" \
  --local-dir "$LOCAL_DIR" \
  --max-workers 4

ACTUAL_SHA256="$(shasum -a 256 "$LOCAL_DIR/model.safetensors" | awk '{print $1}')"
if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
  echo "Model checksum mismatch: $ACTUAL_SHA256" >&2
  exit 1
fi

echo "Model is ready at $LOCAL_DIR"

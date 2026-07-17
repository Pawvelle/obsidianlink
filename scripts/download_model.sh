#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_FILE="$ROOT_DIR/model.lock.json"
cd "$ROOT_DIR"

REPO_ID="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["repo_id"])' "$LOCK_FILE")"
REVISION="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$LOCK_FILE")"
LOCAL_DIR="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["local_dir"])' "$LOCK_FILE")"
EXPECTED_SHA256="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["files"]["model.safetensors"]["sha256"])' "$LOCK_FILE")"

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

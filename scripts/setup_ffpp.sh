#!/usr/bin/env bash
set -euo pipefail

# Download FaceForensics++ into a local directory using the integrated official downloader.
# Usage: ./scripts/setup_ffpp.sh /path/to/ffpp [dataset] [compression] [num_videos]

OUT_DIR="${1:-./data/ffpp_raw}"
DATASET="${2:-all}"
COMPRESSION="${3:-c23}"
NUM_VIDEOS="${4:-}"

ARGS=("$OUT_DIR" --dataset "$DATASET" --compression "$COMPRESSION" --type videos)
if [[ -n "$NUM_VIDEOS" ]]; then
  ARGS+=(--num_videos "$NUM_VIDEOS")
fi

python scripts/download_faceforensics.py "${ARGS[@]}"

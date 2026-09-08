#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python scripts/download_whisper.py "${1:-large-v3-turbo}"

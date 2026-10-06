#!/usr/bin/env bash
# YOLO26n visible detector plus the existing thermal/radar/fusion stack.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "${1:-}" = "--stop" ] || [ "${1:-}" = "stop" ]; then
    exec bash "$HERE/run_live.sh" "$@"
fi
exec bash "$HERE/run_live.sh" --detect-model yolo26n --detect-backend gpu "$@"

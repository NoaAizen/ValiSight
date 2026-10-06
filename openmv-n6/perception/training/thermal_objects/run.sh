#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE=thermal-objects-train:jp6-v1
DATA="${THERMAL_DATA:-$HOME/datasets/FLIR_thermal_vehicles_yolo}"
MODELS="${THERMAL_MODELS:-$HOME/datasets/thermal-object-models}"
RUNS="${THERMAL_RUNS:-$HOME/datasets/thermal-object-runs}"
NAME=thermal-vehicles-train
ACTION="${1:-status}"
case "$ACTION" in
  build) exec docker build --network=host -t "$IMAGE" "$HERE" ;;
  status)
    docker ps -a --filter "name=^/${NAME}$" --format '{{.Names}} {{.Status}}'
    if [ -f "$RUNS/training-status.json" ]; then cat "$RUNS/training-status.json"; fi
    exit ;;
  logs) exec docker logs --tail 60 -f "$NAME" ;;
  stop) exec docker stop --time 30 "$NAME" ;;
  smoke|start|resume) ;;
  *) echo 'Usage: run.sh build|smoke|start|status|logs|stop|resume' >&2; exit 2 ;;
esac
mkdir -p "$RUNS"
[ -f "$DATA/data.yaml" ] && [ -f "$MODELS/yolov8n.pt" ] || {
  echo 'Missing prepared dataset or initialization weights' >&2; exit 1;
}
EXTRA=()
DOCKER_ARGS=(--runtime=nvidia --network=none --shm-size=512m --cpus=3
  --memory=6g --memory-swap=7g --user "$(id -u):$(id -g)"
  -e HOME=/tmp -e PYTHONUNBUFFERED=1
  -e YOLO_CONFIG_DIR=/results/.ultralytics -e MPLCONFIGDIR=/results/.matplotlib
  -e CUDA_MODULE_LOADING=LAZY -e TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=64
  -v "$HERE:/training:ro" -v "$DATA:$DATA:ro"
  -v "$MODELS:/models:ro" -v "$RUNS:/results")
if [ "$ACTION" = smoke ]; then
  DOCKER_ARGS+=(--rm --name thermal-vehicles-smoke)
  EXTRA+=(--smoke)
else
  if [ "$ACTION" = start ]; then
    python3 - "$RUNS/smoke-status.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    assert json.load(f)['state'] == 'completed', 'Run the smoke test first'
PY
    [ ! -d "$RUNS/baseline" ] || { echo 'Baseline already exists; use resume' >&2; exit 1; }
  else
    [ -f "$RUNS/baseline/weights/last.pt" ] || { echo 'No checkpoint to resume' >&2; exit 1; }
    EXTRA+=(--resume /results/baseline/weights/last.pt)
    if docker container inspect "$NAME" >/dev/null 2>&1; then
      [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" = false ] || {
        echo 'Training is already running' >&2; exit 1;
      }
      docker rm "$NAME"
    fi
  fi
  DOCKER_ARGS+=(-d --name "$NAME")
fi
exec docker run "${DOCKER_ARGS[@]}" "$IMAGE" python3 /training/train.py \
  --data "$DATA/data.yaml" --weights /models/yolov8n.pt --out /results \
  --epochs 30 --batch 1 "${EXTRA[@]}"

#!/usr/bin/env bash
# Launch the ValiSight dev container.
#
#   ./run.sh              # interactive shell in /workspace
#   ./run.sh <cmd...>     # run a command instead of a shell
#
# Serial devices (radar) and video devices (RGB camera) are detected at launch
# and passed through only if they exist, so the container still starts when the
# camera is unplugged.

set -euo pipefail

IMAGE="${VALISIGHT_IMAGE:-valisight}"
WORKSPACE="${VALISIGHT_WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "error: image '$IMAGE' not found - build it first:" >&2
    echo "    cd $(dirname "$0") && docker build -t $IMAGE ." >&2
    exit 1
fi

mkdir -p "$WORKSPACE"

args=(
    --rm -i
    # ROS 2 DDS discovery and shared-memory transport both need the host's
    # namespaces to talk to anything running outside the container.
    --network host
    --ipc host
    -v "$WORKSPACE:/workspace"
    -w /workspace
)

# Allocating a TTY breaks when this is piped or run from a script, so only ask
# for one when stdin really is a terminal.
if [ -t 0 ]; then
    args+=(-t)
fi

# nvidia is already the default runtime on this host, so CUDA needs no flag.

found_serial=()
for dev in /dev/ttyACM* /dev/ttyUSB*; do
    [ -e "$dev" ] || continue
    args+=(--device "$dev")
    found_serial+=("$dev")
done

# /dev/ttyACM* numbering follows enumeration order and swaps between replugs --
# the radar and the OpenMV board have already traded places once. The by-id
# symlinks are derived from VID/PID/serial and stay put, so expose them too.
if [ -d /dev/serial/by-id ]; then
    args+=(-v /dev/serial/by-id:/dev/serial/by-id:ro)
fi

found_video=()
for dev in /dev/video*; do
    [ -e "$dev" ] || continue
    args+=(--device "$dev")
    found_video+=("$dev")
done

# CSI/MIPI cameras go through the Argus daemon rather than /dev/video*.
if [ -S /tmp/argus_socket ]; then
    args+=(-v /tmp/argus_socket:/tmp/argus_socket)
fi

# GUI (matplotlib, open3d, rviz) only works when a display is reachable. Over a
# plain SSH session DISPLAY is unset, so export DISPLAY=:0 and run `xhost +local:`
# on the desktop session first.
if [ -n "${DISPLAY:-}" ]; then
    args+=(-e "DISPLAY=$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix)
else
    echo "note: DISPLAY unset - GUI windows will not work in this container" >&2
fi

echo "image     : $IMAGE"
echo "workspace : $WORKSPACE -> /workspace"
echo "serial    : ${found_serial[*]:-none}"
echo "video     : ${found_video[*]:-none}"
echo

if [ "$#" -eq 0 ]; then
    exec docker run "${args[@]}" "$IMAGE" bash
else
    exec docker run "${args[@]}" "$IMAGE" "$@"
fi

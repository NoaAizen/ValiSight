#!/usr/bin/env bash
# Bring the ValiSight live view up.
#
#   ./live.sh                     # live view, no recording
#   ./live.sh --record            # live view + record a session
#   ./live.sh --cfg cfg/odom_d_fastchirp.cfg --record
#   ./live.sh --camera thermal --record --save-frames thermal
#
# Every argument is passed straight through to src/live_server.py, so anything
# that works there works here. What this adds is a pre-flight: it resolves the
# three USB endpoints through /dev/serial/by-id (ttyACM* numbering swaps between
# replugs and has already traded the radar and the N6 once), says plainly which
# ones are missing, and supplies a --cfg default so a recording can never again
# come out not knowing which chirp config produced it.
#
# Runs on the host. Set VALISIGHT_DOCKER=1 to go through docker/run.sh instead.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/src"
BY_ID=/dev/serial/by-id

# The config sent when the caller does not name one. Stock deliberately: its
# v_max is 0.974 m/s, so ordinary walking crosses the Doppler fold and actually
# exercises the unwrapping. A config with more headroom hides that path rather
# than testing it.
DEFAULT_CFG="stock_iwr1843.cfg"

find_dev() {            # find_dev <glob> -> path, or empty
    local hit
    hit=$(ls "$BY_ID"/$1 2>/dev/null | head -1 || true)
    [ -n "$hit" ] && readlink -f "$hit" || true
}

# --- pre-flight ------------------------------------------------------------
radar_cli=$(find_dev '*XDS110*if00')
radar_dat=$(find_dev '*XDS110*if03')
openmv=$(find_dev '*MicroPython*if00')

printf 'radar CLI  : %s\n'  "${radar_cli:-NOT FOUND}"
printf 'radar DATA : %s\n'  "${radar_dat:-NOT FOUND}"
printf 'openmv     : %s\n'  "${openmv:-NOT FOUND}"

if [ -z "$radar_dat" ]; then
    echo
    echo "note: no radar DATA port - the live view will come up without radar." >&2
    echo "      Check the USB cable and that the EVM is powered." >&2
fi
if [ -z "$openmv" ]; then
    echo
    echo "note: no MicroPython board - no camera frames." >&2
    echo "      A board in DFU/bootloader mode does not enumerate as one." >&2
fi
if [ -z "$radar_dat" ] && [ -z "$openmv" ]; then
    echo >&2
    echo "error: neither sensor is attached; there is nothing to serve." >&2
    exit 1
fi

# --- config ----------------------------------------------------------------
# Only supply the default if the caller did not name a config, and only when the
# CLI port is actually there: sending a cfg is what makes a session self-
# describing, but a missing CLI port is not a reason to refuse to start.
args=("$@")
has_cfg=0
for a in "$@"; do
    [ "$a" = "--cfg" ] && has_cfg=1
    case "$a" in --cfg=*) has_cfg=1 ;; esac
done

if [ "$has_cfg" -eq 0 ]; then
    if [ -n "$radar_cli" ] && [ -f "$SRC/$DEFAULT_CFG" ]; then
        args=(--cfg "$DEFAULT_CFG" "$@")
        printf 'radar cfg  : %s (default)\n' "$DEFAULT_CFG"
    elif [ -n "$radar_dat" ]; then
        echo
        echo "warning: no --cfg and no CLI port. The radar keeps whatever config" >&2
        echo "         was last pushed into it, and a recording made now cannot" >&2
        echo "         say what that was." >&2
    fi
fi

# --- where to point a browser ----------------------------------------------
port=8080
for i in "${!args[@]}"; do
    [ "${args[$i]}" = "--port" ] && port="${args[$((i+1))]}"
    case "${args[$i]}" in --port=*) port="${args[$i]#--port=}" ;; esac
done

if command -v ss >/dev/null && ss -ltn "sport = :$port" 2>/dev/null | grep -q LISTEN; then
    echo >&2
    echo "error: port $port is already in use - another live_server is probably" >&2
    echo "       still running. Stop it, or pass --port <n>." >&2
    exit 1
fi

echo
for ip in $(hostname -I 2>/dev/null); do
    printf 'view       : http://%s:%s/\n' "$ip" "$port"
done
echo

# --- go --------------------------------------------------------------------
# live_server resolves data/ relative to its own file, so it must be run from
# src/ -- see the README.
cd "$SRC"
if [ "${VALISIGHT_DOCKER:-0}" = "1" ]; then
    exec "$ROOT/docker/run.sh" bash -c "cd /workspace/src && exec python3 live_server.py $*"
fi
exec python3 live_server.py "${args[@]}"

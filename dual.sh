#!/usr/bin/env bash
# Bring the dual thermal+RGB view up (view_thermal_rgb.py + n6_dual_stream.py).
#
#   ./dual.sh                              # live browser view, no recording
#   ./dual.sh --record --note "baseline_mm=25; target=marker; plan=A"
#   ./dual.sh --selftest                   # 3 frames from each sensor -> PNG, exit
#   ./dual.sh --record --save-frames thermal --web-port 8082
#
# Every argument is passed straight through to view_thermal_rgb.py. What this
# adds is a pre-flight: it resolves the N6 through /dev/serial/by-id (ttyACM*
# numbering swaps between replugs), defaults to the browser view because the
# rig is operated over SSH (no DISPLAY), and refuses a --record run without a
# --note — a protocol session whose meta has no baseline_mm is one the
# registration analyst cannot use (PROTOCOL_thermal_rgb_registration.md).
#
# The radar panel inside the web view is best-effort: if the IWR1843 is
# attached it appears, if not the panel shows its status. Nothing here fails
# on a missing radar — this is the camera bridge, not the fusion server.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BY_ID=/dev/serial/by-id

find_dev() {            # find_dev <glob> -> path, or empty
    local hit
    hit=$(ls "$BY_ID"/$1 2>/dev/null | head -1 || true)
    [ -n "$hit" ] && readlink -f "$hit" || true
}

# --- pre-flight ------------------------------------------------------------
openmv=$(find_dev '*MicroPython*if00')
radar_dat=$(find_dev '*XDS110*if03')

printf 'openmv     : %s\n' "${openmv:-NOT FOUND}"
printf 'radar DATA : %s\n' "${radar_dat:-not attached (panel will say so)}"

if [ -z "$openmv" ]; then
    echo >&2
    echo "error: no MicroPython board on USB - there is nothing to stream." >&2
    echo "       A board in DFU/bootloader mode does not enumerate as one." >&2
    exit 1
fi

# --- argument scan ---------------------------------------------------------
has_note=0 has_web=0 recording=0 selftest=0 gui_flag=0
port=8081
args=("$@")
for i in "${!args[@]}"; do
    case "${args[$i]}" in
        --note|--note=*)         has_note=1 ;;
        --web)                   has_web=1 ;;
        --record)                recording=1 ;;
        --selftest)              selftest=1 ;;
        --no-gui)                gui_flag=1 ;;
        --web-port)              port="${args[$((i+1))]}" ;;
        --web-port=*)            port="${args[$i]#--web-port=}" ;;
    esac
done

if [ "$recording" -eq 1 ] && [ "$has_note" -eq 0 ]; then
    echo >&2
    echo "error: --record without --note. The session meta must say what was" >&2
    echo "       recorded, at minimum:" >&2
    echo '           --note "baseline_mm=<X>; target=marker; plan=A"' >&2
    echo "       (baseline_mm = measured lens-center distance, see the protocol)" >&2
    exit 1
fi

# Default to the browser view unless the caller chose a mode themselves.
if [ "$selftest" -eq 0 ] && [ "$has_web" -eq 0 ] && [ "$gui_flag" -eq 0 ]; then
    args=(--web "${args[@]+"${args[@]}"}")
    has_web=1
fi

if [ "$has_web" -eq 1 ]; then
    if command -v ss >/dev/null && ss -ltn "sport = :$port" 2>/dev/null | grep -q LISTEN; then
        echo >&2
        echo "error: port $port is already in use - another viewer is probably" >&2
        echo "       still running. Stop it, or pass --web-port <n>." >&2
        exit 1
    fi
    echo
    for ip in $(hostname -I 2>/dev/null); do
        printf 'view       : http://%s:%s/\n' "$ip" "$port"
    done
fi
echo

# --- go --------------------------------------------------------------------
# view_thermal_rgb.py resolves the device script and src/ relative to its own
# file, so it runs from the repo root. --port is the N6 serial port, resolved
# here by-id so a replug renumbering never grabs the radar's ttyACM by mistake.
cd "$ROOT"
exec python3 view_thermal_rgb.py --port "$openmv" "${args[@]+"${args[@]}"}"

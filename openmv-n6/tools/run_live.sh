#!/bin/bash
# Bring up the live viewer with the radar, from a cold or a running state.
#
#   ./run_live.sh                     # visible view, person detection, radar on
#   ./run_live.sh --view fused        # anything here is passed through to live.py
#   ./run_live.sh --record captures/session1   # ...including recording
#   ./run_live.sh --stop              # stop the viewer and park the radar
#   SKIP_CFG=1 ./run_live.sh          # leave the radar's chirp config alone
#   KEEP_RADAR=1 ./run_live.sh --stop # stop the viewer, leave the radar chirping
#   RADAR_CFG=$ROOT/radar/configs/radar_10hz_calib.cfg ./run_live.sh
#       # calibration chirp config (8 dB CFAR: static corner reflector visible
#       # past 3 m). Default is the runtime config (15 dB, person-oriented).
#
# WHY THIS EXISTS: /dev/ttyACM* numbering is not stable on this Jetson. The
# three ports have already swapped once - the N6 came up as ACM2 while the
# radar CLI took ACM0 - and the failure is silent and confusing: live.py sends
# MicroPython to the radar's command parser, which answers "is not recognized
# as a CLI command", and the viewer reports a board timeout while the radar
# reports no frames. Nothing in either message points at the ports. So this
# script resolves every port by USB identity and never by number.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
HTTP="${HTTP:-8088}"

# --- port resolution -------------------------------------------------------
# The N6 enumerates as a MicroPython CDC device. The IWR1843 sits behind an
# XDS110 debug probe that exposes two CDC interfaces off one serial number:
# interface 00 is the CLI at 115200 (where the .cfg goes), interface 03 is the
# data stream at 921600. Matching on the interface number rather than on port
# order is what makes this survive a replug.
find_port() {
    local key="$1" want="$2" p
    for p in /dev/ttyACM*; do
        [ -e "$p" ] || continue
        local props; props="$(udevadm info -q property -n "$p" 2>/dev/null)" || continue
        if grep -q "^ID_VENDOR=$key" <<<"$props" \
           && { [ -z "$want" ] || grep -q "^ID_USB_INTERFACE_NUM=$want\$" <<<"$props"; }; then
            echo "$p"; return 0
        fi
    done
    return 1
}

BOARD="$(find_port MicroPython '' || true)"
RADAR_CLI="$(find_port Texas_Instruments 00 || true)"
RADAR_DATA="$(find_port Texas_Instruments 03 || true)"

[ -n "$BOARD" ]      || { echo "no MicroPython CDC port - is the N6 plugged in and out of DFU?" >&2; exit 1; }
[ -n "$RADAR_DATA" ] || { echo "no XDS110 data port (interface 03) - is the IWR1843 powered?" >&2; exit 1; }
[ -n "$RADAR_CLI" ]  || { echo "no XDS110 CLI port (interface 00)" >&2; exit 1; }

echo "board      $BOARD"
echo "radar CLI  $RADAR_CLI"
echo "radar data $RADAR_DATA"

# --- stop whatever is already holding those ports --------------------------
# capture.py counts too: it and live.py share the board's single CDC endpoint
# and cannot both own it. SIGINT first, not SIGKILL - live.py drains the board
# on the way out, and a killed one leaves the Lepton mid-frame.
VIEWERS="tools/(live|capture)\.py"

stop_viewer() {
    pgrep -f "$VIEWERS" >/dev/null || { echo "no viewer running"; return 0; }
    echo "stopping the running viewer..."
    pkill -INT -f "$VIEWERS" || true
    for _ in $(seq 24); do
        pgrep -f "$VIEWERS" >/dev/null || break
        sleep 0.5
    done
    # A viewer that outlives SIGINT is not a nuisance, it is a trap: it keeps
    # port 8088, the replacement dies in server_bind, and the old process goes
    # on answering /health. Every reading then describes the process you meant
    # to replace - stale flags, missing --record, a radar reader that died
    # hours ago - while the launch looks like it worked. Escalate, then verify.
    if pgrep -f "$VIEWERS" >/dev/null; then
        echo "  it ignored SIGINT; sending SIGKILL"
        pkill -KILL -f "$VIEWERS" || true
        sleep 2
    fi
    if pgrep -f "$VIEWERS" >/dev/null; then
        echo "could not stop the running viewer - kill it by hand" >&2
        return 1
    fi
    # The board needs a moment after its reader lets go, and the HTTP socket
    # needs to leave TIME_WAIT before a new server can claim it.
    for _ in $(seq 20); do
        curl -s --max-time 1 "http://localhost:$HTTP/health" >/dev/null 2>&1 || break
        sleep 0.5
    done
    echo "  viewer stopped"
}

# --- --stop: shut down and exit -------------------------------------------
# Parking the radar too, because a chirping IWR1843 left running draws power
# and heats the board for nothing, and its Doppler history is meaningless by
# the time anyone looks again. KEEP_RADAR=1 leaves it transmitting for anyone
# listening with radar_listen.py directly.
if [ "${1:-}" = "--stop" ] || [ "${1:-}" = "stop" ]; then
    stop_viewer
    if [ "${KEEP_RADAR:-0}" != "1" ]; then
        if python3 "$HERE/send_radar_cfg.py" --port "$RADAR_CLI" --stop >/dev/null 2>&1; then
            echo "  radar stopped"
        else
            echo "  radar did not answer sensorStop - it may already be parked" >&2
        fi
    else
        echo "  radar left chirping (KEEP_RADAR=1)"
    fi
    exit 0
fi

stop_viewer

# --- radar config ----------------------------------------------------------
# The IWR1843 emits nothing at all until it is configured, so skipping this
# makes a healthy radar look like a dead link.
if [ "${SKIP_CFG:-0}" != "1" ]; then
    echo "sending chirp config${RADAR_CFG:+ ($RADAR_CFG)}..."
    python3 "$HERE/send_radar_cfg.py" ${RADAR_CFG:+"$RADAR_CFG"} --port "$RADAR_CLI" >/dev/null \
        || { echo "chirp config failed - rerun send_radar_cfg.py by hand to see why" >&2; exit 1; }
fi

# --- launch ----------------------------------------------------------------
# --radar-hfov 62.7 overrides live.py's 70 deg default with the measured value
# (f = 525 px at 640 wide, caliper checkerboard against a tape measure).
# It only matters as a fallback: when the solved calib below exists, its K
# (and R,t and distortion) replace the guess entirely.
ARGS=(-p "$BOARD" --radar "$RADAR_DATA" --radar-hfov 62.7
      --detect person --view visible --http "$HTTP")

# The solved radar<->RGB extrinsic of 2026-08-18 (full R with pitch, t from
# caliper, K + distortion). Bootstrap._load() understands this schema and
# projects with it; the /set push further down is only the legacy fallback.
CAL_SOLVED="$ROOT/calib-artifacts/radar_rgb_2026-08-18.json"
if [ -f "$CAL_SOLVED" ]; then
    ARGS+=(--radar-calib "$CAL_SOLVED")
    echo "extrinsic   $CAL_SOLVED (solved R,t via --radar-calib)"
fi

# The thermal layer is stretched, not registered, until B2 is shot and solved;
# every temperature it quotes carries (unreg). Pick the LUT up automatically
# the moment it exists so this script does not have to be edited then.
WARP="$ROOT/calib-artifacts/warp.lut"
if [ -f "$WARP" ]; then
    ARGS+=(--warp "$WARP")
    echo "warp        $WARP"
else
    echo "warp        none yet - thermal is stretched, temperatures read (unreg)"
fi

LOG="${LOG:-/tmp/live_radar.log}"
nohup setsid python3 "$HERE/live.py" "${ARGS[@]}" "$@" >"$LOG" 2>&1 </dev/null &
NEW_PID=$!
echo "log         $LOG"

# Catch an immediate death - a bad flag, a missing weights file, a port still
# held - before spending 45 s waiting for frames that will never come.
sleep 3
kill -0 "$NEW_PID" 2>/dev/null \
    || { echo "live.py exited immediately:" >&2; tail -20 "$LOG" >&2; exit 1; }

# --- wait for the stream, then apply the solved extrinsic ------------------
# Board bring-up is ~10 s (drain + Lepton sync), and live.py restarts it on a
# fault, so allow for one retry before giving up.
echo -n "waiting for the first frame"
UP=0
for _ in $(seq 45); do
    if curl -s --max-time 2 "http://localhost:$HTTP/health" 2>/dev/null \
       | grep -q '"name": "stream", "level": "ok"\|"name":"stream","level":"ok"'; then
        UP=1; break
    fi
    echo -n .; sleep 1
done
echo

# Legacy fallback only: when no solved calib file was passed via --radar-calib,
# push the old yaw+t solution (2026-08-10) through /set. With a solved calib
# loaded the /set angles are corrections on top of it - pushing absolute values
# here would corrupt the solution, so skip.
CAL="$ROOT/calib-artifacts/T_camera_radar.json"
if [ ! -f "$CAL_SOLVED" ] && [ -f "$CAL" ]; then
    QS="$(python3 - "$CAL" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
t = c.get("t_m", [0, 0, 0])                      # metres in the file, mm in /set
print("yaw=%.4f&tx=%.1f&ty=%.1f&tz=%.1f"
      % (c.get("yaw_deg", 0.0), t[0]*1000, t[1]*1000, t[2]*1000))
PY
)"
    curl -s -o /dev/null --max-time 5 "http://localhost:$HTTP/set?$QS" \
        && echo "extrinsic   $QS" \
        || echo "extrinsic   FAILED to apply - viewer not answering" >&2
fi

echo
if [ "$UP" = 1 ]; then
    echo "live on http://localhost:$HTTP"
else
    echo "stream did not come up in 45 s - check $LOG" >&2
fi
curl -s --max-time 5 "http://localhost:$HTTP/health" 2>/dev/null | python3 -c '
import json, sys
d = json.load(sys.stdin)
print("health: %s" % d["worst"].upper())
for c in d["checks"]:
    if c["level"] != "ok":
        print("  %-5s %-13s %s" % (c["level"].upper(), c["name"], c["text"][:100]))
' || true

#!/bin/bash
# Bring up the live viewer with the radar, from a cold or a running state.
#
#   ./run_live.sh                     # operator view, person detection, radar on
#   ./run_live.sh --view fused        # anything here is passed through to live.py
#   ./run_live.sh --record captures/session1   # ...including recording
#   ./run_live.sh --ai-channels fusion   # start with only the agreement channel
#   ./run_live.sh --ai-channels none     # students loaded, nothing drawn
#   ./run_live.sh --range 20:45          # override the students' training range
#   NO_STUDENTS=1 ./run_live.sh          # do not load the students at all
#   STUDENT_ENGINES=$ROOT/perception/out/gexport/v4/models ./run_live.sh
#       # run a different export's student engines (build them first: trtexec)
#   MAP=none ./run_live.sh               # launch without the map layer
#   MAP=31.7712,35.2043 ./run_live.sh    # a different site (see MAP_GEOID below)
#   MAP_HEADING=215 ./run_live.sh        # compass heading of the camera, if known
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
# Port args in "$@" come AFTER ours and silently override the USB-resolved
# ports - a bare --radar swings the reader onto whatever DATA_PORT's default
# happens to be, which has already been the BOARD (radar3-walkdepth1,
# 2026-08-24: 3 min of video, zero radar frames). This script owns the ports;
# refuse rather than record a crippled session.
for a in "$@"; do
    case "$a" in
        --radar|--radar=*|-p|--port)
            echo "run_live.sh resolves the ports itself - drop '$a' from the" >&2
            echo "arguments (it would override the detected port and can point" >&2
            echo "the radar reader at the board)" >&2
            exit 1;;
    esac
done
# --radar-hfov 62.7 overrides live.py's 70 deg default with the measured value
# (f = 525 px at 640 wide, caliper checkerboard against a tape measure).
# It only matters as a fallback: when the solved calib below exists, its K
# (and R,t and distortion) replace the guess entirely.
ARGS=(-p "$BOARD" --radar "$RADAR_DATA" --radar-hfov 62.7
      --detect person --view operator --http "$HTTP")

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

# --- the AI channels -------------------------------------------------------
# The three student channels - thermal, radar, and the agreement between them -
# are part of a normal launch, not something to remember to switch on. Two
# things have to be true for them to mean anything, and neither can be decided
# inside live.py:
#
#   --range 0:60  the thermal student was trained on Celsius at exactly this
#                 scale - gexport v2's manifest carries c_per_lsb 60/255 and
#                 tmin 0 for every one of its seven sessions. Auto-range hands
#                 the model a scene-relative unit instead, which does not fail,
#                 it just quietly detects worse. Pinning also puts c_per_lsb in
#                 meta.json, which is what makes a recording exportable at all.
#   the warp LUT  thermal boxes sit on the 160x120 plane until the LUT maps
#                 them onto the picture, and fusion pairs on the visible plane.
#                 With no LUT the thermal channel reports and draws nothing and
#                 fusion has nothing to pair.
#
# Everything past that is live: the three channels switch from the page (keys
# t, r, c) or over /set?ai_thermal=0&ai_radar=1&ai_fusion=1.
# STUDENT_ENGINES picks the export whose engines the viewer loads; the default
# is v2, the only one with engines built on this rig. Point it at another
# export's models/ directory after building its engines with trtexec.
STUDENT_DIR="${STUDENT_ENGINES:-$ROOT/perception/out/gexport/v2/models}"
STUDENT_ENGINE="$STUDENT_DIR/thermal_student.engine"
HAS_RANGE=0
for a in "$@"; do
    case "$a" in --range|--range=*) HAS_RANGE=1;; esac
done
if [ "${NO_STUDENTS:-0}" = "1" ]; then
    echo "students   off (NO_STUDENTS=1)"
elif [ -f "$STUDENT_ENGINE" ]; then
    ARGS+=(--students --students-dir "$STUDENT_DIR")
    if [ "$HAS_RANGE" = 0 ]; then
        ARGS+=(--range 0:60)
        echo "students   thermal + radar + fusion, range pinned 0:60 (training scale)"
    else
        # Their range wins - it comes after ours on the command line - but the
        # students were trained at 0:60 and this is the only place that says so.
        echo "students   thermal + radar + fusion, YOUR --range (trained at 0:60)"
    fi
    [ -f "$WARP" ] || echo "  no warp LUT: the thermal channel cannot be drawn and fusion cannot pair" >&2
else
    echo "students   none - $STUDENT_ENGINE is missing (build it with trtexec)" >&2
fi

# --- the map layer ---------------------------------------------------------
# Part of a normal launch rather than something to remember: without --map
# live.py has no position at all, so the map card and the top-down beside the
# picture are hidden and /map answers 404. There is no other source - this rig
# carries no GNSS, and nothing in the code invents one.
#
# The default is the MEASURED position of the rig's site (Givat Ram), not an
# example. A rig that has been carried somewhere else must say so, because
# every number the layer produces is a lookup at this point: the geoid
# undulation, the ground and surface elevation off a 30 m DEM posting, and the
# building footprints the top-down draws. Being 20 m out moved the ground
# elevation by 5.5 m when it was measured on 2026-08-25.
MAP="${MAP:-31.764559,35.191150}"
MAP_DEFAULT="31.764559,35.191150"
# The geoid assertion belongs to the position, not to the script: 19..20.5 m is
# Jerusalem, and asserting it over a site in another country fails the stage
# for the right reason but the wrong reading. So it rides only with the default
# position unless someone states their own.
MAP_GEOID="${MAP_GEOID:-}"
if [ -z "$MAP_GEOID" ] && [ "$MAP" = "$MAP_DEFAULT" ]; then
    MAP_GEOID="19 20.5"
fi
HAS_MAP=0
for a in "$@"; do
    case "$a" in --map|--map=*) HAS_MAP=1;; esac
done
if [ "$HAS_MAP" = 1 ]; then
    echo "map        YOUR --map (this script's MAP=$MAP ignored)"
elif [ "$MAP" = "none" ] || [ -z "$MAP" ]; then
    echo "map        off (MAP=none) - no map card, no top-down, /map is 404"
else
    ARGS+=(--map "$MAP")
    # An `if` and not `[ ... ] && ...`: under `set -e` a bare test that fails
    # as the whole statement is exactly the kind of thing that kills a launcher
    # silently at the one site where somebody set MAP without MAP_GEOID.
    if [ -n "$MAP_GEOID" ]; then
        ARGS+=(--map-geoid $MAP_GEOID)      # unquoted on purpose: LOW HIGH
    fi
    # Heading has no source on this rig either, and unlike the position there
    # is no defensible default: a guess here is a guess the wall fix would
    # search three sigma around and never announce. Left out unless stated,
    # and said out loud, because /mapfix cannot run without it.
    if [ -n "${MAP_HEADING:-}" ]; then
        ARGS+=(--map-heading "$MAP_HEADING")
        echo "map        $MAP, heading $MAP_HEADING deg${MAP_GEOID:+, geoid $MAP_GEOID m}"
    else
        echo "map        $MAP${MAP_GEOID:+, geoid $MAP_GEOID m} - no heading given, so"
        echo "           the numbers are live but 'fix from walls' cannot run"
        echo "           (MAP_HEADING=DEG, or /set?heading=DEG while it runs)"
    fi
    # Yael's package is not in this repo; map_api looks for it beside us. Say
    # so here rather than letting the stage fail into the card with a stack
    # trace nobody reads.
    [ -f "$ROOT/mapinit/__init__.py" ] || [ -f "$HOME/ValiSight_yael/mapinit/__init__.py" ] \
        || echo "  no mapinit package (looked in $ROOT and $HOME/ValiSight_yael):" \
                "the map layer will report DEPLOYMENT and stay empty" >&2
fi

# Per-user by default: two accounts share this Jetson, and a log left behind by
# one is not writable by the other - the redirect then fails and live.py dies
# before it starts, which reads as "the board is broken" rather than "someone
# else ran this last".
LOG="${LOG:-/tmp/live_radar.$(id -un).log}"
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
# fault, so allow for one retry before giving up. The detector's GPU probe can
# add up to ~20 s of retries after a reboot (detect.py GPU_PROBE_TRIES) before
# frames start, hence 70 and not 45.
echo -n "waiting for the first frame"
UP=0
for _ in $(seq 70); do
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
    echo "stream did not come up in 70 s - check $LOG" >&2
fi

# A viewer that came up without its detector, or on the 10x-slower CPU
# fallback, streams identically to a healthy one - say so here or it is
# discovered hours later in the recordings.
grep -m1 -E 'RUNNING WITHOUT DETECTION|falling back to yolov4-tiny' "$LOG" >&2 || true
# The students are opt-in inside live.py and degrade silently by design: a
# missing engine or a GPU that would not come up leaves the rest of the viewer
# running and prints one line. Unrepeated here, that line is three minutes of
# scrollback away by the time anyone notices the boxes are missing.
grep -m3 -E '^students: ' "$LOG" >&2 || true
# Same reasoning for the map: it initialises on its own thread and never takes
# the viewer down, so a failed layer is one line far up the log and an empty
# card on the page.
grep -m1 -E '^map: ' "$LOG" >&2 || true
curl -s --max-time 5 "http://localhost:$HTTP/health" 2>/dev/null | python3 -c '
import json, sys
raw = sys.stdin.read()
if not raw.strip():
    sys.exit(0)          # viewer dead or unreachable - already reported above
d = json.loads(raw)
print("health: %s" % d["worst"].upper())
for c in d["checks"]:
    if c["level"] != "ok":
        print("  %-5s %-13s %s" % (c["level"].upper(), c["name"], c["text"][:100]))
' || true

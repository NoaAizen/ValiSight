#!/bin/bash
# One B2 pose: aim in a browser, capture a pair, verify it, keep it.
#
#   ./b2_shot.sh --view   # start the viewfinder, then aim at http://localhost:8088
#   ./b2_shot.sh 1        # shoot pose 1 (drops the viewfinder and brings it back)
#   ./b2_shot.sh 2        # shoot pose 2 ...
#   ./b2_shot.sh --all    # re-run detect over every kept pose
#   ./b2_shot.sh --stop   # put the viewfinder away
#
# WHY A SCRIPT: the protocol says verify after every pose, not at the end, and
# the two commands that do it need four things right each time - the board's
# port (which is not stable across replugs, see run_live.sh), the 7x6 pattern,
# the 38.3 mm square, and a filename that does not overwrite the previous pose.
# Getting any of them wrong reads as a bad pose rather than a bad command.
#
# WHY IT OWNS THE VIEWFINDER: live.py and capture.py share the board's single
# CDC endpoint and cannot both hold it, but without a live view the board is
# positioned blind - which is how the first attempt at pose 2 ended up with half
# the board outside the frame. So this stops the viewer, shoots, and puts it
# back. Bring-up measured at 23 s against a 14 s capture.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$(dirname "$HERE")"
DEST="$TOOLS/../captures/b2"
PATTERN="7x6"          # 8x7 squares after the masked row
SQUARE="0.0383"        # caliper, mechanical_measurements.yaml
HTTP=8088
mkdir -p "$DEST"

find_board() {
    local p
    for p in /dev/ttyACM*; do
        [ -e "$p" ] || continue
        if udevadm info -q property -n "$p" 2>/dev/null | grep -q "^ID_VENDOR=MicroPython"; then
            echo "$p"; return 0
        fi
    done
    echo "no MicroPython CDC port - is the N6 plugged in and out of DFU?" >&2
    return 1
}

view_running() { pgrep -f "[l]ive\.py" >/dev/null 2>&1; }

view_stop() {
    view_running || return 0
    # SIGINT, not SIGKILL: live.py drains the board on the way out, and a killed
    # one leaves the Lepton mid-frame.
    pkill -INT -f "[l]ive\.py" || true
    for _ in $(seq 20); do view_running || break; sleep 0.5; done
    if view_running; then pkill -KILL -f "[l]ive\.py" || true; sleep 2; fi
    for _ in $(seq 16); do
        curl -s --max-time 1 "http://localhost:$HTTP/health" >/dev/null 2>&1 || break
        sleep 0.5
    done
}

view_start() {
    view_running && { echo "viewfinder already up on http://localhost:$HTTP"; return 0; }
    local board; board="$(find_board)"
    # Deliberately minimal - no radar, no detector. Those only slow the bring-up,
    # and nothing here needs them.
    nohup setsid python3 "$TOOLS/live.py" -p "$board" --view visible --http "$HTTP" \
        >/tmp/b2view.log 2>&1 </dev/null &
    local t0; t0=$(date +%s)
    for _ in $(seq 60); do
        curl -s --max-time 2 "http://localhost:$HTTP/health" 2>/dev/null \
            | grep -q '"name": "stream", "level": "ok"' && break
        sleep 1
    done
    if view_running; then
        echo "viewfinder up in $(( $(date +%s) - t0 ))s -> http://localhost:$HTTP"
    else
        echo "viewfinder failed to start - see /tmp/b2view.log" >&2
    fi
}

case "${1:-}" in
    --view)  view_start; exit 0 ;;
    --stop)  view_stop; echo "viewfinder stopped"; exit 0 ;;
    --all)   python3 "$HERE/calib.py" detect "$DEST" --pattern "$PATTERN" \
                 --square "$SQUARE" -o "$DEST/corners.json"; exit 0 ;;
esac

POSE="${1:?usage: b2_shot.sh <pose-number> | --view | --stop | --all}"
TAG="$(printf 'p%02d' "$POSE")"

WAS_VIEWING=0
if view_running; then WAS_VIEWING=1; echo "dropping the viewfinder to free the board..."; view_stop; fi
BOARD="$(find_board)"

STAGE="$(mktemp -d)"; ONE="$(mktemp -d)"
trap 'rm -rf "$STAGE" "$ONE"' EXIT
echo "pose $TAG on $BOARD"
python3 "$TOOLS/capture.py" -n 1 --preview -p "$BOARD" -o "$STAGE" >/dev/null

# Keep the pose under its own stem so poses accumulate instead of overwriting.
# detect_dir() globs *.json and derives the raw names from the stem, so any
# prefix works as long as the three files share it.
for f in "$STAGE"/0000*; do mv "$f" "$DEST/${TAG}_$(basename "$f")"; done

cp "$DEST/${TAG}"_0000.json "$DEST/${TAG}"_0000_*.raw "$ONE/"
# Detection is the slow half of a pose, so run it once and reuse the output for
# both the display and the verdict.
REPORT="$(python3 "$HERE/calib.py" detect "$ONE" --pattern "$PATTERN" \
          --square "$SQUARE" -o /dev/null 2>&1 || true)"
sed 's/^/  /' <<<"$REPORT"
OK="$(grep -c ' ok$' <<<"$REPORT" || true)"

echo
if [ "$OK" -ge 1 ]; then
    echo "KEEP  $TAG is usable for stereo"
else
    # Park the reject rather than leave it in DEST. solve() would ignore it
    # anyway, but a rejected pose sitting alongside the good ones makes the
    # running count lie, and the count is the only thing telling you when the
    # session is done.
    mkdir -p "$DEST/rejected"
    mv "$DEST/${TAG}"_0000* "$DEST/rejected/" 2>/dev/null || true
    echo "REDO  $TAG did not give a stereo view - fix the pose and shoot $POSE again"
    echo "      (frame kept for inspection in captures/b2/rejected/)"
fi
echo "usable so far: $(ls "$DEST"/*_0000.json 2>/dev/null | wc -l) of 8 needed"

[ "$WAS_VIEWING" = 1 ] && view_start || true

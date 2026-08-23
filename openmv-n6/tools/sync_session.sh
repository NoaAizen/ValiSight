#!/usr/bin/env bash
# sync_session.sh — push capture sessions to the Pi NAS, verify, flag.
#
# Usage:
#   sync_session.sh                      # sync ALL sessions under the default root
#   sync_session.sh walk1 walk2         # sync specific sessions
#   CAPTURES_ROOT=~/thermal-fusion/captures sync_session.sh   # other root
#
# Rules (see memory/pi-nas-storage-plan):
#   - NEVER records over the network; this runs only AFTER a session is closed.
#   - Verify = full checksum pass (rsync -c). Flag .synced.ok written on the
#     NAS only after verification. Source is NEVER deleted by this script.
#   - A session whose meta.json lacks closed_wall died mid-recording: synced
#     anyway, flagged DIED_MIDRECORDING on the NAS. Legacy sessions without
#     meta.json are synced normally.
#   - NAS disk is the 7.3T WD My Book, exFAT (no POSIX perms, 2 s timestamp
#     granularity) => rsync -rt --modify-window=2, not -a.
set -uo pipefail

REPO_ROOT="$HOME/thermal-fusion"
CAPTURES_ROOT="${CAPTURES_ROOT:-$REPO_ROOT/openmv-n6/captures}"
NAS_USER="hailo"
# eth0 first, wifi second, tailscale last (nas-pi on the valisight tailnet)
NAS_HOSTS=(192.168.1.104 192.168.1.112 100.125.148.82)
NAS_BASE="/mnt/mybook/vailsigth_record"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=5)

# pick first reachable host
NAS=""
for h in "${NAS_HOSTS[@]}"; do
    if ssh "${SSH_OPTS[@]}" "$NAS_USER@$h" true 2>/dev/null; then NAS="$NAS_USER@$h"; break; fi
done
[ -n "$NAS" ] || { echo "sync_session: no NAS host reachable (${NAS_HOSTS[*]}) — data stays local, rerun later" >&2; exit 1; }

REL=$(realpath --relative-to="$REPO_ROOT" "$CAPTURES_ROOT")
DEST_ROOT="$NAS_BASE/thermal-fusion/$REL"
ssh "${SSH_OPTS[@]}" "$NAS" "mkdir -p \"$DEST_ROOT\"" || { echo "sync_session: cannot create $DEST_ROOT" >&2; exit 1; }

# session list: args, or every directory under the root
if [ $# -gt 0 ]; then SESSIONS=("$@"); else
    mapfile -t SESSIONS < <(find "$CAPTURES_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
fi

RSYNC_COMMON=(-rts --modify-window=2 --exclude='.synced.ok' --exclude='DIED_MIDRECORDING' --exclude='*.tmp')
ok=0; fail=0; died=0
for s in "${SESSIONS[@]}"; do
    src="$CAPTURES_ROOT/$s"
    [ -d "$src" ] || { echo "SKIP  $s (not a directory)"; continue; }
    dst="$DEST_ROOT/$s"

    if ! rsync "${RSYNC_COMMON[@]}" "$src/" "$NAS:$dst/"; then
        echo "FAIL  $s (transfer)"; fail=$((fail+1)); continue
    fi
    # full-checksum verify: any line of output means a mismatch
    diff_out=$(rsync "${RSYNC_COMMON[@]}" -c --dry-run --out-format='%n' "$src/" "$NAS:$dst/")
    if [ -n "$diff_out" ]; then
        echo "FAIL  $s (checksum mismatch):"; echo "$diff_out" | sed 's/^/        /'
        fail=$((fail+1)); continue
    fi
    # died-mid-recording flag: meta.json exists but has no closed_wall
    if [ -f "$src/meta.json" ] && ! grep -q closed_wall "$src/meta.json"; then
        ssh "${SSH_OPTS[@]}" "$NAS" "touch \"$dst/DIED_MIDRECORDING\""
        echo "DIED  $s (synced+verified, flagged: no closed_wall in meta.json)"
        died=$((died+1))
    fi
    ssh "${SSH_OPTS[@]}" "$NAS" "date -Is > \"$dst/.synced.ok\""
    touch "$src/.synced.ok"
    echo "OK    $s"
    ok=$((ok+1))
done
echo "---"
echo "synced+verified: $ok   died-flagged: $died   failed: $fail   (NAS: $NAS)"
[ "$fail" -eq 0 ]

"""Frame dimensions and the portable fusion library's host location."""
from pathlib import Path

OUT_W, OUT_H = 640, 400
TH_W, TH_H = 160, 120
LIB = str(Path(__file__).resolve().parents[2] / "host" / "libfusion.so")


# Free heap below this and the board gets restarted before the automatic collector
# can fire. The collector is the hazard, not the memory: a gc.collect() with the
# Lepton up wedges it permanently and no soft re-init recovers it - only a fresh
# csi.CSI() object does, which is precisely what a restart builds.
#
# 4MB against a measured ~208 B/frame is about 19000 frames, 36 minutes, of slack
# after the trigger. Deliberately enormous. The restart costs ~10s of bring-up and
# happens once every few hours, so there is nothing to be gained by cutting it
# fine and a dead stream to be lost by getting it wrong.
HEAP_FLOOR = 4 << 20

# The largest gap between snapshots this part is KNOWN to survive. Measured
# 2026-08-09: plain sleeps of 150/300/600/1000/1500/2500 ms between snapshots
# were all harmless, and a gc.collect() of any length wedges it. 2500ms is
# therefore the edge of the tested envelope, not a limit anyone has found - past
# it there is simply no measurement. The board's own thermal interval is the only
# way to see it, since a gap on this side is indistinguishable from a slow link.
LEPTON_SAFE_GAP_MS = 2500

# Boxes older than this are not drawn. The detector runs at ~10Hz against an
# 8.8Hz stream, so in normal operation a box is at most one frame old; this only
# fires when detection has actually fallen behind or died.
DETECT_STALE_S = 1.0



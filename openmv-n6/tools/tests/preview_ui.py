#!/usr/bin/env python3
"""Serve the live viewer's page with no board attached.

    ./tests/preview_ui.py            then open http://localhost:8089

The page is the real PAGE out of live.py driven by the real handlers, so what
this shows is what the board will show - the only fake thing is where the frames
come from. One recorded pair is fused over and over, and the state dict is filled
with the kind of numbers a healthy session produces, so the panels have something
to draw.

For working on the viewer. Nothing here belongs in a measurement: the timing
window is invented, and the frame is the same frame every time.
"""
import os
import sys
import threading
import time
from http.server import ThreadingHTTPServer

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import live                                  # noqa: E402
from test_live import Args, load_pair, HANDWAVE3   # noqa: E402


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else HANDWAVE3
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8089
    y, thermal, meta = load_pair(d)

    pipe = live.Pipeline(Args())
    lo, hi = meta.get("tmin", 10), meta.get("tmax", 45)
    pipe.set_range(lo, hi)

    state = {
        "range": (lo, hi), "fps": 8.7, "last_frame_t": time.time(),
        # A three-valued delta function, as measured on the board over 5220
        # frames, plus the FFC gap that timing() is supposed to filter out.
        "dt_window": [113, 114, 114, 115, 114, 1824, 114, 113] * 8,
        "skew_window": [11, 12, 12, 13, 12] * 12,
        "last_ffc_t": time.time() - 47, "ffcs": 3,
        "heap_free": 13 << 20, "rendered": 5220, "dropped": 21,
        "torn_window": [0] * 60, "rows_window": [14] * 60,
        "detections": [], "resyncs": 0,
        # The host panel is real even here: this process is the one under
        # test, so the CPU it uses fusing the same frame over and over is
        # exactly what the panel is meant to show.
        "soc": live.hostsoc.Soc(),
        # The AI card is hidden unless the students are loaded, so the preview
        # has to claim they are or the panel cannot be worked on at all. Both
        # sub-objects are stand-ins for engines; nothing here runs one.
        "students": {"thermal": object(), "radar": object(),
                     "th2vis": object()},
        "student_t": time.time(),
        # More found than shown, so the card's "1 of 3" path is on screen and
        # the confidence sliders have something to be about.
        "student_seen": {"thermal": 3, "radar": 2},
        "student_thermal": [{"x": 20, "y": 30, "w": 12, "h": 26, "conf": 0.81,
                             "vis": (80, 120, 48, 104)}],
        "student_radar": [{"x": 96, "y": 130, "w": 44, "h": 100, "conf": 0.63}],
        "student_fused": [{"x": 80, "y": 120, "w": 48, "h": 104, "conf": 0.93,
                           "conf_thermal": 0.81, "conf_radar": 0.63,
                           "du": 6.0, "radar_m": 4.2}],
    }

    # The red path is the one nobody sees until it matters, so it has to be
    # reachable on purpose. These are the failures that mean "do not record".
    if "--fault" in sys.argv:
        state.update({
            "torn_window": [1] * 20 + [0] * 40, "resyncs": 4,
            "last_resync": "header out of step after 2 frames",
            "starved": 2, "last_starve_ms": 3100, "stalls": 5,
            "bad_jpeg": 3, "dropped": 900, "rendered": 1200,
            "heap_free": 5 << 20, "restarts": 2, "last_restart": "heap floor",
            "rows_window": [12, 14, 9],
        })

    def frames():
        """Refuse the picture, keep the panels honest: one fused frame, redone at
        the sensor's own cadence so the page's 1 Hz poll is not the only clock."""
        while True:
            rgb = pipe.process(y, thermal)
            state["frame"] = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
            state["last_frame_t"] = time.time()
            # A slow drain, so the heap trace has something to show.
            state["heap_free"] = max(4 << 20, state["heap_free"] - 6000)
            time.sleep(0.114)

    threading.Thread(target=frames, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", port), live.make_handler(state, pipe))
    print("preview on http://localhost:%d  (frames from %s)" % (port, os.path.basename(d)))
    srv.serve_forever()


if __name__ == "__main__":
    main()

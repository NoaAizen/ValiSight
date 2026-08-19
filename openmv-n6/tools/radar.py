"""IWR1843 radar feed + overlay for live.py.

The radar arrives on its own USB (XDS110), completely independent of the N6
link, so this never touches the thermal/visible stream: a silent or absent
radar just means no markers. Wiring in live.py is three lines - construct a
RadarFeed, hand it to the Streamer via state, call draw() on the outgoing
frame.

What the markers mean, and what they do not:
  - horizontal position IS calibrated: radar azimuth goes through the measured
    radar->thermal fit of 2026-08-05 (yaw 0.92 deg, scale 0.8914, f 147 px on
    the 160-wide thermal frame), then stretches with the thermal layer onto
    the output. When live.py runs with the placeholder warp that stretch is
    exact; with a real --warp it is close but not the warp itself.
  - vertical position is NOT measured. The radar's elevation was never
    calibrated (known debt of the extrinsic artifact), so markers sit in a
    fixed band near the bottom. Range is written next to each marker instead.

If the radar is streaming, points appear within a second. If it is silent
(fresh power-up loses the config), the feed pushes tools/radar.cfg through
the CLI port once after 4s, same recovery live_server uses.
"""
import math
import os
import threading
import time

import cv2

import iwr1843_uart
import radar_gate
import radar_classify_n6

# The measured radar->thermal azimuth fit (radar_thermal_extrinsic_20260805).
YAW_DEG = 0.92
AZ_SCALE = 0.8914
TH_F_PX = 147.0
TH_CX = 80.0
TH_W = 160

# Gating and the static threshold live in radar_gate / radar_classify_n6 —
# the same modules the rest of the project trusts, not local copies of them.


def _by_id(tail):
    import glob
    hits = sorted(glob.glob("/dev/serial/by-id/*XDS110*" + tail))
    return os.path.realpath(hits[0]) if hits else None


class RadarFeed(threading.Thread):
    """Reads TLV frames off the DATA port; keeps the latest clusters.

    Clustering here is one pass of range-binning, not the project's real
    tracker: enough to draw one marker per object instead of thirty per wall.
    Nothing downstream consumes it - display only, like the rest of live.py's
    view layer.
    """

    def __init__(self, cfg=None):
        super().__init__(daemon=True)
        self.cfg = cfg or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "radar.cfg")
        self.clusters = []
        self.tracks = []
        self._next_id = 1
        self.fps = 0.0
        self.status = "starting"
        self.stop = threading.Event()
        self.start()

    # ---- tracking ------------------------------------------------------
    # Single-frame labels flicker: a walker's radial Doppler passes through
    # zero at every turn and on any crossing path, so frame-by-frame he keeps
    # relabeling as static. The aliasing experiment (10/08) showed the cure:
    # DISPLACEMENT is the evidence Doppler cannot fake. Tracks associate
    # clusters over time by position; a track whose position moves is a person
    # even in the frames where its Doppler reads zero, and the label sticks
    # (hysteresis) instead of strobing.

    GATE_M = 1.2          # association gate, loosened for range jitter
    DROP_S = 2.5          # far targets detect sparsely; bridge the gaps
    WALK_MPS = 0.25       # displacement speed that counts as walking
    EMA = 0.4             # position smoothing for a stable box

    def _track(self, clusters, now):
        cand = [c for c in clusters if c["label"] != "static" or True]
        for tr in self.tracks:
            tr["matched"] = False
        for c in cand:
            cx, cy, cz = c["centroid"]
            best, best_d = None, self.GATE_M
            for tr in self.tracks:
                if tr["matched"]:
                    continue
                d = math.hypot(tr["x"] - cx, tr["y"] - cy)
                if d < best_d:
                    best, best_d = tr, d
            if best is None:
                self.tracks.append({
                    "id": self._next_id, "x": cx, "y": cy, "z": cz,
                    "hist": [(now, cx, cy)], "seen": now, "matched": True,
                    "score": 0, "cluster": c, "born": now})
                self._next_id += 1
                continue
            a = self.EMA
            best["x"] = best["x"] * (1 - a) + cx * a
            best["y"] = best["y"] * (1 - a) + cy * a
            best["z"] = best["z"] * (1 - a) + cz * a
            best["hist"].append((now, cx, cy))
            best["hist"] = [h for h in best["hist"] if now - h[0] < 1.0]
            best["seen"] = now
            best["matched"] = True
            best["cluster"] = c

            # displacement speed over the last second - Doppler-independent
            h0 = best["hist"][0]
            dt = now - h0[0]
            disp = math.hypot(cx - h0[1], cy - h0[2]) / dt if dt > 0.3 else 0.0
            f = c  # freshest cluster features
            moving = (disp > self.WALK_MPS or abs(f["doppler_mps"]) > self.WALK_MPS
                      or f["v_spread"] > 0.5)
            # hysteresis: evidence charges the score fast, silence drains slow
            best["score"] = min(best["score"] + (2 if moving else -1), 8)
            best["disp_mps"] = disp

        self.tracks = [t for t in self.tracks if now - t["seen"] < self.DROP_S]

    def people(self):
        """Confirmed person tracks: stable position + sticky label."""
        out = []
        for tr in self.tracks:
            if tr["score"] < 2:
                continue
            c = tr["cluster"]
            if c["extent_m"] > 1.8 and c["n_points"] >= 6:
                continue                     # vehicle-shaped, not a person
            out.append(tr)
        return out

    def run(self):
        import serial
        data_port = _by_id("if03")
        while not data_port and not self.stop.is_set():
            # radar unplugged (rig moved, cable out): wait for it instead of
            # giving up - replugging now recovers without a server restart
            self.status = "no radar on USB - waiting"
            time.sleep(3.0)
            data_port = _by_id("if03")
        if not data_port:
            return
        ser = None
        while ser is None and not self.stop.is_set():
            # a restarting sibling can hold the port for a few seconds; a
            # one-shot open turned that into a dead feed for the whole session
            try:
                ser = serial.Serial(data_port, 921600, timeout=0.05)
            except Exception as e:
                self.status = "port busy, retrying: %s" % str(e)[:40]
                time.sleep(2.0)
        if ser is None:
            return
        ser.reset_input_buffer()
        reader = iwr1843_uart.RadarReader()
        n_total, configured = 0, False
        t0 = win_t = time.time()
        win_n = 0
        while not self.stop.is_set():
            try:
                n = ser.in_waiting
                chunk = ser.read(n if n else 1)
            except Exception as e:
                self.status = "serial died: %s" % e
                return
            for fr in reader.feed(chunk):
                # The project's real pipeline: SNR gate, then the rule-based
                # classifier (static / pedestrian / vehicle / unknown).
                keep, _ = radar_gate.gate_points(fr["points"], fr["snr"],
                                                 fr["noise"])
                self.clusters = radar_classify_n6.classify_frame(keep)[:8]
                self._track(self.clusters, time.time())
                n_total += 1
                win_n += 1
            now = time.time()
            if now - win_t >= 1.0:
                self.fps, win_t, win_n = win_n / (now - win_t), now, 0
                if n_total:
                    self.status = "ok"
            if not n_total and not configured and now - t0 > 4.0:
                # Fresh power-up: the demo firmware forgets its config and sits
                # silent until someone speaks CLI to it.
                configured = True
                cli = _by_id("if00")
                if cli and os.path.exists(self.cfg):
                    self.status = "pushing config"
                    try:
                        with serial.Serial(cli, 115200, timeout=1) as cs:
                            iwr1843_uart.send_config(cs, self.cfg, verbose=False)
                        self.status = "config sent"
                    except Exception as e:
                        self.status = "config failed: %s" % str(e)[:50]
                else:
                    self.status = "silent, no CLI port"


# Marker style per class (BGR). The label IS the point of the overlay, so it is
# drawn next to every marker rather than encoded in colour alone.
STYLE = {
    "pedestrian": ((80, 220, 80), "PERSON"),
    "vehicle": ((60, 140, 255), "VEHICLE"),
    "static": ((150, 150, 150), "static"),
    "unknown": ((200, 200, 200), "?"),
}

TH_CY = 60.0            # thermal mid-row; elevation zero lands here
PERSON_H_M = 1.7        # assumed person height for the box
PERSON_W_M = 0.6        # assumed person width
VEHICLE_H_M = 1.5
VEH_BOX_EXTENT = 1.8    # tracked object wider than this draws as a vehicle

# Radar-only bounding boxes. Everything here comes from the radar: azimuth and
# range through the measured 2026-08-05 fit, elevation from the cluster's own
# z (the radar measures it, +-40 deg FOV - never calibrated against the camera,
# so expect the box to sit high or low before it sits wrong left-right), and
# the box SIZE from an assumed physical height/width divided by range - the
# radar's extent smears with motion (see radar_classify_n6), so a fixed human
# height projects more honestly than the measured extent does.


def _project(c):
    """Cluster -> (col, row) on the 160x120 thermal grid, or None."""
    cx, cy, cz = c["centroid"]
    az = math.degrees(math.atan2(-cy, cx))
    az_cam = (az - YAW_DEG) / AZ_SCALE
    col = TH_CX - TH_F_PX * math.tan(math.radians(az_cam))
    ground = math.hypot(cx, cy)
    if ground < 0.1 or not (0 <= col < TH_W):
        return None
    el = math.atan2(cz, ground)
    # Elevation fit from radar_thermal_elevation_solve.py, SMOKE-TEST grade
    # (indoor 05/08 holds, rms 7.9 rows): row = A + B*tan(el). Replaces the
    # naive f=147 assumption, which the fit shows overshoots ~3x. Redo
    # outdoors per CALIBRATION_PLAN stage C and update these two numbers.
    EL_A, EL_B = 52.4, -53.5
    row = EL_A + EL_B * math.tan(el)
    return col, min(max(row, 0.0), 119.0)


def draw(img, feed):
    """Radar-derived boxes + class labels onto the outgoing frame."""
    if feed is None:
        return img
    h, w = img.shape[:2]
    kx, ky = w / float(TH_W), h / 120.0
    put = lambda t, org, col, s=0.5: (
        cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 0), 3, cv2.LINE_AA),
        cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, s, col, 1, cv2.LINE_AA))
    put("radar: %s %.0f fps" % (feed.status, feed.fps), (8, h - 10), (255, 200, 80))

    # quiet gray dots for the static clutter, straight off the current frame
    for c in feed.clusters:
        if c["label"] != "static":
            continue
        pr = _project(c)
        if pr is None:
            continue
        x, y = int(pr[0] * kx), int(pr[1] * ky)
        cv2.circle(img, (x, y), 5, STYLE["static"][0], 2, cv2.LINE_AA)
        put("%.1fm" % c["range_m"], (x + 8, y + 4), STYLE["static"][0])

    # people come from the TRACKER: smoothed position, sticky label, and a
    # displacement speed that stays honest when the Doppler folds or nulls
    for tr in feed.people():
        rng = math.hypot(tr["x"], tr["y"])
        fake = {"centroid": (tr["x"], tr["y"], tr["z"])}
        pr = _project(fake)
        if pr is None or rng < 0.2:
            continue
        x, y = int(pr[0] * kx), int(pr[1] * ky)
        c = tr["cluster"]
        is_vehicle = c["extent_m"] > VEH_BOX_EXTENT and c["n_points"] >= 6
        color, label = STYLE["vehicle" if is_vehicle else "pedestrian"]
        size_h = VEHICLE_H_M if is_vehicle else PERSON_H_M
        size_w = max(c["extent_m"], PERSON_W_M)
        bh = TH_F_PX * size_h / rng * ky
        bw = TH_F_PX * size_w / rng * kx
        p1 = (int(x - bw / 2), int(y - bh / 2))
        p2 = (int(x + bw / 2), int(y + bh / 2))
        cv2.rectangle(img, p1, p2, color, 2)
        speed = tr.get("disp_mps", 0.0)
        if abs(c["doppler_mps"]) > speed:
            speed = abs(c["doppler_mps"])
        put("%s %.1fm %.1fm/s" % (label, rng, speed),
            (max(p1[0], 2), max(p1[1] - 8, 16)), color, 0.6)
    return img

"""
Live camera + IWR1843BOOST fusion — real-time version of photo_to_radar.py.

Produces the same overlay as the *_radar_expected.jpg preview, but with REAL
radar measurements while the camera runs:

  camera frame -> YOLO boxes
  radar UART   -> point cloud -> clusters (static/pedestrian/... + material)
  match clusters to boxes by azimuth+range -> draw  "sofa->static 1.5m"

The fusion logic itself (geometry, matching, pipeline, low-light) lives in
fusion_core.py as testable classes; this file owns the hardware and the UI:

  Detector         YOLOv4-tiny wrapper (loaded once)
  RadarThread      DATA-UART reader with a sliding point window
  SessionLogger    logs/session_*/ recording for offline analysis
  OverlayRenderer  draws boxes/labels/status on the frame
  LiveApp          wires everything: setup, main loop, shutdown

Hardware setup:
  - IWR1843BOOST flashed with the mmWave SDK 3.x out-of-box demo
    (SW1 in functional mode), USB to PC -> two XDS110 COM ports appear.
  - Camera mounted on/next to the radar, pointing the same way (boresight
    aligned; use --yaw-offset if the radar is rotated vs the camera).

Usage:
    python live_radar_camera.py                       # auto-detect COM ports
    python live_radar_camera.py --cfg-port COM4 --data-port COM5
    python live_radar_camera.py --cam 1 --hfov 70
    python live_radar_camera.py --no-radar            # camera-only test mode

Keys:  q = quit,  s = save annotated screenshot
"""
import argparse
import collections
import json
import math
import os
import sys
import threading
import time

import cv2
import numpy as np

try:
    import msvcrt                      # Windows console keys (q/s in the cmd)
except ImportError:
    msvcrt = None

WIN_TITLE = "VailSight live scan"

# Hebrew project path + cp1252 console = UnicodeEncodeError on print
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fusion_core import (CameraGeometry, ClusterMatcher, RadarPipeline,
                         LowLightEnhancer, DARK_CONF_THR)
from radar_material import METAL_DB, FABRIC_DB, MATERIAL_MAX_RANGE_M
from photo_to_radar import (COCO, PROFILES, YOLO_CFG, YOLO_WEIGHTS, YOLO_SIZE,
                            CONF_THR, NMS_THR, imwrite_unicode)
from iwr1843_uart import RadarReader, send_config, find_com_ports

POINT_WINDOW_S = 0.5     # aggregate radar points over this window (stabilises
                         # sparse static returns before clustering)
EMA_ALPHA = 0.35         # per-object range smoothing


def _cfg_lines(path):
    """Command lines of the chirp config, for embedding into meta.json —
    the cfg NAME alone stops meaning anything once configs/*.cfg is edited."""
    try:
        with open(path) as f:
            return [l.strip() for l in f
                    if l.strip() and not l.strip().startswith("%")]
    except OSError:
        return None


def _git_rev():
    """Short git commit of the code that recorded the session (best-effort)."""
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL, timeout=5).decode().strip()
    except Exception:
        return None


class Detector:
    """YOLOv4-tiny, loaded once (photo_to_radar.detect_objects reloads per call)."""

    def __init__(self):
        if not (os.path.isfile(YOLO_CFG) and os.path.isfile(YOLO_WEIGHTS)):
            sys.exit("YOLO model files missing in models/ folder.")
        # readNetFromDarknet fails on non-ASCII (Hebrew) paths — copy the
        # model files to an ASCII temp dir and load from there
        import shutil, tempfile
        tmp = os.path.join(tempfile.gettempdir(), "vailsight_yolo")
        os.makedirs(tmp, exist_ok=True)
        cfg_p = os.path.join(tmp, "yolov4-tiny.cfg")
        w_p = os.path.join(tmp, "yolov4-tiny.weights")
        for src, dst in ((YOLO_CFG, cfg_p), (YOLO_WEIGHTS, w_p)):
            if (not os.path.isfile(dst)
                    or os.path.getsize(dst) != os.path.getsize(src)):
                shutil.copyfile(src, dst)
        self.net = cv2.dnn.readNetFromDarknet(cfg_p, w_p)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.out_names = self.net.getUnconnectedOutLayersNames()

    def detect(self, img, conf_thr=CONF_THR):
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (YOLO_SIZE, YOLO_SIZE),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.out_names)
        H, W = img.shape[:2]
        boxes, confs, ids = [], [], []
        for out in outs:
            for det in out:
                scores = det[5:]
                cid = int(np.argmax(scores))
                # P(class AND object) = class score * objectness; class score
                # alone overstates confidence, worst in the dark where the
                # threshold is already relaxed to DARK_CONF_THR
                conf = float(scores[cid]) * float(det[4])
                if conf < conf_thr:
                    continue
                cx, cy, bw, bh = det[0] * W, det[1] * H, det[2] * W, det[3] * H
                boxes.append([int(cx - bw / 2), int(cy - bh / 2),
                              int(bw), int(bh)])
                confs.append(conf)
                ids.append(cid)
        keep = cv2.dnn.NMSBoxes(boxes, confs, conf_thr, NMS_THR)
        dets = []
        for i in np.array(keep).flatten() if len(keep) else []:
            label = COCO[ids[i]]
            if label in PROFILES:            # only radar-relevant classes
                dets.append({"label": label, "conf": confs[i],
                             "box": tuple(boxes[i])})
        return dets


class SessionLogger:
    """Records everything needed for offline analysis.

    Session folder layout (logs/session_YYYYmmdd_HHMMSS/):
      meta.json    — run parameters, start/end time, counters
      radar.jsonl  — one line per radar frame: t, frame no,
                     points [x,y,z,v,snr,noise]
      fusion.jsonl — one line per processed camera frame: t, detections,
                     clusters (with material/reflectivity), assignments
      frames/      — periodic annotated snapshots (and on 's' keypress)
    """

    def __init__(self, root, meta, enabled=True, snapshot_every_s=5.0):
        self.enabled = enabled
        self.n_radar = self.n_fusion = self.n_snap = 0
        if not enabled:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(root, "logs", "session_" + stamp)
        self.frames_dir = os.path.join(self.dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.meta = dict(meta, start_time=time.time(), start_stamp=stamp)
        self.radar_f = open(os.path.join(self.dir, "radar.jsonl"), "w")
        self.fusion_f = open(os.path.join(self.dir, "fusion.jsonl"), "w")
        self.lock = threading.Lock()
        self.snapshot_every_s = snapshot_every_s
        self._last_snap = 0.0
        # write meta NOW, not only on clean shutdown — a hard exit used to
        # leave sessions with data but no meta.json at all
        self.write_meta()
        print("Logging session to:", self.dir)

    def write_meta(self):
        if not self.enabled:
            return
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2)

    def log_radar(self, t, frame_no, points):
        if not self.enabled:
            return
        rec = {"t": round(t, 3), "frame": frame_no,
               "points": [[round(v, 3) if v is not None else None for v in p]
                          for p in points]}
        with self.lock:
            self.radar_f.write(json.dumps(rec) + "\n")
            self.n_radar += 1

    def log_fusion(self, t, dets, clusters, assigned):
        if not self.enabled:
            return
        cl_out = [{k: v for k, v in c.items() if k != "points"}
                  for c in clusters]
        rec = {"t": round(t, 3),
               "detections": [{"label": d["label"],
                               "conf": round(float(d["conf"]), 2),
                               "box": [int(v) for v in d["box"]]}
                              for d in dets],
               "clusters": cl_out,
               "assigned": [clusters.index(a) if a in clusters else None
                            for a in assigned]}
        with self.lock:
            self.fusion_f.write(json.dumps(rec) + "\n")
            self.n_fusion += 1

    def log_tracks(self, signatures):
        """One row per physical object seen this session -> tracks.jsonl.
        With meta 'label' these rows are ready-made ML training samples."""
        if not self.enabled or not signatures:
            return
        label = self.meta.get("label")
        with open(os.path.join(self.dir, "tracks.jsonl"), "w") as f:
            for s in signatures:
                f.write(json.dumps(dict(s, session_label=label)) + "\n")

    def maybe_snapshot(self, img, force=False):
        if not self.enabled:
            return None
        now = time.time()
        if not force and now - self._last_snap < self.snapshot_every_s:
            return None
        self._last_snap = now
        path = os.path.join(self.frames_dir, "%.3f.jpg" % now)
        imwrite_unicode(path, img)
        self.n_snap += 1
        return path

    def close(self):
        if not self.enabled:
            return
        self.meta.update(end_time=time.time(), radar_frames=self.n_radar,
                         fusion_frames=self.n_fusion, snapshots=self.n_snap)
        self.write_meta()
        self.radar_f.close()
        self.fusion_f.close()
        print("Session saved: %s  (radar %d | fusion %d | snapshots %d)" %
              (self.dir, self.n_radar, self.n_fusion, self.n_snap))


class RadarThread(threading.Thread):
    """Reads the DATA UART continuously; keeps a sliding window of points."""

    def __init__(self, port_name, logger=None):
        super().__init__(daemon=True)
        import serial
        self.ser = serial.Serial(port_name, 921600, timeout=0.05)
        self.reader = RadarReader()
        self.lock = threading.Lock()
        self.window = collections.deque()      # (timestamp, points)
        self.frames_rx = 0
        self.running = True
        self.logger = logger

    def run(self):
        while self.running:
            try:
                data = self.ser.read(4096)
            except OSError:            # SerialException is an OSError
                # USB unplug / port closed mid-read: a daemon thread dying
                # silently leaves the app blind with "radar: OK" on screen
                if self.running:
                    print("Radar UART read failed - radar thread stopped "
                          "(unplugged?)")
                self.running = False
                break
            for fr in self.reader.feed(data):
                now = time.time()
                # attach per-point SNR + noise (TLV 7) -> 6-tuples; snr+noise
                # is the absolute signal level the material scoring needs
                snr = fr["snr"] or []
                noise = fr.get("noise") or []
                pts = [p + (snr[i] if i < len(snr) else None,
                            noise[i] if i < len(noise) else None)
                       for i, p in enumerate(fr["points"])]
                if self.logger:
                    self.logger.log_radar(now, fr["frame"], pts)
                with self.lock:
                    self.frames_rx += 1
                    self.window.append((now, pts))
                    while self.window and now - self.window[0][0] > POINT_WINDOW_S:
                        self.window.popleft()

    def recent_points(self):
        cutoff = time.time() - POINT_WINDOW_S
        with self.lock:
            pts = []
            for ts, p in self.window:
                if ts >= cutoff:
                    pts.extend(p)
            return pts, self.frames_rx

    def stop(self):
        self.running = False
        try:
            self.ser.close()
        except Exception:
            pass


class OverlayRenderer:
    """Draws radar tracks, camera detections and status on the frame."""

    COLORS = {"pedestrian": (60, 200, 60), "person": (60, 200, 60),
              "vehicle": (60, 120, 255), "object": (0, 180, 255),
              "static": (220, 220, 220), "unknown": (0, 220, 220),
              "camera-only": (110, 110, 110),
              "tree": (60, 180, 20), "ground": (40, 90, 150)}

    def __init__(self, geometry):
        self.geo = geometry
        self.ranges_ema = {}            # per-object range display smoothing

    def radar_only(self, img, clusters, assigned):
        """Draw radar tracks that matched no camera box as projected BOUNDING
        BOXES: width from the lateral extent, height from the vertical extent.

        Non-moving objects (static / tree / ground) and people get bright
        thick boxes; jittery 'unknown' tracks are thin and dim."""
        H, W = img.shape[:2]
        fx = self.geo.fx(W)
        taken = {id(c) for c in assigned if c is not None}
        for c in clusters:
            if id(c) in taken or c["n_points"] < 3:
                continue
            pos = self.geo.project(c["centroid"], img.shape)
            if pos is None:
                continue
            px, py = pos

            # physical size -> pixel box (lateral = width, vertical = height)
            ext = c.get("ext_xyz") or (c.get("extent_m", 0.5),) * 3
            rng = max(c["range_m"], 0.3)
            hw = int(min(max(fx * max(ext[1], 0.35) / (2.0 * rng), 14), W // 3))
            hh = int(min(max(fx * max(ext[2], 0.35) / (2.0 * rng), 14), H // 2))
            x0, y0 = max(px - hw, 0), max(py - hh, 0)
            x1, y1 = min(px + hw, W - 1), min(py + hh, H - 1)

            name = c.get("semantic") or c["label"]
            col = self.COLORS.get(name, (255, 255, 255))
            emphasized = name in ("static", "tree", "ground",
                                  "person", "pedestrian")
            if name == "unknown":
                col = (110, 110, 110)         # de-emphasize jitter
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 2 if emphasized else 1)

            tid = ("T%d " % c["track_id"]) if "track_id" in c else ""
            txt = "%s%s %.1fm" % (tid, name, c["range_m"])
            mat = c.get("material")
            if mat in ("metal", "fabric"):
                txt += " [%s]" % mat
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.45, 1)
            tx = min(max(x0, 4), W - tw - 4)
            ty = y0 - 5 if y0 - th - 8 > 0 else y0 + th + 5
            # filled backdrop keeps labels readable over busy scenes
            cv2.rectangle(img, (tx - 2, ty - th - 3), (tx + tw + 2, ty + 3),
                          (0, 0, 0), -1)
            cv2.putText(img, txt, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    def detections(self, img, dets, assigned):
        """Draw the camera boxes with their matched radar state + range."""
        for d, cl in zip(dets, assigned):
            bx, by, bw, bh = d["box"]
            if cl is not None:
                state = cl["label"]
                key = "%s@%d" % (d["label"], round(bx / 40))  # coarse pos key
                prev = self.ranges_ema.get(key)
                rng = cl["range_m"] if prev is None else \
                    (1 - EMA_ALPHA) * prev + EMA_ALPHA * cl["range_m"]
                if len(self.ranges_ema) > 256:   # bound the smoothing cache
                    self.ranges_ema.clear()      # (slow leak on long runs)
                self.ranges_ema[key] = rng
                txt = "%s->%s %.1fm" % (d["label"], state, rng)
                mat = cl.get("material")
                if mat in ("metal", "fabric"):
                    txt += " [%s]" % mat
            else:
                state = "camera-only"
                txt = "%s (no radar return)" % d["label"]
            c = self.COLORS.get(state, (255, 255, 255))
            cv2.rectangle(img, (bx, by), (bx + bw, by + bh), c, 2)
            cv2.putText(img, txt, (max(bx, 4), max(by - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)

    def status(self, img, fps, radar_ok, n_pts, gamma=1.0):
        txt = "FPS %.1f | radar: %s | pts in window: %d" % (
            fps, "OK" if radar_ok else "NO DATA", n_pts)
        cv2.putText(img, txt, (8, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (60, 220, 60) if radar_ok else (60, 60, 230), 1)
        if gamma < 1.0:
            cv2.putText(img, "LOW LIGHT gamma %.2f - radar keeps identity"
                        % gamma, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (60, 200, 255), 1)


# module-level alias kept for scripts that import COLORS directly
COLORS = OverlayRenderer.COLORS


class LiveApp:
    """Owns the hardware + UI and runs the capture/fusion/draw loop."""

    def __init__(self, args):
        self.args = args
        self.geo = CameraGeometry(args.hfov, args.yaw_offset)
        self.pipeline = RadarPipeline(self.geo, args.radar_height,
                                      args.metal_db, args.fabric_db,
                                      args.material_max_range)
        self.matcher = ClusterMatcher(self.geo)
        self.night = LowLightEnhancer()
        self.renderer = OverlayRenderer(self.geo)
        self.logger = SessionLogger(
            os.path.dirname(os.path.abspath(__file__)),
            meta={"cam": args.cam, "hfov": args.hfov,
                  "yaw_offset": args.yaw_offset, "metal_db": args.metal_db,
                  "fabric_db": args.fabric_db,
                  "material_max_range": args.material_max_range,
                  "cfg": os.path.basename(args.cfg),
                  "cfg_lines": _cfg_lines(args.cfg),
                  "git_rev": _git_rev(),
                  "no_radar": args.no_radar, "label": args.label,
                  "radar_height": args.radar_height},
            enabled=not args.no_log,
            snapshot_every_s=args.snap_every if args.snap_every > 0 else 1e12)
        self.radar = None
        self.dca = self.dca_cap = None
        self.cap = None
        self.detector = None

    # -- setup ---------------------------------------------------------------
    def open_radar(self):
        if self.args.no_radar:
            return
        import serial
        a = self.args
        cfg_port, data_port = a.cfg_port, a.data_port
        if not (cfg_port and data_port):
            auto_cfg, auto_data = find_com_ports()
            cfg_port = cfg_port or auto_cfg
            data_port = data_port or auto_data
        if not data_port:
            sys.exit("Radar COM ports not found. Plug in the IWR1843BOOST or "
                     "pass --cfg-port/--data-port (or use --no-radar).")
        print("Radar ports: CONFIG=%s  DATA=%s" % (cfg_port, data_port))

        line_map = self._open_dca() if a.dca else None

        if not a.no_send_cfg:
            if not cfg_port:
                sys.exit("CONFIG port not found; pass --cfg-port or "
                         "--no-send-cfg.")
            cs = serial.Serial(cfg_port, 115200, timeout=0.3)
            try:
                # record the firmware version — a session is not reproducible
                # without knowing what firmware produced it
                cs.write(b"version\n")
                time.sleep(0.4)
                ver = cs.read(512).decode("ascii", "ignore").strip()
                if ver and self.logger.enabled:
                    self.logger.meta["radar_firmware"] = " ".join(ver.split())
                    self.logger.write_meta()
                print("Sending chirp config: %s" % a.cfg)
                send_config(cs, a.cfg, line_map=line_map)
            finally:
                cs.close()
        self.radar = RadarThread(data_port, logger=self.logger)
        self.radar.start()

    def _open_dca(self):
        """FPGA raw capture: connect + arm the DCA1000 BEFORE sensorStart
        (the record must be running when the first LVDS chirp arrives), and
        flip the LVDS HW session on in the streamed config."""
        import dca1000
        a = self.args
        self.dca = dca1000.DCA1000(a.dca_ip or dca1000.FPGA_IP)
        self.dca.connect()
        print("DCA1000 FPGA connected, version %s" % self.dca.fpga_version())
        self.dca.configure()
        raw_dir = self.logger.dir if self.logger.enabled else \
            os.path.dirname(os.path.abspath(__file__))
        raw_path = os.path.join(raw_dir, "adc_raw.bin")
        self.dca_cap = dca1000.RawCapture(raw_path)
        self.dca_cap.start()
        self.dca.start_record()
        print("FPGA raw capture -> %s" % raw_path)
        if self.logger.enabled:
            self.logger.meta.update(
                dca=True, fpga_ip=a.dca_ip or dca1000.FPGA_IP,
                adc_frame_bytes=dca1000.frame_bytes_from_cfg(a.cfg))
        return lambda l: ("lvdsStreamCfg -1 0 1 0"
                          if l.startswith("lvdsStreamCfg") else l)

    def open_camera(self):
        self.cap = cv2.VideoCapture(self.args.cam, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            sys.exit("Cannot open camera index %d" % self.args.cam)
        for _ in range(20):          # auto-exposure warm-up (first frames dark)
            self.cap.read()
        self.detector = Detector()

    # -- per-frame -----------------------------------------------------------
    def process_frame(self, img, dets, frame_i):
        """One fusion step. Returns (img, dets, tclusters, assigned, extras)."""
        a = self.args
        img, gamma, mean_b = self.night.enhance(img)

        # N=1 must run every frame: frame_i % 1 is always 0, never 1, so
        # compare against 1 % N (0 when N=1, 1 otherwise)
        de = max(a.detect_every, 1)
        if frame_i % de == 1 % de or not dets:
            dets = self.detector.detect(
                img, self.night.conf_threshold(mean_b, CONF_THR))

        pts, rx = (self.radar.recent_points() if self.radar else ([], 0))
        tclusters = self.pipeline.process(
            pts, time.time(),
            img if self.night.color_checks_usable(mean_b) else None)

        if self.radar is not None:
            assigned = self.matcher.match(dets, tclusters, img.shape)
            self.pipeline.note_camera_matches(dets, assigned)
        else:
            # camera-only: everything unmatched -> pinhole fallback labels
            assigned = [None] * len(dets)
        return img, dets, tclusters, assigned, (gamma, pts, rx)

    def _debug_refl(self, tclusters, frame_i):
        if not (self.args.debug_refl and tclusters and frame_i % 15 == 0):
            return
        for c in tclusters:
            print("T%-3d refl %-7s %5s dB  %s %.1fm  az %+.0f  n=%d" % (
                c["track_id"], c["material"],
                "-" if c["refl_db"] is None else "%.1f" % c["refl_db"],
                c["label"], c["range_m"],
                math.degrees(math.atan2(-c["centroid"][1],
                                        max(c["centroid"][0], 0.01))),
                c["n_points"]))

    def _poll_keys(self):
        """Keys work in BOTH the video window (waitKey) and the console."""
        k = cv2.waitKey(1) & 0xFF
        if msvcrt and msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch in (b"q", b"Q"):
                k = ord("q")
            elif ch in (b"s", b"S"):
                k = ord("s")
        return k

    def _save_screenshot(self, img):
        out = self.logger.maybe_snapshot(img, force=True) \
            if self.logger.enabled \
            else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "live_scan_%d.jpg" % int(time.time()))
        if not self.logger.enabled:
            imwrite_unicode(out, img)
        print("Saved:", out)

    # -- main loop -------------------------------------------------------------
    def run(self):
        self.open_radar()
        self.open_camera()
        dets = []
        frame_i, t_last, fps, last_rx = 0, time.time(), 0.0, 0
        last_rx_t = 0.0                 # when the frame counter last advanced
        print("Running. q = quit, s = save screenshot.")
        t_start = time.time()
        try:
            while True:
                if (self.args.max_seconds
                        and time.time() - t_start > self.args.max_seconds):
                    print("Reached --max-seconds (%.0fs), stopping."
                          % self.args.max_seconds)
                    break
                ok, img = self.cap.read()
                if not ok:
                    break
                frame_i += 1

                img, dets, tclusters, assigned, (gamma, pts, rx) = \
                    self.process_frame(img, dets, frame_i)
                self._debug_refl(tclusters, frame_i)

                now = time.time()
                # "radar: OK" means frames arrived within the last second —
                # a latched counter comparison stays green forever after the
                # first frame, even with the USB cable pulled
                if self.radar is not None and rx > last_rx:
                    last_rx, last_rx_t = rx, now
                radar_ok = self.radar is not None and now - last_rx_t < 1.0
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_last, 1e-3))
                t_last = now
                self.logger.log_fusion(now, dets, tclusters, assigned)

                self.renderer.radar_only(img, tclusters, assigned)
                self.renderer.detections(img, dets, assigned)
                self.renderer.status(img, fps,
                                     radar_ok or self.args.no_radar,
                                     len(pts), gamma)
                cv2.imshow(WIN_TITLE, img)
                self.logger.maybe_snapshot(img)

                k = self._poll_keys()
                # closing the video window with X also stops cleanly
                try:
                    if cv2.getWindowProperty(WIN_TITLE,
                                             cv2.WND_PROP_VISIBLE) < 1:
                        print("Window closed, stopping.")
                        break
                except cv2.error:
                    break
                if k in (ord("q"), ord("Q")):
                    break
                if k == ord("s"):
                    self._save_screenshot(img)
        finally:
            self.close()

    def close(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        if self.radar:
            self.radar.stop()
        if self.dca is not None:
            self.dca.stop_record()
            self.dca_cap.stop()
            self.dca.close()
            s = self.dca_cap.stats()
            print("FPGA raw capture: %d packets, %d drop gaps "
                  "(%d repaired late), %.2f MB"
                  % (s["packets"], s["drop_gaps"],
                     s.get("late_backfilled", 0), s["bytes"] / 1e6))
            if self.logger.enabled:
                self.logger.meta["adc_capture"] = s
        self.logger.log_tracks(self.pipeline.signatures())
        self.logger.close()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Live camera + IWR1843 fusion")
    ap.add_argument("--cam", type=int, default=0, help="camera index")
    ap.add_argument("--hfov", type=float, default=60.0,
                    help="camera horizontal FOV in degrees")
    ap.add_argument("--cfg-port", help="radar CONFIG COM port (115200)")
    ap.add_argument("--data-port", help="radar DATA COM port (921600)")
    ap.add_argument("--cfg", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "configs",
        "iwr1843_live.cfg"), help="chirp config file to send on start")
    ap.add_argument("--no-send-cfg", action="store_true",
                    help="radar already configured/running — skip config")
    ap.add_argument("--yaw-offset", type=float, default=0.0,
                    help="radar-vs-camera yaw misalignment, degrees (+right)")
    ap.add_argument("--detect-every", type=int, default=2,
                    help="run YOLO every N camera frames")
    ap.add_argument("--no-radar", action="store_true",
                    help="camera-only mode (pinhole ranges, no serial)")
    ap.add_argument("--metal-db", type=float, default=METAL_DB,
                    help="peak score (dB above the range baseline) at/above "
                         "which -> metal")
    ap.add_argument("--fabric-db", type=float, default=FABRIC_DB,
                    help="peak score (dB above the range baseline) at/below "
                         "which -> fabric")
    ap.add_argument("--material-max-range", type=float,
                    default=MATERIAL_MAX_RANGE_M,
                    help="beyond this range (m) material is reported as "
                         "'unknown' — CFAR censoring leaves no material "
                         "signal at long range")
    ap.add_argument("--debug-refl", action="store_true",
                    help="print per-cluster reflectivity scores (for "
                         "calibrating --metal-db/--fabric-db against known "
                         "objects)")
    ap.add_argument("--no-log", action="store_true",
                    help="disable session logging (logs/session_*/)")
    ap.add_argument("--snap-every", type=float, default=5.0,
                    help="seconds between logged snapshots (0 = only on 's')")
    ap.add_argument("--label", default=None,
                    help="ground-truth tag for this whole session (e.g. "
                         "'metal_can', 'person_walking') — makes the recorded "
                         "track signatures usable as ML training data")
    ap.add_argument("--max-seconds", type=float, default=0,
                    help="auto-stop after this many seconds (0 = until 'q'); "
                         "handy for uniform labeled recordings")
    ap.add_argument("--radar-height", type=float, default=1.0,
                    help="radar mounting height above the ground in metres — "
                         "needed to recognize the GROUND plane")
    ap.add_argument("--dca", action="store_true",
                    help="capture raw ADC via the DCA1000EVM FPGA in parallel "
                         "with the UART point cloud (saves adc_raw.bin in the "
                         "session folder; auto-enables lvdsStreamCfg)")
    ap.add_argument("--dca-ip", default=None,
                    help="DCA1000 FPGA IP (default 192.168.33.180)")
    args = ap.parse_args(argv)

    if args.dca and args.no_radar:
        sys.exit("--dca needs the radar (remove --no-radar).")
    if args.dca and args.no_send_cfg:
        sys.exit("--dca must send the config so lvdsStreamCfg is enabled — "
                 "remove --no-send-cfg.")
    return args


def main():
    LiveApp(parse_args()).run()


if __name__ == "__main__":
    main()

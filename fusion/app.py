"""Main fusion loop + operating-mode selector.  Run: python -m fusion.app

Modes (the darkness policy, enforced here):
  rgb         lit scene            -> RGB detector + radar   (full detection)
  thermal     dark/smoke, Lepton   -> thermal detector + radar (full detection)
  radar-only  dark, no thermal yet -> radar clusters only: position + velocity
                                      + coarse class, no visual box

An RGB camera cannot see in complete darkness — RGB detections are NEVER
emitted in the dark, whatever the detector might hallucinate on a noisy
boosted frame.
"""
import argparse
import os
import sys
import time

import cv2

if hasattr(sys.stdout, "reconfigure"):     # Hebrew paths + cp1252 consoles
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from fusion.camera import make_camera
from fusion.detector import detect
from fusion.fuse import fuse, FusedObject
from fusion.radar import RadarSource
from radar_tracker import Tracker

WIN_TITLE = "VailSight fusion"

DARK_ENTER = 35.0    # mean luminance below this -> the scene is dark
DARK_EXIT = 50.0     # ...and back above this -> lit again (hysteresis keeps
                     # the mode from flapping at dusk)

MODE_COLORS = {"rgb": (60, 200, 60), "thermal": (0, 180, 255),
               "radar-only": (60, 60, 230)}
CLASS_COLORS = {"pedestrian": (60, 200, 60), "vehicle": (60, 120, 255),
                "static": (220, 220, 220)}


def select_mode(mean_luma, thermal_available, was_dark=False):
    """Pure mode decision (unit-tested): returns (mode, is_dark).

    RGB is allowed only when the scene is lit; in the dark the choice is
    thermal when available, else radar-only. Hysteresis via was_dark.
    """
    dark = mean_luma < (DARK_EXIT if was_dark else DARK_ENTER)
    if not dark:
        return "rgb", False
    return ("thermal" if thermal_available else "radar-only"), True


class ModeSelector:
    """Stateful wrapper: feed luminance, get the active mode + change log."""

    def __init__(self, thermal_available=False):
        self.thermal_available = thermal_available
        self.mode = None
        self._dark = False

    def update(self, mean_luma):
        mode, self._dark = select_mode(mean_luma, self.thermal_available,
                                       self._dark)
        if mode != self.mode:
            print("[mode] %s -> %s (luma %.1f)" % (self.mode, mode, mean_luma))
            self.mode = mode
        return mode


def radar_only_objects(clusters):
    """Radar clusters -> FusedObjects with no box (position/velocity/class)."""
    return [FusedObject(c["label"], (0, 0, 0, 0), range_m=c["range_m"],
                        doppler_mps=c["doppler_mps"], radar_class=c["label"],
                        n_radar_pts=c["n_points"], score=None)
            for c in clusters if c["n_points"] >= 3]


def draw_objects(img, objects, mode):
    for o in objects:
        x, y, w, h = o["box"]
        col = CLASS_COLORS.get(o["radar_class"], (255, 255, 255))
        if w > 0:
            cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
            txt = o["label"]
            if o["range_m"] is not None:
                txt += " %.1fm %+.1fm/s" % (o["range_m"], o["doppler_mps"])
                if o["radar_class"]:
                    txt += " (%s)" % o["radar_class"]
            else:
                txt += " (no radar return)"
            cv2.putText(img, txt, (max(x, 4), max(y - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    cv2.putText(img, "mode: %s" % mode, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, MODE_COLORS.get(mode, (255, 255, 255)), 2)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Radar+camera fusion loop")
    ap.add_argument("--cam", type=int, default=0, help="RGB camera index")
    ap.add_argument("--calib", default=None,
                    help="radar<->camera calib.json (default: calib_rgb.json "
                         "next to the package, if present)")
    ap.add_argument("--cfg-port", help="radar CONFIG COM port")
    ap.add_argument("--data-port", help="radar DATA COM port")
    ap.add_argument("--cfg", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "iwr1843_live.cfg"))
    ap.add_argument("--no-send-cfg", action="store_true")
    ap.add_argument("--no-radar", action="store_true",
                    help="camera-only development mode")
    ap.add_argument("--detect-every", type=int, default=2)
    ap.add_argument("--max-seconds", type=float, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    rgb = make_camera("rgb", index=args.cam)
    thermal = make_camera("thermal")
    calib = rgb.calib()
    if args.calib:
        from calibrate_radar_camera import load_calib
        calib = load_calib(args.calib)
    if calib is None:
        print("NOTE: no calib_rgb.json — fused range needs the geometric "
              "calibration (see calibrate_radar_camera.py); running "
              "detection + radar-only side by side.")

    radar = None
    if not args.no_radar:
        radar = RadarSource.open(args.cfg_port, args.data_port,
                                 cfg_path=args.cfg,
                                 send_cfg=not args.no_send_cfg)

    selector = ModeSelector(thermal_available=thermal.available())
    tracker = Tracker()          # persistent identity across frames/modes
    boxes = []
    frame_i, t0, last_mode = 0, time.time(), None
    print("Running. q = quit.")
    try:
        while True:
            if args.max_seconds and time.time() - t0 > args.max_seconds:
                break
            frame = rgb.get_frame()
            if frame is None:
                break
            frame_i += 1
            mode = selector.update(float(frame[::4, ::4].mean()))
            if mode != last_mode:
                # thermal boxes (warm/human/hot) must not survive into rgb
                # frames (and vice versa) — wrong labels AND wrong calibration
                boxes, last_mode = [], mode

            pts = radar.latest_points() if radar else []
            clusters = radar.clusters(include_points=True) if radar else []
            # tracks, not raw clusters, feed radar-only output: a person who
            # stops moving has ~0 Doppler and would re-classify as "static"
            # every frame — exactly the dark-scene failure the tracker's
            # person-stickiness exists to prevent
            tclusters = [tr.as_cluster()
                         for tr in tracker.update(clusters, time.time())]

            if mode == "rgb":
                de = max(args.detect_every, 1)   # N=1 -> 1 % N == 0 == i % 1
                if frame_i % de == 1 % de or not boxes:
                    boxes = detect(frame, "rgb")
                cam_calib, view = calib, frame
            elif mode == "thermal":
                tframe = thermal.get_frame()
                boxes = detect(tframe, "thermal")
                cam_calib, view = thermal.calib(), \
                    (tframe if tframe is not None else frame)
            else:                                   # radar-only
                boxes, cam_calib, view = [], None, frame

            if mode != "radar-only" and boxes:
                objects = fuse(boxes, pts, cam_calib, clusters, kind=mode)
            else:
                objects = radar_only_objects(tclusters)
                for o in objects:                   # no box -> emit as text
                    print("  radar: %-10s %5.1fm %+5.1fm/s (%d pts)"
                          % (o["radar_class"], o["range_m"],
                             o["doppler_mps"], o["n_radar_pts"]))

            draw_objects(view, objects, mode)
            cv2.imshow(WIN_TITLE, view)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q")):
                break
    finally:
        rgb.release()
        cv2.destroyAllWindows()
        if radar:
            radar.stop()


if __name__ == "__main__":
    main()

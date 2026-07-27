"""
Photo -> expected radar classification (IWR1843 pipeline preview).

Give it a photo of the room/scene and it predicts what the on-device radar
classifier (radar_classify_n6.py) is expected to output:

  photo -> YOLO object detection -> estimate 3D position per object
        -> synthesize the radar point cloud each object would return
        -> run the REAL classify_frame() from radar_classify_n6.py
        -> report expected labels + annotated image

Runs on PC (CPython + OpenCV), not on the N6 — it is a planning/preview tool:
"if I point the IWR1843 at this scene, what returns should I expect?"

Usage:
    python photo_to_radar.py room.jpg
    python photo_to_radar.py room.jpg --hfov 70        # camera horizontal FOV
    python photo_to_radar.py room.jpg --all-static     # nobody is moving
    python photo_to_radar.py room.jpg --json           # machine-readable output

Notes:
  - A still photo cannot show motion, so movable objects (people, vehicles)
    are simulated at their typical speed; --all-static simulates everyone
    standing still (then the radar sees them as 'static' clutter).
  - Range is estimated from bounding-box height via the pinhole model, so it
    is approximate (+-20%%) — good enough to predict the classifier's output.
"""
import argparse
import json
import math
import os
import random
import sys

import cv2
import numpy as np

from radar_classify_n6 import classify_frame, STATIC_V

# --- object detector ----------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
YOLO_CFG = os.path.join(MODEL_DIR, "yolov4-tiny.cfg")
YOLO_WEIGHTS = os.path.join(MODEL_DIR, "yolov4-tiny.weights")
YOLO_SIZE = 416
CONF_THR = 0.35
NMS_THR = 0.45

COCO = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
    "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# --- radar behaviour profile per detected class --------------------------------
# real_h  : typical real-world height (m) -> range from box height
# extent  : physical size (m) the radar cluster spreads over
# n_pts   : typical detected-points count at ~5 m (scaled by 1/range)
# speed   : typical radial speed if moving (m/s); 0 = always static
# vspread : micro-Doppler spread (m/s) — limbs/wheels
#
# SYNTHETIC-MODEL CAVEATS (this is a preview tool, not a measurement):
# 1. speeds above ~0.65 m/s exceed the default configs' unambiguous Doppler —
#    the REAL radar reports them aliased (folded modulo ~1.3 m/s), so e.g. a
#    6 m/s car will NOT show 6 m/s on hardware.
# 2. n ~ 1/range is a crude stand-in: real CFAR detection counts fall off a
#    cliff when SNR crosses the threshold, they do not decay linearly.
# All values here are invented plausible numbers, not calibrated.
PROFILES = {
    "person":     dict(real_h=1.70, extent=0.6, n_pts=8,  speed=1.2, vspread=0.9),
    "cat":        dict(real_h=0.30, extent=0.4, n_pts=3,  speed=0.8, vspread=0.4),
    "dog":        dict(real_h=0.55, extent=0.6, n_pts=4,  speed=1.0, vspread=0.5),
    "bicycle":    dict(real_h=1.05, extent=1.6, n_pts=6,  speed=3.0, vspread=0.6),
    "motorbike":  dict(real_h=1.10, extent=1.8, n_pts=8,  speed=6.0, vspread=0.5),
    "car":        dict(real_h=1.50, extent=3.5, n_pts=18, speed=6.0, vspread=0.3),
    "bus":        dict(real_h=3.00, extent=8.0, n_pts=30, speed=6.0, vspread=0.3),
    "truck":      dict(real_h=2.80, extent=6.0, n_pts=26, speed=6.0, vspread=0.3),
    # indoor static furniture / structure (speed=0 -> clutter)
    "chair":      dict(real_h=0.90, extent=0.6, n_pts=4,  speed=0.0, vspread=0.0),
    "sofa":       dict(real_h=0.85, extent=1.8, n_pts=8,  speed=0.0, vspread=0.0),
    "bed":        dict(real_h=0.60, extent=1.9, n_pts=8,  speed=0.0, vspread=0.0),
    "diningtable": dict(real_h=0.75, extent=1.4, n_pts=6, speed=0.0, vspread=0.0),
    "tvmonitor":  dict(real_h=0.60, extent=0.9, n_pts=4,  speed=0.0, vspread=0.0),
    "refrigerator": dict(real_h=1.70, extent=0.8, n_pts=6, speed=0.0, vspread=0.0),
    "bench":      dict(real_h=0.80, extent=1.5, n_pts=5,  speed=0.0, vspread=0.0),
    "pottedplant": dict(real_h=0.60, extent=0.4, n_pts=2, speed=0.0, vspread=0.0),
    # small indoor objects (weak/point returns)
    "bottle":     dict(real_h=0.25, extent=0.15, n_pts=2, speed=0.0, vspread=0.0),
    "cup":        dict(real_h=0.10, extent=0.10, n_pts=2, speed=0.0, vspread=0.0),
    "laptop":     dict(real_h=0.25, extent=0.35, n_pts=3, speed=0.0, vspread=0.0),
    "backpack":   dict(real_h=0.45, extent=0.40, n_pts=3, speed=0.0, vspread=0.0),
    "suitcase":   dict(real_h=0.60, extent=0.50, n_pts=4, speed=0.0, vspread=0.0),
    "microwave":  dict(real_h=0.30, extent=0.50, n_pts=3, speed=0.0, vspread=0.0),
    "oven":       dict(real_h=0.60, extent=0.60, n_pts=4, speed=0.0, vspread=0.0),
    "sink":       dict(real_h=0.20, extent=0.50, n_pts=3, speed=0.0, vspread=0.0),
    "toilet":     dict(real_h=0.40, extent=0.50, n_pts=3, speed=0.0, vspread=0.0),
}


def detect_objects(img):
    """Run YOLOv4-tiny; return [{label, conf, box:(x,y,w,h)}]."""
    if not (os.path.isfile(YOLO_CFG) and os.path.isfile(YOLO_WEIGHTS)):
        sys.exit("Model files missing in %s — see header of this script." % MODEL_DIR)
    net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (YOLO_SIZE, YOLO_SIZE),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    H, W = img.shape[:2]
    boxes, confs, ids = [], [], []
    for out in outs:
        for det in out:
            scores = det[5:]
            cid = int(np.argmax(scores))
            conf = float(scores[cid]) * float(det[4])   # class * objectness
            if conf < CONF_THR:
                continue
            cx, cy, bw, bh = det[0] * W, det[1] * H, det[2] * W, det[3] * H
            boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
            confs.append(conf)
            ids.append(cid)

    keep = cv2.dnn.NMSBoxes(boxes, confs, CONF_THR, NMS_THR)
    dets = []
    for i in np.array(keep).flatten():
        dets.append({"label": COCO[ids[i]], "conf": confs[i],
                     "box": tuple(boxes[i])})
    return dets


def estimate_position(box, profile, img_shape, hfov_deg):
    """Bounding box -> radar-frame position (x=forward, y=left, z=up), metres."""
    H, W = img_shape[:2]
    fx = (W / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = fx  # square pixels
    bx, by, bw, bh = box
    rng = profile["real_h"] * fy / max(bh, 1)          # pinhole: Z = h_real*f/h_px
    rng = min(max(rng, 0.3), 80.0)
    az = math.atan(((bx + bw / 2.0) - W / 2.0) / fx)    # + = right of centre
    el = math.atan((H / 2.0 - (by + bh / 2.0)) / fy)    # + = above centre
    x = rng * math.cos(az)                              # forward
    y = -rng * math.sin(az)                             # left (camera-right = -y)
    z = rng * math.sin(el)
    return x, y, z, rng, math.degrees(az)


def synthesize_points(det_pos, profile, moving, rng_m, seed):
    """Make the (x,y,z,doppler) points this object would return to the radar."""
    r = random.Random(seed)
    x0, y0, z0 = det_pos
    n = max(2, int(round(profile["n_pts"] * min(1.5, 5.0 / max(rng_m, 1.0)))))
    ext = profile["extent"]
    if moving and profile["speed"] > 0:
        v0 = profile["speed"] * r.choice((-1, 1))       # toward/away — unknowable from a photo
        vspread = profile["vspread"]
    else:
        v0, vspread = 0.0, 0.0
    pts = []
    for _ in range(n):
        pts.append((
            x0 + r.gauss(0, ext / 4.0),
            y0 + r.gauss(0, ext / 4.0),
            z0 + r.gauss(0, min(ext, profile["real_h"]) / 4.0),
            v0 + r.gauss(0, vspread / 2.0) + r.gauss(0, 0.05),  # + radar noise
        ))
    return pts


def annotate(img, results):
    colors = {"pedestrian": (60, 200, 60), "vehicle": (60, 120, 255),
              "static": (200, 200, 200), "unknown": (0, 220, 220),
              "no return": (80, 80, 80)}
    for res in results:
        bx, by, bw, bh = res["box"]
        c = colors.get(res["expected"], (255, 255, 255))
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), c, 2)
        txt = "%s->%s %.1fm" % (res["object"], res["expected"], res["range_m"])
        cv2.putText(img, txt, (bx, max(by - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
    return img


def imread_unicode(path):
    """cv2.imread fails on non-ASCII (Hebrew) Windows paths — decode manually."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".jpg", img)
    if ok:
        buf.tofile(path)
    return ok


def main():
    ap = argparse.ArgumentParser(description="Photo -> expected radar classes")
    ap.add_argument("image", help="photo of the room/scene")
    ap.add_argument("--hfov", type=float, default=60.0,
                    help="camera horizontal FOV in degrees (phone ~60-70, Lepton 57)")
    ap.add_argument("--all-static", action="store_true",
                    help="simulate everything motionless (nobody walking)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of table")
    args = ap.parse_args()

    img = imread_unicode(args.image)
    if img is None:
        sys.exit("Cannot read image: %s" % args.image)

    dets = [d for d in detect_objects(img) if d["label"] in PROFILES]
    if not dets:
        sys.exit("No radar-relevant objects detected in the photo.")

    # build the synthetic scene cloud + remember which points came from whom
    scene_pts, owners = [], []
    per_obj = []
    for i, d in enumerate(dets):
        prof = PROFILES[d["label"]]
        x, y, z, rng_m, az = estimate_position(d["box"], prof, img.shape, args.hfov)
        pts = synthesize_points((x, y, z), prof, not args.all_static, rng_m, seed=i)
        scene_pts += pts
        owners += [i] * len(pts)
        per_obj.append({"det": d, "pos": (x, y, z), "range_m": rng_m, "az_deg": az})

    # run the ACTUAL on-device classifier over the combined cloud
    clusters = classify_frame(scene_pts)

    # match each object to the nearest predicted cluster centroid
    results = []
    for o in per_obj:
        ox, oy, oz = o["pos"]
        best, bestd = None, 1e9
        for c in clusters:
            cx, cy, cz = c["centroid"]
            dd = (ox - cx) ** 2 + (oy - cy) ** 2 + (oz - cz) ** 2
            if dd < bestd:
                best, bestd = c, dd
        d = o["det"]
        prof = PROFILES[d["label"]]
        movable = prof["speed"] > 0
        results.append({
            "object": d["label"],
            "confidence": round(d["conf"], 2),
            "box": d["box"],
            "range_m": round(o["range_m"], 1),
            "azimuth_deg": round(o["az_deg"], 1),
            "expected": best["label"] if best else "no return",
            "expected_doppler_mps": best["doppler_mps"] if best else None,
            "cluster_points": best["n_points"] if best else 0,
            "if_stationary": "static" if movable and not args.all_static else None,
        })

    if args.json:
        print(json.dumps({"objects": results, "raw_clusters": clusters}, indent=2))
    else:
        print("\nExpected radar returns for this scene "
              "(%s):" % ("all motionless" if args.all_static else "movers at typical speed"))
        print("-" * 78)
        print("%-14s %-6s %-8s %-9s %-12s %s" %
              ("object", "conf", "range", "azimuth", "expected", "note"))
        print("-" * 78)
        for r in results:
            note = ""
            if r["if_stationary"]:
                note = "if standing still -> 'static' (|v|<%.2f m/s)" % STATIC_V
            print("%-14s %-6.2f %-8s %-9s %-12s %s" %
                  (r["object"], r["confidence"], "%.1fm" % r["range_m"],
                   "%+.0f deg" % r["azimuth_deg"], r["expected"], note))
        print("-" * 78)
        counts = {}
        for r in results:
            counts[r["expected"]] = counts.get(r["expected"], 0) + 1
        print("Summary:", ", ".join("%d x %s" % (v, k) for k, v in counts.items()))

    out_path = os.path.splitext(args.image)[0] + "_radar_expected.jpg"
    if imwrite_unicode(out_path, annotate(img.copy(), results)):
        print("Annotated image saved: %s" % out_path)


if __name__ == "__main__":
    main()

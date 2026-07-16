"""
Camera <-> radar fusion core — pure logic, no camera/serial/UI.

Everything here is testable offline (see tests/):

  CameraGeometry   pinhole camera model + radar-vs-camera yaw alignment;
                   the ONE place pixel<->azimuth conversion lives, so the
                   matcher and the overlay can never disagree about yaw
  ClusterMatcher   greedy 1:1 assignment of radar clusters to YOLO boxes
  RadarPipeline    points -> clusters -> material -> mixed-cluster split ->
                   persistent tracks -> scene semantics
  LowLightEnhancer gamma + CLAHE night vision for the detector

live_radar_camera.py wires these to the real camera, UART and screen.
"""
import math

import cv2
import numpy as np

from radar_classify_n6 import classify_frame
from radar_material import (annotate_clusters, split_mixed_clusters,
                            METAL_DB, FABRIC_DB, MATERIAL_MAX_RANGE_M)
from radar_tracker import Tracker
from scene_semantics import resolve_semantics
from photo_to_radar import PROFILES

MATCH_AZ_DEG = 10.0      # max azimuth gap between box centre and cluster
RANGE_WEIGHT = 0.15      # cost = az_err_deg + RANGE_WEIGHT * range_err_m
PERSON_METAL_COST = 5.0  # keeps a person box off the metal glint split from
                         # an object they are holding (same az + range)

DARK_MEAN = 50.0         # frame mean below this -> low-light pipeline
VERY_DARK_MEAN = 25.0    # below this also denoise + relax YOLO confidence
DARK_CONF_THR = 0.22     # YOLO confidence in the dark (recall > precision:
                         # a missed person costs more than a false box)


class CameraGeometry:
    """Pinhole camera model + radar-vs-camera yaw alignment.

    Convention: azimuth is degrees in the CAMERA frame, + = right of centre.
    A radar cluster's camera azimuth = its radar azimuth + yaw_offset_deg
    (yaw_offset = how far the radar boresight points right of the camera's).
    """

    def __init__(self, hfov_deg=60.0, yaw_offset_deg=0.0):
        self.hfov_deg = float(hfov_deg)
        self.yaw_offset_deg = float(yaw_offset_deg)

    def fx(self, width):
        """Focal length in pixels for a frame of this width."""
        return (width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    # -- camera side --------------------------------------------------------
    def box_azimuth_range(self, box, label, img_shape):
        """Camera azimuth (deg) and pinhole range estimate for a YOLO box."""
        H, W = img_shape[:2]
        fx = self.fx(W)
        bx, by, bw, bh = box
        az = math.degrees(math.atan(((bx + bw / 2.0) - W / 2.0) / fx))
        rng = PROFILES[label]["real_h"] * fx / max(bh, 1)
        return az, min(max(rng, 0.3), 80.0)

    def box_half_width_deg(self, box, img_shape):
        """Half the angular width of a box — used to widen the match gate."""
        W = img_shape[1]
        return math.degrees(math.atan((box[2] / 2.0) / self.fx(W)))

    # -- radar side ---------------------------------------------------------
    def cluster_azimuth_deg(self, centroid):
        """Radar cluster centroid -> azimuth in the CAMERA frame (deg)."""
        cx, cy, _ = centroid
        return (math.degrees(math.atan2(-cy, max(cx, 0.01)))
                + self.yaw_offset_deg)

    def project(self, centroid, img_shape):
        """Radar centroid -> (px, py) pixel position, or None when the point
        is behind the radar or outside the camera FOV (+5 deg slack)."""
        H, W = img_shape[:2]
        cx, cy, cz = centroid
        if cx <= 0.05:                        # behind / beside the radar
            return None
        az = math.radians(self.cluster_azimuth_deg(centroid))
        if abs(math.degrees(az)) > self.hfov_deg / 2.0 + 5:
            return None
        el = math.atan2(cz, math.sqrt(cx * cx + cy * cy))
        fx = self.fx(W)
        return (int(W / 2.0 + fx * math.tan(az)),
                int(H / 2.0 - fx * math.tan(el)))


class ClusterMatcher:
    """Greedy 1:1 assignment of radar clusters to camera detections.

    Cost = azimuth error (deg) + RANGE_WEIGHT * |range_radar - range_pinhole|,
    plus PERSON_METAL_COST when a 'person' box competes for a metal cluster.
    """

    def __init__(self, geometry, max_az_deg=MATCH_AZ_DEG,
                 range_weight=RANGE_WEIGHT,
                 person_metal_cost=PERSON_METAL_COST):
        self.geo = geometry
        self.max_az_deg = max_az_deg
        self.range_weight = range_weight
        self.person_metal_cost = person_metal_cost

    def match(self, dets, clusters, img_shape):
        """Returns a list parallel to dets: matched cluster dict or None."""
        cands = []
        for di, d in enumerate(dets):
            az_cam, rng_est = self.geo.box_azimuth_range(d["box"], d["label"],
                                                         img_shape)
            half_w = self.geo.box_half_width_deg(d["box"], img_shape)
            for ci, c in enumerate(clusters):
                az_err = abs(az_cam - self.geo.cluster_azimuth_deg(c["centroid"]))
                if az_err > self.max_az_deg + half_w:
                    continue
                cost = az_err + self.range_weight * abs(c["range_m"] - rng_est)
                # a person box should latch onto the body cluster, not the
                # metal glint split off an object they are holding
                if d["label"] == "person" and c.get("material") == "metal":
                    cost += self.person_metal_cost
                cands.append((cost, di, ci))
        cands.sort()
        assigned = [None] * len(dets)
        used_c = set()
        for cost, di, ci in cands:
            if assigned[di] is None and ci not in used_c:
                assigned[di] = clusters[ci]
                used_c.add(ci)
        return assigned


class RadarPipeline:
    """Radar points -> confirmed, characterized track-clusters.

    One call per frame: cluster -> reflectivity/material -> split mixed
    person+metal clusters -> persistent tracks -> semantics (tree/ground).
    Owns the Tracker, so camera identity votes go through here too.
    """

    def __init__(self, geometry, radar_height=1.0, metal_db=METAL_DB,
                 fabric_db=FABRIC_DB, max_range=MATERIAL_MAX_RANGE_M,
                 tracker=None):
        self.geo = geometry
        self.radar_height = radar_height
        self.metal_db = metal_db
        self.fabric_db = fabric_db
        self.max_range = max_range
        self.tracker = tracker or Tracker()

    def process(self, points, t, img=None):
        """points: [(x,y,z,v[,snr[,noise]]), ...] from the sliding window.
        img (optional) enables the color-based semantic checks; pass None in
        the dark — a boosted night frame is green sensor noise and every tall
        object would become a "tree".
        Returns the display-ready track-cluster dicts."""
        clusters = []
        if points:
            clusters = split_mixed_clusters(
                annotate_clusters(classify_frame(points, include_points=True),
                                  self.metal_db, self.fabric_db,
                                  max_range=self.max_range),
                self.metal_db, self.fabric_db, max_range=self.max_range)
        tracks = self.tracker.update(clusters, t)
        tclusters = [tr.as_cluster() for tr in tracks]
        resolve_semantics(tclusters, img, self.geo.hfov_deg,
                          self.geo.yaw_offset_deg, self.radar_height)
        return tclusters

    def note_camera_matches(self, dets, assigned):
        """Fuse camera identity into the tracks: a box matched now keeps the
        object named later, when the camera goes blind in the dark."""
        for d, cl in zip(dets, assigned):
            if cl is not None and "track_id" in cl:
                self.tracker.note_camera(cl["track_id"], d["label"])

    def signatures(self):
        return self.tracker.all_signatures()


class LowLightEnhancer:
    """Gamma + CLAHE low-light enhancement for the detector.

    Linear gain (convertScaleAbs) clips highlights and amplifies sensor noise
    1:1 — YOLO recall in the dark stayed near zero.  Gamma lifts shadows
    without clipping, CLAHE restores the local contrast the detector needs,
    and a median filter kills the salt noise when the scene is nearly black.
    """

    def __init__(self, dark_mean=DARK_MEAN, very_dark_mean=VERY_DARK_MEAN,
                 target=110.0, min_gamma=0.35):
        self.dark_mean = dark_mean
        self.very_dark_mean = very_dark_mean
        self.target = target
        self.min_gamma = min_gamma          # cap the lift; below it noise wins
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    @staticmethod
    def frame_mean(img):
        return float(img[::4, ::4].mean())

    def enhance(self, img):
        """Returns (image, gamma_used, frame_mean); gamma 1.0 = untouched."""
        mean_b = self.frame_mean(img)
        if mean_b >= self.dark_mean:
            return img, 1.0, mean_b
        g = (math.log(self.target / 255.0)
             / math.log(max(mean_b, 3.0) / 255.0))
        g = max(g, self.min_gamma)
        lut = np.array([(i / 255.0) ** g * 255.0 for i in range(256)],
                       dtype=np.uint8)
        out = cv2.LUT(img, lut)
        if mean_b < self.very_dark_mean:
            out = cv2.medianBlur(out, 3)
        yuv = cv2.cvtColor(out, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = self._clahe.apply(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR), g, mean_b

    def conf_threshold(self, mean_b, normal_thr, dark_thr=DARK_CONF_THR):
        """In the dark trade precision for recall — missing the person
        entirely is the worse failure."""
        return dark_thr if mean_b < self.very_dark_mean else normal_thr

    def color_checks_usable(self, mean_b):
        """False when the frame is too dark for color-based semantics."""
        return mean_b >= self.very_dark_mean

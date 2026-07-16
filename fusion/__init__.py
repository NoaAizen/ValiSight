"""Camera-agnostic radar+camera fusion package.

Modules:
  camera    Camera interface: RGBCamera (now) / ThermalCamera (Lepton, later)
  detector  detect(frame, kind): YOLO for RGB, temperature bands for thermal
  radar     IWR1843 DATA-UART ingest -> latest_points() + classified clusters
  fuse      calib.json projection + box<->point association -> FusedObject
  app       main loop + light/dark mode selector (python -m fusion.app)

Darkness policy (HARD): RGB detections are never emitted in the dark — the
mode selector routes to thermal when available, else radar-only.
"""
from fusion.camera import Camera, RGBCamera, ThermalCamera, make_camera
from fusion.detector import Box, detect
from fusion.fuse import fuse, FusedObject
# fusion.app (select_mode, ModeSelector, main) is imported directly so that
# `python -m fusion.app` does not double-execute the module

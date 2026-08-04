"""Interactive NUC/FPN calibration for the Lepton — run this YOURSELF at the rig.

    python3 tools/calibrate_nuc.py

Pan the camera while it runs. The live meter tells you, every 2 seconds,
whether you are moving enough — no guessing, no chat round-trips:

    NOT MOVING — pan the camera!
    moving — go WIDER  (0.06 / 0.12)
    GOOD (0.19) — hold this kind of sweep...

It stops by itself once the capture will clear the scene-change gate with
margin, runs lepton_fix.measure_pixel_offsets (the real gate), and writes the
candidate map + a PNG rendering to src/cfg/. The PNG then gets a HUMAN look
before acceptance — the gate cannot see a person burned into the map
(CLAUDE.md trap #14); eyes can.

Procedure (trap #14 is the law here): stand BEHIND the camera, NOBODY in
frame, pan slowly across a thermally varied scene (door, AC, electronics —
not one blank wall), and end the sweep facing a different direction than you
started.
"""
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import view_thermal_rgb as vtr          # N6Stream + protocol decode
import lepton_fix

DEVICE_SCRIPT = os.path.join(_ROOT, "n6_thermal_stream.py")
BLOCK = 43                              # ~5 s at 8.7 Hz
GATE = lepton_fix.FPN_MIN_SCENE_CHANGE  # 0.12
STOP_AT = 0.18                          # stop with margin over the gate
MIN_MOVING_S = 45.0                     # moving footage needed before stop
TIMEOUT_S = 240.0


def _lp(a):
    return lepton_fix._lowpass(a)


def _change(ref_lp, block_frames):
    b = _lp(np.stack(block_frames).astype(np.float32).mean(axis=0))
    x = ref_lp.ravel() - ref_lp.mean()
    y = b.ravel() - b.mean()
    d = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return 1.0 - (float((x * y).sum() / d) if d > 1e-9 else 1.0)


def capture():
    port = vtr.find_openmv_port()
    if not port:
        sys.exit("no OpenMV board found — is the N6 plugged in?")
    print("connecting (Lepton settle ~5s)... start panning as soon as the "
          "meter appears.")
    stream = vtr.N6Stream(port, script=DEVICE_SCRIPT)
    frames, ref = [], None
    motion_at = None
    last_meter = last_frame = t0 = time.time()
    try:
        while True:
            now = time.time()
            if now - t0 > TIMEOUT_S:
                print("\ntimeout — analysing what we have")
                break
            if now - last_frame > 8.0:
                print("  stream stalled — restarting it, KEEP PANNING")
                stream.terminate()
                time.sleep(1.0)
                stream = vtr.N6Stream(vtr.find_openmv_port() or port,
                                      script=DEVICE_SCRIPT)
                last_frame = time.time()
            line = next(stream.lines())
            if line[:2] != b"T:":
                continue
            _, payload = vtr.split_stamped(line[2:])
            _, raw = vtr.decode_thermal(payload)
            if raw is None:
                continue
            last_frame = time.time()
            frames.append(np.frombuffer(raw, np.uint8).reshape(120, 160).copy())

            if ref is None and len(frames) >= BLOCK:
                ref = _lp(np.stack(frames[:BLOCK]).astype(np.float32).mean(axis=0))
            if ref is None or len(frames) < 2 * BLOCK:
                continue
            if now - last_meter < 2.0:
                continue
            last_meter = now
            ch = _change(ref, frames[-BLOCK:])
            if ch < 0.02:
                msg = "NOT MOVING — pan the camera!"
            elif ch < GATE:
                msg = "moving — go WIDER  (%.2f / %.2f)" % (ch, GATE)
            else:
                msg = "GOOD (%.2f) — hold this kind of sweep..." % ch
            if motion_at is None and ch > 0.02:
                motion_at = len(frames)
            moving_s = 0.0 if motion_at is None else (len(frames) - motion_at) / 8.7
            print("  [%3ds] %-46s %s" % (now - t0, msg,
                                         "#" * int(min(ch, 0.30) * 80)),
                  flush=True)
            if ch >= STOP_AT and moving_s >= MIN_MOVING_S:
                print("\nenough motion collected — analysing")
                break
    finally:
        stream.terminate()
    return frames, motion_at


def main():
    frames, motion_at = capture()
    print("captured %d frames" % len(frames))
    if motion_at and motion_at > BLOCK:
        cut = motion_at - BLOCK          # keep one block before motion began
        frames = frames[cut:]
        print("trimmed %d static head frames" % cut)

    off, note = lepton_fix.measure_pixel_offsets(frames, lepton_fix.DEAD_ROWS)
    print("verdict:", note)
    if off is None:
        sys.exit("REJECTED — run again and follow the meter")

    sd = float(off.std())
    stamp = time.strftime("%Y%m%d_%H%M")
    base = os.path.join(_ROOT, "src", "cfg", "lepton_fpn_candidate_" + stamp)
    np.save(base + ".npy", off)
    import cv2
    vis = np.clip((off - off.min()) / max(float(off.ptp()), 1e-6) * 255,
                  0, 255).astype(np.uint8)
    cv2.imwrite(base + ".png", cv2.resize(vis, (640, 480),
                                          interpolation=cv2.INTER_NEAREST))
    print("candidate saved: %s.npy (+.png)" % base)
    print("map sd = %.2f DN  (real FPN ~2; the person-burn-in specimen was 12.4)" % sd)
    if sd > 4.0:
        print("WARNING: sd is suspiciously high — likely scene or person "
              "burn-in. Look at the PNG before trusting this map.")
    else:
        print("looks plausible — have the PNG visually checked for any "
              "silhouette before renaming it to accepted.")


if __name__ == "__main__":
    main()

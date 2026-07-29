"""Live view of the ValiSight sensors, served over HTTP from the Jetson.

    python3 live_server.py [--port 8080] [--camera rgb|thermal]

Open http://<jetson>:8080 from any machine on the network.

ONE CAMERA AT A TIME. The PAG7936 and the Lepton share the N6's CSI interface,
and once one has been initialised the other's captures fail ("Frame capture has
failed"). Each works perfectly on its own, so switching cameras means a
MicroPython soft reset (Ctrl-D) to clear the CSI state and re-init the other --
about five seconds, which is why it is a deliberate toggle rather than
per-frame alternation.

The radar is a separate serial port and streams continuously regardless.
Camera frames use ONE REPL round trip each (capture, JPEG and base64 in a single
statement); the chunked 512-byte transfer used for file pulls costs a round trip
per chunk and is far too slow to watch.

The two cameras take different paths on purpose. RGB is encoded to JPEG on the
board -- 640x400 colour is far too much to ship raw. Thermal is shipped RAW
(19200 bytes, ~25 KB base64) and encoded on the Jetson, because the frame needs
host-side repair before anything else happens to it: this module's Lepton has 14
dead rows, and interpolating them has to come before JPEG, not after.

The thermal flicker was the sensor's AGC shipping with AGC_HEQ_DAMPENING_FACTOR
at 0, so it rebuilt its histogram mapping from scratch every frame. init_camera
writes that register; the AGC itself stays on, because it is what gives the
image its contrast. See lepton_fix.py for both, and for what was tried first
and did not work.
"""
import argparse, base64, glob, hashlib, io, json, math, os, struct, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import serial
from PIL import Image
import chirp_geometry
import iwr1843_uart
from iwr1843_uart import RadarReader
import frame_clock
import attitude as attitude_mod
import lepton_fix
import recorder as recorder_mod
import radar_gate
import radar_classify_n6

# The pure fusion layer lives at the repo root (core/), one level up.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from core.detect import thermal as thermal_detect            # noqa: E402
from core.detect import merge as detect_merge                # noqa: E402
from core.fusion import azimuth as az_fuse                   # noqa: E402
from PIL import ImageDraw                                    # noqa: E402

# Learned person detector (adapter, heavy): runs in its OWN thread at its own
# pace (~105 ms/frame on CPU) so the 8.8 fps capture loop never blocks on it.
# The worker always consumes the LATEST frame and drops the rest; the fusion
# step merges whatever AI boxes are freshest. Model missing -> silently off.
AI = {"frame": None, "boxes": [], "on": False}
ai_cv = threading.Condition()


def ai_worker():
    import thermal_ai
    try:
        thermal_ai._load()
    except Exception as e:                          # noqa: BLE001
        print("thermal AI : disabled (%s)" % e)
        with lock:
            state["ai"] = "disabled: %s" % e
        return
    with ai_cv:
        AI["on"] = True
    with lock:
        state["ai"] = "on"
    print("thermal AI : YOLOv4-tiny person detector on")
    while True:
        with ai_cv:
            while AI["frame"] is None:
                ai_cv.wait()
            frame, AI["frame"] = AI["frame"], None
        try:
            boxes = thermal_ai.detect_person(frame)
        except Exception:                           # noqa: BLE001
            boxes = []
        with ai_cv:
            AI["boxes"] = boxes

BY_ID = "/dev/serial/by-id"
PROMPT = b">>> "
CAMERAS = ("rgb", "thermal")

# Frames go out as an MJPEG stream rather than inside state.json: polling for a
# base64 image caps the display at the poll rate and re-decodes a data: URI every
# time. The condition lets the stream handler block until a frame actually
# arrives instead of spinning.
frames = {"jpeg": None, "seq": 0, "mono_us": None, "host_wall": None}
frame_cv = threading.Condition()

state = {
    "camera": "rgb",
    "palette": "blackhot",
    "switching": False,
    "ffc_running": False,
    "calibrating": None,      # frames captured so far, None when idle
    "fpn": "not calibrated",
    "cam_error": None,
    "cam_fps": 0.0,
    # One rate per sensor rather than one "camera" rate. The cameras cannot run
    # at once, so the idle one is pinned to 0 on every switch instead of being
    # left showing the rate it had before it was stopped.
    "fps": {"rgb": 0.0, "thermal": 0.0, "radar": 0.0},
    "dead_rows": [],
    "offset_rows": [],
    # Sustained bytes/s actually crossing each link. This is the number to
    # judge a clock setting by: a nominal baud rate says nothing once REPL
    # turnaround and base64's 4-bytes-per-3 expansion are in the path.
    "link": {"openmv_bps": 0, "openmv_peak_bps": 0,
             "radar_bps": 0, "radar_peak_bps": 0},
    "frame": {"seq": 0, "mono_us": None, "host_wall": None},
    "imu": None,
    "attitude": None,
    "radar": {"frames": 0, "raw": 0, "kept": 0, "clusters": [], "fps": 0.0},
    "fusion": [],             # thermal x radar fused objects (decision path)
    "ai": "off",              # learned person detector: off / on / disabled
    "radar_error": None,
}
desired = {"camera": "rgb", "palette": "blackhot"}
lock = threading.Lock()
REC = None                      # recorder.Recorder once --record is given
ATT = {"level": False, "reset_heading": False, "ffc": False,
       "calibrate": False, "clear_fpn": False}   # one-shot UI requests

# Per-pixel fixed-pattern map, measured on this module and reused across runs.
# Kept beside the chirp configs rather than under data/: it is a fact about the
# rig's hardware, not a capture. See lepton_fix.measure_pixel_offsets.
FPN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "cfg", "lepton_fpn.npy")
FPN = {"map": None, "note": "not calibrated"}


def by_id(pattern):
    for p in sorted(glob.glob(os.path.join(BY_ID, pattern))):
        return os.path.realpath(p)
    return None


def read_to_prompt(ser, timeout=20):
    buf, t0 = b"", time.time()
    while time.time() - t0 < timeout:
        n = ser.in_waiting
        if n:
            buf += ser.read(n)
            if buf.endswith(PROMPT):
                break
        else:
            time.sleep(0.005)
    return buf


def repl(ser, line, timeout=20):
    ser.write(b"\x03")
    time.sleep(0.05)
    ser.reset_input_buffer()
    return _exchange(ser, line, timeout)


def repl_fast(ser, line, timeout=20):
    """Hot-path variant: no Ctrl-C and no 50 ms settle.

    The grab statement always returns to the prompt on its own, so interrupting
    first only costs latency on every single frame.
    """
    return _exchange(ser, line, timeout)


def _exchange(ser, line, timeout):
    ser.write(line.encode() + b"\r\n")
    raw = read_to_prompt(ser, timeout).decode(errors="replace")
    out = raw.split("\r\n", 1)[-1]
    for tail in (">>> ", "... "):
        while out.endswith(tail):
            out = out[: -len(tail)]
    return out.strip()


def last_line(text):
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def soft_reset(ser):
    """Ctrl-D at the friendly prompt: the only way to clear a CSI init."""
    ser.write(b"\x03")
    time.sleep(0.2)
    ser.write(b"\x04")
    time.sleep(4.0)
    ser.reset_input_buffer()
    ser.write(b"\r\n")
    read_to_prompt(ser, 6)


# Board-side IMU ring buffer. Sampling the IMU once per camera grab gave 8.77 Hz
# with 61 ms of dt jitter and worst-case 1.8 s gaps -- useless for tracking
# attitude through a walk, and attitude error is what turns into position error.
# A timer callback fills a ring instead, drained whole with each frame.
#
# Measured on the board: 200.0 Hz, dt median 5000 us, sd 4 us. The callback runs
# as a soft IRQ (it allocates a tuple without complaint), so a `busy` flag rather
# than disable_irq() is what keeps the drain from racing it.
#
# The register write raises ODR from 52 Hz to 208 Hz and DELIBERATELY leaves
# full-scale alone. OpenMV's imu module hard-codes the +/-8g and +/-2000 dps
# scale factors: changing FS behind it made |a| report 1998.9 mg instead of
# 1004.8, and since a camera switch soft-resets the driver back to defaults, any
# host-side correction factor would go stale invisibly. ODR is the change that
# actually matters and it carries no such hazard.
IMU_RING = 96
IMU_HZ = 200
_IMU_SETUP = [
    "import imu, machine, array, struct",
    "_wr = getattr(imu, '__write_reg')",
    "_wr(0x10, 0x5C); _wr(0x11, 0x5C)",          # CTRL1_XL / CTRL2_G: ODR 208 Hz
    "_N = %d" % IMU_RING,
    "_ts = array.array('i', [0] * _N)",
    "_ia = array.array('f', [0.0] * (_N * 6))",
    "_ix = [0]; _ov = [0]; _busy = [0]",
    ("exec(\"def _tick(t):\\n"
     " if _busy[0]:\\n  return\\n"
     " i = _ix[0]\\n"
     " if i >= _N:\\n  _ov[0] += 1\\n  return\\n"
     " a = imu.acceleration_mg(); g = imu.angular_rate_mdps()\\n"
     " _ts[i] = time.ticks_us()\\n"
     " j = 6 * i\\n"
     " _ia[j] = a[0]; _ia[j+1] = a[1]; _ia[j+2] = a[2]\\n"
     " _ia[j+3] = g[0]; _ia[j+4] = g[1]; _ia[j+5] = g[2]\\n"
     " _ix[0] = i + 1\")"),
    "_tim = machine.Timer(-1); _tim.init(freq=%d, callback=_tick)" % IMU_HZ,
]

# One 28-byte record per sample: ticks_us then 6 floats.
IMU_REC = "<i6f"
IMU_REC_LEN = 28


def _grab_stmt(encode, with_imu):
    """One REPL round trip: capture, stamp, drain the IMU ring, encode, emit.

    The stamp is taken immediately after snapshot() returns and before any
    encoding, so it dates the capture rather than the JPEG. The IMU ring is
    drained in the same round trip on purpose: fetching it separately would put
    an unknown REPL turnaround between the frame and the attitude that is
    supposed to describe it.

    Wire format, one line: mono_us n_imu n_overflow <imu_base64> <frame_base64>
    Packed rather than printed as text -- 23 samples is 644 bytes packed against
    roughly 1.5 KB as decimal, and it rides alongside a 25 KB frame either way.
    """
    if not with_imu:
        return ("_f = c.snapshot(); _t = time.ticks_us(); " + encode +
                "print(_t, 0, 0, '-', "
                "binascii.b2a_base64(_p).decode().strip())")
    return (
        "_f = c.snapshot(); _t = time.ticks_us(); "
        "_busy[0] = 1; _n = _ix[0]; "
        "_blob = b''.join(struct.pack('" + IMU_REC + "', _ts[k], _ia[6*k], "
        "_ia[6*k+1], _ia[6*k+2], _ia[6*k+3], _ia[6*k+4], _ia[6*k+5]) "
        "for k in range(_n)); "
        "_ix[0] = 0; _o = _ov[0]; _ov[0] = 0; _busy[0] = 0; " + encode +
        "print(_t, _n, _o, binascii.b2a_base64(_blob).decode().strip(), "
        "binascii.b2a_base64(_p).decode().strip())")


def init_camera(ser, cam, quality, damping=lepton_fix.AGC_DAMPING, with_imu=True):
    """Bring one camera up. Returns (grab statement, payload kind)."""
    if cam == "rgb":
        setup = ["import csi, binascii, time",
                 "import imu" if with_imu else "pass",
                 "c = csi.CSI(cid=csi.PAG7936); c.reset()",
                 "c.pixformat(csi.RGB565); c.framesize(csi.QVGA)"]
        grab = _grab_stmt("_p = _f.to_jpeg(quality=%d).bytearray(); " % quality,
                          with_imu)
        kind = "jpeg"
    else:
        # AGC stays ON -- it is what gives the image its contrast. The flicker
        # came from it shipping with zero temporal damping, so the histogram
        # mapping was rebuilt from scratch every frame. Damping only limits how
        # fast that mapping may move; measured max |dmean| 12.19 -> 1.70 DN
        # with scene contrast unchanged. ROI is written back to the full frame
        # because these attributes survive a soft reset of the MCU.
        setup = ["import csi, binascii, time",
                 "import imu" if with_imu else "pass",
                 "c = csi.CSI(cid=csi.LEPTON); c.reset()",
                 "c.pixformat(csi.GRAYSCALE); c.framesize(csi.QQVGA)",
                 "[c.snapshot() for _ in range(20)]",
                 "c.ioctl(csi.IOCTL_LEPTON_RUN_COMMAND, 0x0242)",
                 "c.ioctl(csi.IOCTL_LEPTON_SET_ATTRIBUTE, 0x%04X, bytes([0,0,0,0,159,0,119,0]))"
                 % lepton_fix.CID_AGC_ROI,
                 "c.ioctl(csi.IOCTL_LEPTON_SET_ATTRIBUTE, 0x%04X, bytes([%d,%d]))"
                 % (lepton_fix.CID_AGC_HEQ_DAMPENING_FACTOR, damping & 0xFF, damping >> 8)]
        # Raw pixels, not JPEG: the dead rows have to be interpolated before
        # the frame is compressed, and to_ironbow() on the board never had the
        # second frame buffer it needs anyway.
        grab = _grab_stmt("_p = _f.bytearray(); ", with_imu)
        kind = "raw"
    # The IMU ring has to exist on the board BEFORE the first grab: _grab_stmt
    # drains it inline and references _busy/_ix/_ts/_ia/_ov by name. Sending
    # _IMU_SETUP was lost at some point while the constant stayed behind, so
    # every grab raised `NameError: name '_busy' isn't defined` -- and because
    # the responses below were discarded, and grab_payload threw away anything
    # under 128 characters, the symptom on screen was a camera "running" at
    # 64 fps with zero frames and no error at all.
    if with_imu:
        setup = setup + _IMU_SETUP

    failures = []
    for line in setup:
        resp = repl(ser, line, timeout=40)
        if resp and ("Traceback" in resp or "Error" in resp):
            failures.append((line, last_line(resp)))
    if failures:
        # Report, do not raise: a board that half-initialises still shows a
        # picture, and a live view that refuses to start tells you less than one
        # that starts and says what is broken. But it must SAY it.
        with lock:
            state["cam_error"] = "init failed: %s -> %s" % (
                failures[0][0][:60], failures[0][1][:90])
        for line, err in failures:
            print("cam init   : %-52s %s" % (line[:52], err[:80]))
    return grab, kind


class Grab:
    """What one round trip to the board produced."""

    __slots__ = ("payload", "mono_us", "imu", "wire_len", "error",
                 "imu_samples", "imu_overflow")

    def __init__(self, payload=None, mono_us=None, imu=None, wire_len=0,
                 error=None, imu_samples=(), imu_overflow=0):
        self.payload, self.mono_us, self.imu = payload, mono_us, imu
        self.wire_len, self.error = wire_len, error
        # The whole 200 Hz ring drained with this frame, oldest first, each
        # (ticks_us, ax, ay, az, gx, gy, gz). `imu` is the newest one, kept
        # because the UI and the state dict only ever show a single reading.
        self.imu_samples = list(imu_samples)
        self.imu_overflow = imu_overflow


def grab_payload(ser, grab, with_imu=True):
    """One frame off the board: mono_us [ax ay az gx gy gz] <base64>.

    wire_len counts the characters the board actually put on the wire, not the
    decoded frame size -- base64 costs 4 bytes per 3, and the link rate that
    matters for clock configuration is the one the USB endpoint really carries.
    """
    raw = repl_fast(ser, grab, timeout=20)
    out = last_line(raw)
    # Look for the failure in the WHOLE response, not in its last line. This
    # used to test `out.startswith("Traceback")` after last_line() had already
    # thrown everything but the final line away -- and "Traceback (most recent
    # call last):" is the FIRST line, so that branch could never fire. What
    # survived was the exception line itself, typically ~30 characters, which
    # then failed the `len(out) <= 128` test below and was discarded as
    # "nothing useful". The board was reporting a real fault on every single
    # frame and the host was deleting the message: measured 6868 consecutive
    # empty grabs with cam_error still None.
    if raw and ("Traceback" in raw or "Error:" in raw or "error:" in raw):
        return Grab(error=(out or raw.strip().replace("\r\n", " | "))[:200],
                    wire_len=len(raw))
    if not out:
        return Grab()                           # genuinely nothing on the wire
    if len(out) <= 128:
        # Too short to be a frame, but the board said SOMETHING. That is a
        # diagnosis, not noise -- report it rather than dropping it.
        return Grab(error="board replied %d chars, not a frame: %r"
                          % (len(out), out[:120]), wire_len=len(out))
    # Wire format, from _grab_stmt: mono_us n_imu n_overflow <imu_b64> <frame_b64>
    # -- five fields, with or without the IMU (the no-IMU statement sends 0 0 '-').
    # This used to split for seven, i.e. the pre-ring format where each grab
    # carried ONE reading as six decimal floats. The board-side ring landed and
    # its consumer here did not, so every frame failed to parse.
    parts = out.split(None, 4)
    if len(parts) != 5:
        return Grab(error="short header: %r" % out[:80], wire_len=len(out))
    try:
        mono_us = int(parts[0])
        n_imu, n_ov = int(parts[1]), int(parts[2])
        samples = []
        if with_imu and n_imu and parts[3] != "-":
            blob = base64.b64decode(parts[3])
            want = n_imu * IMU_REC_LEN
            if len(blob) < want:
                return Grab(error="imu blob short: %d of %d bytes"
                                  % (len(blob), want), wire_len=len(out))
            for k in range(n_imu):
                samples.append(struct.unpack_from(IMU_REC, blob,
                                                  k * IMU_REC_LEN))
        imu = tuple(samples[-1][1:]) if samples else None
        return Grab(base64.b64decode(parts[4]), mono_us, imu, len(out),
                    imu_samples=samples, imu_overflow=n_ov)
    except Exception as e:
        return Grab(error="bad frame header (%s)" % e, wire_len=len(out))


def repair_thermal(buf, bad, opts):
    """Raw Lepton bytes -> the DECISION-PATH frame (repaired, pre-regain).

    Order is not arbitrary. Level-correct first, because several offset rows sit
    directly beside a dead one and would otherwise poison its interpolation.
    Fill the dead rows next. Regain is display-only and deliberately NOT here:
    detection runs on this frame, and a per-frame display stretch has no place
    in the decision path.
    """
    dead, offset = bad
    frame = lepton_fix.decode(buf)
    if frame is None or opts.no_repair:
        return frame
    frame = lepton_fix.destripe(frame, dead, offset)
    frame = lepton_fix.repair(frame, dead)
    # Per-pixel correction goes AFTER the row work and BEFORE regain. After,
    # because dead rows are invented by repair() and have no sensor pattern
    # to correct. Before, because regain multiplies by ~4.4 and correcting a
    # stretched frame would need a stretched map -- one that goes stale the
    # moment the scene's span changes.
    return lepton_fix.apply_pixel_offsets(frame, FPN["map"])


OVERLAY_SCALE = 3           # 160x120 -> 480x360 so overlay text is readable
SENSOR_COLORS = {"thermal": (255, 170, 40), "radar": (60, 220, 230)}


def draw_fusion_overlay(rgb_img, objects, clusters):
    """Draw fused objects + radar azimuth ticks on the (already colourised)
    display image. Display only — nothing here feeds back into detection."""
    img = rgb_img.resize((rgb_img.width * OVERLAY_SCALE,
                          rgb_img.height * OVERLAY_SCALE), Image.NEAREST)
    d = ImageDraw.Draw(img)
    s = OVERLAY_SCALE
    for o in objects or []:
        x, y, w, h = [v * s for v in o["box"]]
        color = SENSOR_COLORS.get(o["contributing_sensor"], (200, 200, 200))
        d.rectangle([x, y, x + w, y + h], outline=color, width=2)
        line1 = "%sconf %.2f  s=%.2f  [%s]" % (
            ("%s  " % o["label"]) if o.get("label") else "",
            o["confidence"], o["uncertainty"], o["contributing_sensor"])
        line2 = ("%.1fm  %+.1fm/s  %s" % (o["range_m"], o["doppler_mps"],
                                          o["radar_class"])
                 if o["range_m"] is not None else "camera only")
        tx = min(x + 2, img.width - 150)      # keep labels inside the frame
        d.text((tx, max(0, y - 22)), line1, fill=color)
        d.text((tx, max(10, y - 11)), line2, fill=color)
    # radar presence ticks along the bottom edge, matched or not: the dark /
    # thermal-blind beats must still show where the radar sees something
    for c in clusters or []:
        az = az_fuse.cluster_azimuth_deg(c["centroid"])
        fx = az_fuse.focal_px(rgb_img.width)
        u = rgb_img.width / 2.0 + fx * math.tan(math.radians(az))
        if 0 <= u < rgb_img.width:
            u *= s
            d.line([u, img.height - 14, u, img.height], fill=(60, 220, 230),
                   width=2)
            d.text((min(u + 3, img.width - 60), img.height - 13),
                   "%.1fm %s" % (c["range_m"], c["label"][:4]),
                   fill=(60, 220, 230))
    return img


def thermal_to_jpeg(buf, bad, regain, opts, dt, frame=None, objects=None,
                    clusters=None):
    """Displayable JPEG: repaired frame -> regain -> palette -> overlay.

    ``frame``: the repair_thermal() result when the caller already computed
    it (the fusion path does); decoded from ``buf`` otherwise.
    """
    dead, _ = bad
    if frame is None:
        frame = repair_thermal(buf, bad, opts)
    if frame is None:
        return None
    if not opts.no_repair and not opts.no_regain:
        frame = regain.apply(frame, dead, dt)
    # Read the palette per frame rather than binding it once at startup. It is
    # a host-side lookup table applied after every correction step, so unlike a
    # camera switch it costs nothing and needs no soft reset -- there is no
    # reason to make someone restart the server to change it.
    with lock:
        pal = desired["palette"]
    img = Image.fromarray(lepton_fix.colourise(frame, pal))
    if objects is not None or clusters:
        img = draw_fusion_overlay(img, objects, clusters)
    out = io.BytesIO()
    img.save(out, "JPEG", quality=opts.quality)
    return out.getvalue()


def learn_bad_rows(ser, grab, opts):
    """Measure this module's defect map instead of trusting a constant.

    A replacement Lepton will have a different one, and a silently stale list
    would correct healthy rows while leaving broken ones on screen -- the
    failure that looks like it is working.
    """
    fallback = (lepton_fix.DEAD_ROWS, lepton_fix.OFFSET_ROWS)
    if not opts.detect_dead:
        return fallback, "hard-coded"
    probe = []
    for _ in range(8):
        g = grab_payload(ser, grab, opts.imu)
        f = lepton_fix.decode(g.payload)
        if f is not None:
            probe.append(f)
    found = lepton_fix.detect_bad_rows(probe)
    if found is None:
        return fallback, "detection inconclusive, using hard-coded"

    # Detection can fail QUIETLY as well as loudly, and the quiet way is the one
    # that reaches the screen. Measured on this rig: it returned a single dead
    # row against the 14 this module is known to have, so repair() filled one and
    # left thirteen raw -- and regain() then stretched those to saturation, which
    # is the bright horizontal banding it is supposed to remove.
    #
    # The two errors are not symmetric. Flagging a healthy row costs one
    # interpolated line nobody can see; missing a dead one puts a white bar
    # across the image. So a detection that finds far FEWER defects than the map
    # measured on this module is treated as a failed measurement, not as a
    # healed sensor -- a Lepton does not grow its rows back. A genuinely
    # different module (a replacement) will still be believed, because the test
    # is against a large shortfall rather than against any disagreement.
    n_found, n_known = len(found[0]), len(fallback[0])
    if n_known and n_found < 0.6 * n_known:
        note = ("detection found %d dead rows against %d known on this module "
                "-- distrusted, using the measured map. Re-measure with "
                "lepton_diag if the sensor was replaced."
                % (n_found, n_known))
        with lock:
            state["cam_error"] = note
        return fallback, note
    return found, "detected from %d frames" % len(probe)


def camera_thread(port, opts):
    try:
        ser = serial.Serial(port, 115200, timeout=1)
    except Exception as e:
        with lock:
            state["cam_error"] = "open failed: %s" % e
        return

    time.sleep(0.5)
    ser.write(b"\x03")
    time.sleep(0.2)
    ser.reset_input_buffer()
    ser.write(b"\r\n")
    read_to_prompt(ser, 3)

    current, grab, kind = None, None, "jpeg"
    bad = (lepton_fix.DEAD_ROWS, lepton_fix.OFFSET_ROWS)
    regain = lepton_fix.Regain()
    unwrap = frame_clock.Unwrapper()
    att = attitude_mod.Attitude()
    epoch = 0
    link = frame_clock.RateMeter()
    rec = REC
    ticks, t_last, empty = 0, time.time(), 0
    t_frame = time.time()

    while True:
        try:
            with lock:
                want = desired["camera"]
            if want != current:
                with lock:
                    state["switching"] = True
                soft_reset(ser)
                grab, kind = init_camera(ser, want, opts.quality, opts.agc_damping,
                                         opts.imu)
                unwrap.reset()
                epoch += 1        # the board clock restarted; see frame_clock.stamp
                att.reset()       # do not integrate across the clock discontinuity
                if kind == "raw":
                    bad, how = learn_bad_rows(ser, grab, opts)
                    regain = lepton_fix.Regain()
                    print("lepton rows  dead: %s\n             offset: %s  (%s)"
                          % (list(bad[0]), list(bad[1]), how))
                current = want
                with lock:
                    state["camera"] = current
                    state["switching"] = False
                    state["cam_error"] = None
                    state["dead_rows"] = list(bad[0]) if kind == "raw" else []
                    state["offset_rows"] = list(bad[1]) if kind == "raw" else []
                    for c in CAMERAS:                 # the one we just left is
                        state["fps"][c] = 0.0         # not producing any more
                    state["cam_fps"] = 0.0
                ticks, t_last, empty = 0, time.time(), 0

            # Run FFC here, between grabs, on the capture thread. It must not be
            # fired from the HTTP handler: two threads writing the same REPL
            # interleave their statements and the reply to one is read as the
            # reply to the other. The shutter takes ~1 s and the frames during it
            # are the shutter itself, so they are dropped rather than shown.
            with lock:
                do_cal, ATT["calibrate"] = ATT["calibrate"], False
                do_clear, ATT["clear_fpn"] = ATT["clear_fpn"], False
            if do_clear:
                FPN["map"], FPN["note"] = None, "cleared"
                try:
                    os.remove(FPN_PATH)
                except OSError:
                    pass
                with lock:
                    state["fpn"] = FPN["note"]
            if do_cal:
                if kind != "raw":
                    with lock:
                        state["cam_error"] = ("pixel calibration applies to the "
                                              "thermal camera only")
                else:
                    shots = []
                    n_want = lepton_fix.FPN_MIN_FRAMES + 20
                    with lock:
                        state["calibrating"] = 0
                    while len(shots) < n_want:
                        gg = grab_payload(ser, grab, opts.imu)
                        f = lepton_fix.decode(gg.payload)
                        if f is not None:
                            f = lepton_fix.repair(
                                lepton_fix.destripe(f, bad[0], bad[1]), bad[0])
                            shots.append(f)
                            with lock:
                                state["calibrating"] = len(shots)
                        elif gg.error:
                            break
                    off, note = lepton_fix.measure_pixel_offsets(shots, bad[0])
                    with lock:
                        state["calibrating"] = None
                    if off is None:
                        FPN["note"] = "rejected: " + note
                        with lock:
                            state["cam_error"] = FPN["note"]
                            state["fpn"] = FPN["note"]
                    else:
                        FPN["map"], FPN["note"] = off, note
                        try:
                            np.save(FPN_PATH, off)
                            FPN["note"] += " (saved)"
                        except Exception as e:      # noqa: BLE001
                            FPN["note"] += " (NOT saved: %s)" % e
                        with lock:
                            state["cam_error"] = None
                            state["fpn"] = FPN["note"]
                    print("pixel cal  : %s" % FPN["note"])

            with lock:
                do_ffc, ATT["ffc"] = ATT["ffc"], False
            if do_ffc:
                if kind != "raw":
                    with lock:
                        state["cam_error"] = "FFC applies to the thermal camera only"
                else:
                    with lock:
                        state["ffc_running"] = True
                    resp = repl(ser, "c.ioctl(csi.IOCTL_LEPTON_RUN_COMMAND, 0x0242)",
                                timeout=20)
                    time.sleep(1.2)             # shutter close, measure, reopen
                    for _ in range(3):
                        grab_payload(ser, grab, opts.imu)   # discard shutter frames
                    with lock:
                        state["ffc_running"] = False
                        state["cam_error"] = (last_line(resp)[:120]
                                              if resp and "Error" in resp else None)

            g = grab_payload(ser, grab, opts.imu)
            if g.error:
                with lock:
                    state["cam_error"] = g.error
            elif g.payload:
                now = time.time()
                dt, t_frame = min(now - t_frame, 1.0), now
                if kind == "raw":
                    # Decision path: repaired pre-regain frame -> warm-blob
                    # detection -> azimuth association with the radar's live
                    # clusters -> uncertainty-fused objects. Display gets the
                    # same objects drawn on top; state.json gets the records.
                    dframe = repair_thermal(g.payload, bad, opts)
                    objects, clus = None, []
                    if dframe is not None and not opts.no_fusion:
                        boxes = thermal_detect.detect(dframe, bad[0])
                        with ai_cv:
                            if AI["on"]:
                                AI["frame"] = dframe
                                ai_cv.notify()
                                boxes = detect_merge.merge_thermal_boxes(
                                    boxes, AI["boxes"])
                        with lock:
                            clus = [dict(c)
                                    for c in state["radar"]["clusters"]]
                        objects = az_fuse.fuse_thermal(
                            boxes, clus, dframe.shape[1])
                        with lock:
                            state["fusion"] = objects
                    jpeg = thermal_to_jpeg(g.payload, bad, regain, opts, dt,
                                           frame=dframe, objects=objects,
                                           clusters=clus)
                else:
                    jpeg = g.payload
                if jpeg:
                    # mono_us is the board's own axis, unwrapped past the 2**30
                    # rollover. host_wall is the Jetson's clock and is for
                    # humans only -- it carries USB and REPL latency and can
                    # step. They are never mixed.
                    mono = unwrap.unwrap(g.mono_us) if g.mono_us is not None else None
                    with frame_cv:
                        frames["jpeg"] = jpeg
                        frames["seq"] += 1
                        frames["mono_us"] = mono
                        frames["host_wall"] = now
                        frame_cv.notify_all()
                    # Every IMU sample carries its OWN board timestamp. Place it
                    # on the unwrapped axis by its signed distance from the frame
                    # stamp rather than by unwrapping it directly -- the frame
                    # stamp has already gone through the unwrapper, and feeding
                    # it older values afterwards would walk the state backwards.
                    imu_seq = []
                    if mono is not None:
                        for s in g.imu_samples:
                            d = frame_clock.ticks_diff(g.mono_us, s[0])
                            imu_seq.append((mono - d, s[1:4], s[4:7]))

                    if rec is not None and mono is not None:
                        rec.frame(frame_clock.stamp(frames["seq"], mono, now,
                                                    len(g.payload), current, epoch),
                                  g.payload)
                        # One line per SAMPLE, not one per frame. The ring runs
                        # at 200 Hz; recording only the newest reading throws
                        # away 95% of it and reintroduces the 8.77 Hz jitter the
                        # ring exists to remove.
                        for ts, a, gy in imu_seq:
                            rec.imu(ts, a, gy, epoch)
                    # Consume the UI's one-shot requests and run the attitude
                    # filter OUTSIDE the state lock: `lock` is a plain Lock, so
                    # taking it again while already held would deadlock the
                    # capture thread outright.
                    if g.imu:
                        with lock:
                            want_level, ATT["level"] = ATT["level"], False
                            want_rh, ATT["reset_heading"] = ATT["reset_heading"], False
                        if want_level:
                            att.level(g.imu[:3])
                        if want_rh:
                            att.reset_heading()
                        # Integrate every sample, oldest first. Handing the
                        # filter one reading per frame is the 8.77 Hz path the
                        # ring was built to replace, and gyro integration is
                        # exactly where the missing samples show up as drift.
                        for ts, a, gy in imu_seq:
                            att.update(a, gy, ts)
                        if not imu_seq:
                            att.update(g.imu[:3], g.imu[3:], mono or 0)
                    with lock:
                        state["cam_error"] = None
                        state["frame"] = {"seq": frames["seq"], "mono_us": mono,
                                          "epoch": epoch, "host_wall": round(now, 3)}
                        if g.imu:
                            state["attitude"] = att.as_dict()
                            state["imu"] = {
                                "accel_mg": [round(v, 1) for v in g.imu[:3]],
                                "gyro_mdps": [round(v, 1) for v in g.imu[3:]],
                                "mono_us": mono}
                else:
                    with lock:
                        state["cam_error"] = "short frame: %d bytes" % len(g.payload)
            else:
                # No error, no payload. The board answered the REPL and returned
                # nothing -- which is what a wedged sensor looks like from here.
                # Counted, because an empty round trip is fast and would
                # otherwise inflate the frame rate: measured 64 "fps" against
                # seq = 0 and 0 B/s on the wire, with cam_error still None.
                empty += 1
                if empty >= 20:
                    with lock:
                        state["cam_error"] = (
                            "%d empty grabs in a row -- the board is answering "
                            "but sending no frame. Usually its USB stack or the "
                            "sensor is wedged; replug the board." % empty)

            if link.add(g.wire_len, time.time()):
                with lock:
                    state["link"]["openmv_bps"] = round(link.bps)
                    state["link"]["openmv_peak_bps"] = round(link.peak_bps)
            # Count DELIVERED frames, not loop iterations. The two are the same
            # only while the board is healthy, and they diverge exactly when
            # something is wrong -- which is the moment the number is read.
            if g.payload:
                ticks += 1
                empty = 0
            now = time.time()
            if now - t_last >= 2.0:
                rate = round(ticks / (now - t_last), 2)
                with lock:
                    state["cam_fps"] = rate
                    state["fps"][current] = rate
                ticks, t_last = 0, now
        except Exception as e:
            with lock:
                state["cam_error"] = "%s: %s" % (type(e).__name__, e)
            time.sleep(1.0)


BAUD_BYTES_PER_S = 921600 / 10.0        # 8N1: 10 bits on the wire per byte


def send_radar_cfg(cfg_path):
    """Push a .cfg over the radar CLI port and return what to record about it.

    Returns None when no --cfg was given, so the caller can say out loud that
    the session does not know its own configuration. Raises nothing on a config
    the radar rejects -- it prints and returns None, because a live view that
    refuses to start over a chirp-config typo is worse than one that starts on
    the previous config and says so.

    The recorded `lines` are what was SENT, taken from send_config's own log,
    not a re-read of the file: if the file changes tomorrow the session still
    knows what it ran.
    """
    if not cfg_path:
        return None
    cli = by_id("*XDS110*if00")
    if not cli:
        print("radar cfg  : CLI port not found under %s -- cfg NOT sent" % BY_ID)
        return None
    try:
        with serial.Serial(cli, 115200, timeout=2) as ser:
            log = iwr1843_uart.send_config(ser, cfg_path)
    except Exception as e:                     # noqa: BLE001 - report, don't die
        print("radar cfg  : FAILED (%s) -- radar left on its previous config" % e)
        return None

    text = open(cfg_path).read()
    geom = chirp_geometry.from_cfg_text(text)
    print("radar cfg  : %s accepted (%d lines)"
          % (os.path.basename(cfg_path), len(log)))
    if geom:
        print("             %s" % chirp_geometry.summary_line(geom))
        if geom["slots_wasted"]:
            print("             WARNING: %d chirp slot(s) transmit nothing; "
                  "v_max is worse than the TX count suggests"
                  % geom["slots_wasted"])
    return {
        "name": os.path.basename(cfg_path),
        "path": cfg_path,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
        "lines": [line for line, _resp in log],
        "derived": geom,
        "sent": True,
    }


def radar_thread(port):
    # timeout=0.005, not 0.1. With a 100 ms frame period a 0.1 s timeout blocks
    # the full timeout, so a frame sits in the kernel buffer for 0-100 ms with a
    # mean of ~50 ms of latency that is never recorded. A tight loop stamping
    # time.monotonic() right after each read collapses batches to one frame and
    # makes arrival timing meaningful. Measured after this change: batch_n == 1
    # on 249/249 frames.
    try:
        ser = serial.Serial(port, 921600, timeout=0.005)
    except Exception as e:
        with lock:
            state["radar_error"] = "open failed: %s" % e
        return

    reader = RadarReader()
    ser.reset_input_buffer()
    # The radar's own timeCpuCycles is the good clock here: measured 200.000 MHz
    # with 9.0 us of jitter, against milliseconds for anything host-stamped. It
    # dates the measurement, not its arrival, so it is what Delta-t comes from.
    cyc = frame_clock.Unwrapper(period=iwr1843_uart.CYCLES_PERIOD)
    frames = raw = kept = 0
    link = frame_clock.RateMeter()
    t_last = time.time()
    while True:
        try:
            n = ser.in_waiting
            chunk = ser.read(n) if n else ser.read(1)
            t_arr = time.monotonic()        # monotonic: wall clock can step
            if not chunk:
                continue
            if link.add(len(chunk), time.time()):
                with lock:
                    state["link"]["radar_bps"] = round(link.bps)
                    state["link"]["radar_peak_bps"] = round(link.peak_bps)
            for fr in reader.feed(chunk):
                frames += 1
                pts = fr["points"]
                raw += len(pts)
                if REC is not None:
                    # Raw and UNGATED on purpose. gate_points' isolation filter
                    # drops points with no neighbour within 1.2 m, which is
                    # exactly the sparse world-fixed returns odometry runs on --
                    # it measured kept=0 out of 39 raw on this scene. Recording
                    # its output would make the dataset unusable for that, the
                    # same argument recorder.py makes for raw thermal frames.
                    REC.radar_frame({
                        "frame": fr["frame"],
                        "cycles_raw": fr["cycles"],
                        "cycles": cyc.unwrap(fr["cycles"]),
                        "t_arr_mono": round(t_arr - fr["bytes_after"]
                                            / BAUD_BYTES_PER_S, 6),
                        "batch_i": fr["batch_i"], "batch_n": fr["batch_n"],
                        "bytes_after": fr["bytes_after"],
                        "n_obj": fr["n_obj"], "pts": [list(p) for p in pts],
                        "snr": fr["snr"], "noise": fr["noise"],
                        "stats": fr["stats"]})
                keep, _ = radar_gate.gate_points(pts, fr["snr"], fr["noise"])
                kept += len(keep)
                with lock:
                    state["radar"]["clusters"] = radar_classify_n6.classify_frame(keep)[:12]
            now = time.time()
            if now - t_last >= 1.0:
                rate = round(frames / (now - t_last), 1)
                with lock:
                    state["radar"].update(frames=frames, raw=raw, kept=kept, fps=rate)
                    state["fps"]["radar"] = rate
                    state["radar_error"] = None
                frames = raw = kept = 0
                t_last = now
        except Exception as e:
            with lock:
                state["radar_error"] = "%s: %s" % (type(e).__name__, e)
            time.sleep(1.0)


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ValiSight live</title><style>
:root{--bg:#0b1012;--panel:#121a1c;--line:#243033;--ink:#e6eeef;--dim:#7a8a8d;
--accent:#33c2cc;--warn:#d3a04a;--mono:ui-monospace,Menlo,Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--mono);padding:20px}
.wrap{max-width:1000px;margin:0 auto}
h1{font-size:15px;letter-spacing:.16em;text-transform:uppercase;margin:0 0 4px}
.sub{color:var(--dim);font-size:12px;margin-bottom:20px}
.grid{display:grid;grid-template-columns:minmax(280px,1.1fr) minmax(260px,1fr);gap:16px}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:3px;overflow:hidden}
.tabs{display:flex;border-bottom:1px solid var(--line)}
.tabs button{flex:1;background:none;border:none;color:var(--dim);font-family:var(--mono);
font-size:11px;letter-spacing:.12em;text-transform:uppercase;padding:10px;cursor:pointer}
.tabs button:hover{color:var(--ink)}
.tabs button[aria-selected="true"]{color:var(--accent);box-shadow:inset 0 -2px 0 var(--accent)}
.tabs button:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
/* pixelated by default: at 160x120 magnified ~3.5x every sensor pixel becomes a
   hard block, which is what you want when judging dead rows or a repair map and
   is exactly what reads as "blurry stripes" when you just want to see the scene.
   The .smooth class hands that choice to the viewer -- it changes nothing in the
   data, only how the browser interpolates between pixels it was given. */
#img{display:block;width:100%;background:#000;image-rendering:pixelated;min-height:200px}
#img.smooth{image-rendering:auto}
.cap{padding:9px 13px;border-top:1px solid var(--line);font-size:11.5px;color:var(--dim);
display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}
h2{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--dim);
margin:0;padding:11px 13px;border-bottom:1px solid var(--line)}
.stats{display:grid;grid-template-columns:repeat(2,1fr);gap:1px;background:var(--line)}
.stat{background:var(--panel);padding:11px 13px}
.stat .k{font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim)}
.stat .v{font-size:19px;font-weight:600;font-variant-numeric:tabular-nums;margin-top:2px}
table{border-collapse:collapse;width:100%;font-size:11.5px}
th{text-align:left;font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim);
padding:7px 13px;border-bottom:1px solid var(--line);border-top:1px solid var(--line)}
td{padding:5px 13px;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}
.pedestrian{color:var(--accent)}.vehicle{color:var(--warn)}.static{color:var(--dim)}
.err{color:var(--warn);font-size:11.5px;padding:0 13px 11px;min-height:14px}
.rates{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);
border:1px solid var(--line);border-radius:3px;margin-bottom:16px}
.rate{background:var(--panel);padding:10px 13px;display:flex;align-items:baseline;
justify-content:space-between;gap:8px}
.rate .k{font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--dim)}
.rate .v{font-size:17px;font-weight:600;font-variant-numeric:tabular-nums}
.rate .u{font-size:10px;color:var(--dim);margin-left:3px;font-weight:400}
.rate.idle .v{color:var(--dim)}
.rate.live .v{color:var(--accent)}
.hud{display:flex;gap:1px;background:var(--line);flex-wrap:wrap}
.hud canvas{background:#070c0d;display:block;flex:0 0 auto}
.hudnums{background:var(--panel);flex:1 1 200px;display:grid;
grid-template-columns:repeat(2,1fr);gap:1px;background:var(--line);align-content:start}
.hudnums .stat{padding:9px 13px}
.hudnums .v{font-size:17px}
.hudnums .v.sm{font-size:12px}
.hudbar{display:flex;gap:10px;align-items:center;padding:10px 13px;
border-top:1px solid var(--line);flex-wrap:wrap}
.hudbar button{background:#182224;border:1px solid var(--line);color:var(--ink);
font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
padding:7px 12px;border-radius:2px;cursor:pointer}
.hudbar button:hover{border-color:var(--accent);color:var(--accent)}
.hudbar span{font-size:11px;color:var(--dim)}
</style></head><body><div class="wrap">
<h1>ValiSight &mdash; live</h1>
<div class="sub">OpenMV N6 + IWR1843BOOST &middot; <span id="clock"></span></div>
<div class="rates">
  <div class="rate" id="f-rgb"><span class="k">RGB</span>
    <span><span class="v">&mdash;</span><span class="u">fps</span></span></div>
  <div class="rate" id="f-thermal"><span class="k">Thermal</span>
    <span><span class="v">&mdash;</span><span class="u">fps</span></span></div>
  <div class="rate" id="f-radar"><span class="k">Radar</span>
    <span><span class="v">&mdash;</span><span class="u">fps</span></span></div>
</div>
<div class="card" style="margin-bottom:16px">
  <h2>SoC links &mdash; sustained throughput</h2>
  <div class="stats" style="grid-template-columns:repeat(4,1fr)">
    <div class="stat"><div class="k">OpenMV VCP</div><div class="v" id="l-omv">0</div>
      <div class="k" id="l-omv-pk">peak &mdash;</div></div>
    <div class="stat"><div class="k">Radar UART</div><div class="v" id="l-rad">0</div>
      <div class="k" id="l-rad-pk">peak &mdash;</div></div>
    <div class="stat"><div class="k">frame mono_us</div><div class="v" id="l-mono">&mdash;</div>
      <div class="k">N6 ticks_us, unwrapped</div></div>
    <div class="stat"><div class="k">frame seq</div><div class="v" id="l-seq">0</div>
      <div class="k" id="l-wall">host wall &mdash;</div></div>
  </div>
  <div class="err" id="linknote" style="color:var(--dim)"></div>
</div>
<div class="card" id="imucard" style="margin-bottom:16px">
  <h2>Attitude &mdash; OpenMV N6 IMU</h2>
  <div class="hud">
    <canvas id="ball" width="260" height="260" aria-label="attitude indicator"></canvas>
    <canvas id="ring" width="260" height="260" aria-label="relative heading"></canvas>
    <div class="hudnums">
      <div class="stat"><div class="k">roll</div><div class="v" id="a-r">&mdash;</div></div>
      <div class="stat"><div class="k">pitch</div><div class="v" id="a-p">&mdash;</div></div>
      <div class="stat"><div class="k">tilt from level</div><div class="v" id="a-t">&mdash;</div></div>
      <div class="stat"><div class="k">heading (relative)</div><div class="v" id="a-y">&mdash;</div>
        <div class="k" id="a-d">&nbsp;</div></div>
      <div class="stat"><div class="k">accel mg</div><div class="v sm" id="i-a">&mdash;</div></div>
      <div class="stat"><div class="k">gyro mdps</div><div class="v sm" id="i-g">&mdash;</div>
        <div class="k" id="i-m">&nbsp;</div></div>
    </div>
  </div>
  <div class="hudbar">
    <button onclick="fetch('/level',{cache:'no-store'})">set level here</button>
    <button onclick="fetch('/reset-heading',{cache:'no-store'})">zero heading</button>
    <span id="a-cal"></span>
  </div>
  <div class="err" id="a-note" style="color:var(--dim)"></div>
</div>
<div class="grid">
  <div class="card">
    <div class="tabs" role="tablist">
      <button role="tab" id="t-rgb" aria-selected="true" onclick="pick('rgb')">RGB &middot; PAG7936</button>
      <button role="tab" id="t-thermal" aria-selected="false" onclick="pick('thermal')">Thermal &middot; Lepton</button>
    </div>
    <img id="img" src="/stream.mjpg" alt="live camera frame">
    <!-- Palette is a host-side LUT, so unlike the camera tabs above it takes
         effect on the very next frame with no soft reset. Only meaningful on
         thermal; hidden on RGB rather than shown disabled, because a control
         that does nothing invites the belief that it did something. -->
    <div class="tabs" id="palrow" role="group" aria-label="thermal palette" hidden>
      <button id="p-blackhot" onclick="setpal('blackhot')">Black-hot</button>
      <button id="p-whitehot" onclick="setpal('whitehot')">White-hot</button>
      <button id="p-ironbow" onclick="setpal('ironbow')">Ironbow</button>
      <button id="p-cal" onclick="calib()" title="pan the camera slowly while this runs">Calibrate pixels</button>
      <button id="p-calclr" onclick="calib(1)" title="discard the per-pixel map">Clear</button>
      <button id="p-ffc" onclick="ffc()" title="drop the shutter and re-measure every pixel's offset">Run FFC</button>
    </div>
    <div class="tabs" role="group" aria-label="display">
      <button id="p-smooth" onclick="smooth()" title="interpolate between sensor pixels -- display only, the data is unchanged">Smooth</button>
    </div>
    <div class="cap"><span id="camname">&mdash;</span><span id="camfps">&mdash;</span></div>
    <div class="err" id="camerr"></div>
  </div>
  <div class="card">
    <h2>Radar</h2>
    <div class="stats">
      <div class="stat"><div class="k">frames/s</div><div class="v" id="rfps">0</div></div>
      <div class="stat"><div class="k">points raw</div><div class="v" id="rraw">0</div></div>
      <div class="stat"><div class="k">kept</div><div class="v" id="rkept">0</div></div>
      <div class="stat"><div class="k">clusters</div><div class="v" id="rcl">0</div></div>
    </div>
    <table><thead><tr><th>Label</th><th>Range m</th><th>Doppler</th><th>Pts</th></tr></thead>
    <tbody id="tb"></tbody></table>
    <div class="err" id="raderr"></div>
  </div>
</div></div><script>
const CSS = getComputedStyle(document.documentElement);
const ACC = CSS.getPropertyValue('--accent').trim() || '#33c2cc';
const DIM = CSS.getPropertyValue('--dim').trim() || '#7a8a8d';
const WARN = CSS.getPropertyValue('--warn').trim() || '#d3a04a';

// Attitude ball: the horizon rolls and slides with pitch, the aircraft mark
// stays fixed. Pitch is clamped for DISPLAY only -- the readout stays true.
function drawBall(at){
  const cv=document.getElementById('ball'), x=cv.getContext('2d');
  const w=cv.width, h=cv.height, cx=w/2, cy=h/2, R=Math.min(w,h)/2-14;
  x.clearRect(0,0,w,h);
  x.save(); x.beginPath(); x.arc(cx,cy,R,0,7); x.clip();
  x.translate(cx,cy); x.rotate(-at.roll*Math.PI/180);
  const px = Math.max(-90,Math.min(90,at.pitch))*(R/90);
  x.fillStyle='#16323a'; x.fillRect(-R*2,-R*2+px,R*4,R*2);      // sky
  x.fillStyle='#0b1416'; x.fillRect(-R*2,px,R*4,R*2);           // ground
  x.strokeStyle=ACC; x.lineWidth=1.5;
  x.beginPath(); x.moveTo(-R*1.5,px); x.lineTo(R*1.5,px); x.stroke();
  x.strokeStyle=DIM; x.lineWidth=1; x.font='9px monospace'; x.fillStyle=DIM;
  for(let p=-60;p<=60;p+=15){ if(!p) continue;
    const y=px-p*(R/90), len=(p%30===0)?R*0.42:R*0.22;
    x.beginPath(); x.moveTo(-len,y); x.lineTo(len,y); x.stroke();
    if(p%30===0){ x.textAlign='left'; x.fillText(p+'\\u00b0', len+4, y+3); }
  }
  x.restore();
  x.strokeStyle=DIM; x.lineWidth=1; x.beginPath(); x.arc(cx,cy,R,0,7); x.stroke();
  x.strokeStyle=WARN; x.lineWidth=2;                            // fixed aircraft mark
  x.beginPath(); x.moveTo(cx-26,cy); x.lineTo(cx-9,cy);
  x.moveTo(cx+9,cy); x.lineTo(cx+26,cy);
  x.moveTo(cx,cy-5); x.lineTo(cx,cy+5); x.stroke();
  x.fillStyle=DIM; x.font='9px monospace'; x.textAlign='center';
  x.fillText('ROLL / PITCH \\u00b7 from gravity, no drift', cx, h-4);
}

// Heading ring. The wedge is the accumulated 1-sigma drift, drawn to scale --
// once it swallows the ring the heading means nothing and that is the point.
function drawRing(at){
  const cv=document.getElementById('ring'), x=cv.getContext('2d');
  const w=cv.width,h=cv.height,cx=w/2,cy=h/2,R=Math.min(w,h)/2-14;
  x.clearRect(0,0,w,h);
  x.strokeStyle=DIM; x.lineWidth=1; x.beginPath(); x.arc(cx,cy,R,0,7); x.stroke();
  x.save(); x.translate(cx,cy); x.rotate(-at.yaw*Math.PI/180);
  x.strokeStyle=DIM; x.font='9px monospace'; x.fillStyle=DIM; x.textAlign='center';
  for(let d=0;d<360;d+=15){
    const a=d*Math.PI/180, maj=(d%45===0);
    x.beginPath();
    x.moveTo(Math.sin(a)*R, -Math.cos(a)*R);
    x.lineTo(Math.sin(a)*(R-(maj?11:6)), -Math.cos(a)*(R-(maj?11:6)));
    x.stroke();
    if(maj) x.fillText(d+'', Math.sin(a)*(R-22), -Math.cos(a)*(R-22)+3);
  }
  const dr=Math.min(at.drift_deg,180)*Math.PI/180;
  if(dr>0.004){
    x.beginPath(); x.moveTo(0,0);
    x.arc(0,0,R*0.82,-Math.PI/2-dr,-Math.PI/2+dr);
    x.closePath(); x.fillStyle='rgba(211,160,74,0.22)'; x.fill();
  }
  x.restore();
  x.strokeStyle=ACC; x.lineWidth=2; x.fillStyle=ACC;            // fixed nose mark
  x.beginPath(); x.moveTo(cx,cy-R+3); x.lineTo(cx-6,cy-R+15); x.lineTo(cx+6,cy-R+15);
  x.closePath(); x.fill();
  x.strokeStyle=DIM; x.lineWidth=1;
  x.beginPath(); x.moveTo(cx,cy-9); x.lineTo(cx,cy+9);
  x.moveTo(cx-9,cy); x.lineTo(cx+9,cy); x.stroke();
  x.fillStyle=DIM; x.font='9px monospace'; x.textAlign='center';
  x.fillText('HEADING \\u00b7 relative, gyro-integrated', cx, h-4);
}

let want = 'rgb';
async function pick(c){
  want = c;
  document.getElementById('t-rgb').setAttribute('aria-selected', c==='rgb');
  document.getElementById('t-thermal').setAttribute('aria-selected', c==='thermal');
  try{ await fetch('/switch?cam='+c, {cache:'no-store'}); }catch(e){}
}
const PALETTES = ['blackhot','whitehot','ironbow'];
function smooth(){
  const img = document.getElementById('img');
  const on  = img.classList.toggle('smooth');
  const b   = document.getElementById('p-smooth');
  b.setAttribute('aria-selected', on);
  b.textContent = on ? 'Pixels' : 'Smooth';
}
async function calib(clear){
  const b = document.getElementById('p-cal');
  if(clear){ try{ await fetch('/calibrate?clear=1',{cache:'no-store'}); }catch(e){} return; }
  b.disabled = true; b.textContent = 'PAN NOW\\u2026';
  try{ await fetch('/calibrate', {cache:'no-store'}); }catch(e){}
  setTimeout(()=>{ b.disabled=false; b.textContent='Calibrate pixels'; }, 12000);
}
async function ffc(){
  const b = document.getElementById('p-ffc');
  b.disabled = true; b.textContent = 'FFC\\u2026';
  try{ await fetch('/ffc', {cache:'no-store'}); }catch(e){}
  setTimeout(()=>{ b.disabled = false; b.textContent = 'Run FFC'; }, 2500);
}
async function setpal(p){
  // Do not paint the selection optimistically -- the server rejects an unknown
  // name, and tick() reflects what it actually applied.
  try{ await fetch('/palette?p='+p, {cache:'no-store'}); }catch(e){}
}
function showpal(cam, active){
  document.getElementById('palrow').hidden = (cam !== 'thermal');
  for(const p of PALETTES){
    const b = document.getElementById('p-'+p);
    if(b) b.setAttribute('aria-selected', p === active);
  }
}
async function tick(){
  try{
    const s = await (await fetch('/state.json',{cache:'no-store'})).json();
    // The <img> holds a live MJPEG connection; only the caption changes here.
    const dead = (s.dead_rows||[]).length;
    document.getElementById('camname').textContent = s.switching
      ? 'switching camera\\u2026 (soft reset)'
      : (s.camera==='rgb' ? 'PAG7936 RGB \\u00b7 320\\u00d7200'
                          : 'Lepton 3.5 \\u00b7 160\\u00d7120 \\u00b7 AGC damped \\u00b7 '
                            + (s.palette || 'blackhot')
                            + (dead ? ' \\u00b7 ' + dead + ' dead rows filled' : ''));
    document.getElementById('camfps').textContent = s.cam_fps + ' fps';
    showpal(s.camera, s.palette);
    // A camera that is not selected reports 0, which is a fact about the
    // hardware (they share the CSI bus) and not a stall -- say "idle", and
    // keep "0.00" free to mean the selected sensor has actually stopped.
    const f = s.fps||{};
    for(const k of ['rgb','thermal','radar']){
      const el = document.getElementById('f-'+k);
      const v = f[k]||0, on = v > 0;
      el.className = 'rate ' + (on ? 'live' : 'idle');
      el.querySelector('.v').textContent =
        on ? v.toFixed(k==='radar'?1:2) : (k==='radar' ? '\\u2014' : 'idle');
      el.querySelector('.u').style.visibility = on ? 'visible' : 'hidden';
    }
    const kb = b => b >= 1048576 ? (b/1048576).toFixed(2)+' MB/s'
                  : b >= 1024    ? (b/1024).toFixed(1)+' KB/s'
                                 : b+' B/s';
    const L = s.link||{};
    document.getElementById('l-omv').textContent = kb(L.openmv_bps||0);
    document.getElementById('l-rad').textContent = kb(L.radar_bps||0);
    document.getElementById('l-omv-pk').textContent = 'peak '+kb(L.openmv_peak_bps||0);
    document.getElementById('l-rad-pk').textContent = 'peak '+kb(L.radar_peak_bps||0);
    const fm = s.frame||{};
    // Shown modulo 10^6 us: the full unwrapped value is many digits and the
    // low microseconds are the part worth watching tick over.
    document.getElementById('l-mono').textContent =
      fm.mono_us==null ? '\\u2014' : (fm.mono_us%1000000).toString().padStart(6,'0');
    document.getElementById('l-seq').textContent = fm.seq||0;
    document.getElementById('l-wall').textContent = fm.host_wall==null ? 'host wall \\u2014'
      : 'host wall '+new Date(fm.host_wall*1000).toLocaleTimeString();
    document.getElementById('linknote').textContent =
      'USB CDC full-speed ceiling is ~1.5 MB/s raw; base64 costs 4 bytes per 3.';
    const q = s.imu, at = s.attitude;
    if(q){
      document.getElementById('i-a').textContent = q.accel_mg.join(' / ');
      document.getElementById('i-g').textContent = q.gyro_mdps.join(' / ');
      document.getElementById('i-m').textContent =
        '|a| ' + Math.hypot(...q.accel_mg).toFixed(0) + ' mg';
    }
    if(at){
      document.getElementById('a-r').textContent = at.roll.toFixed(1) + '\\u00b0';
      document.getElementById('a-p').textContent = at.pitch.toFixed(1) + '\\u00b0';
      document.getElementById('a-t').textContent = at.tilt.toFixed(1) + '\\u00b0';
      document.getElementById('a-y').textContent = at.yaw.toFixed(1) + '\\u00b0';
      document.getElementById('a-d').textContent =
        at.heading_age_s > 0 ? '\\u00b1' + at.drift_deg.toFixed(1)
                               + '\\u00b0 drift after ' + at.heading_age_s.toFixed(0) + ' s'
                             : 'not integrating yet';
      document.getElementById('a-cal').textContent = at.calibrated
        ? 'gyro bias calibrated: ' + at.bias_dps.join(' / ') + ' dps'
          + (at.stationary ? ' \\u00b7 at rest' : ' \\u00b7 moving')
        : 'hold still to calibrate the gyro bias\\u2026';
      document.getElementById('a-note').textContent =
        'Roll and pitch come from gravity and do not drift. Heading is '
        + 'gyro-integrated with no magnetometer, so it is RELATIVE and drifts '
        + '~26\\u00b0/min uncorrected. Position is not shown because '
        + 'double-integrating this accelerometer gives ~1 m of error in 10 s '
        + 'and ~35 m in 60 s.';
      drawBall(at); drawRing(at);
    }
    document.getElementById('camerr').textContent = s.cam_error||'';
    document.getElementById('raderr').textContent = s.radar_error||'';
    const rd = s.radar;
    document.getElementById('rfps').textContent = rd.fps;
    document.getElementById('rraw').textContent = rd.raw;
    document.getElementById('rkept').textContent = rd.kept;
    document.getElementById('rcl').textContent = (rd.clusters||[]).length;
    document.getElementById('tb').innerHTML = (rd.clusters||[]).map(c=>
      `<tr><td class="${c.label}">${c.label}</td><td>${c.range_m}</td>`+
      `<td>${c.doppler_mps}</td><td>${c.n_points}</td></tr>`).join('');
    document.getElementById('clock').textContent = new Date().toLocaleTimeString();
  }catch(e){}
  setTimeout(tick, 400);
}
tick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        last = -1
        try:
            while True:
                with frame_cv:
                    frame_cv.wait_for(lambda: frames["seq"] != last, timeout=5.0)
                    jpeg, last = frames["jpeg"], frames["seq"]
                if not jpeg:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                 b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n")
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass          # viewer navigated away

    def do_GET(self):
        if self.path.startswith("/stream.mjpg"):
            self._stream()
        elif self.path.startswith("/level"):
            with lock:
                ATT["level"] = True
            self._send(b'{"ok":true}', "application/json")
        elif self.path.startswith("/reset-heading"):
            with lock:
                ATT["reset_heading"] = True
            self._send(b'{"ok":true}', "application/json")
        elif self.path.startswith("/switch"):
            cam = "thermal" if "thermal" in self.path else "rgb"
            with lock:
                desired["camera"] = cam
            self._send(b'{"ok":true}', "application/json")
        elif self.path.startswith("/calibrate"):
            # Manual on purpose. The capture is only valid while the camera is
            # panning, and nothing here can know when that is true -- so it is
            # the operator who says "now", and the maths refuses the result if
            # the view did not actually change.
            with lock:
                if "clear" in self.path:
                    ATT["clear_fpn"] = True
                else:
                    ATT["calibrate"] = True
            self._send(b'{"ok":true}', "application/json")
        elif self.path.startswith("/ffc"):
            # Flat-field correction: the Lepton drops its mechanical shutter and
            # re-measures every pixel's offset. It is the ONLY thing that removes
            # per-pixel fixed-pattern noise -- destripe() is row-only, and
            # measured on 200 recorded frames the residual pattern is 2.06 DN
            # per-pixel against 0.86 row and 0.55 column, so the axis the
            # pipeline corrects is the smallest of the three. Frame averaging
            # does not help either: 8 frames buys 16%, because the residual is
            # fixed, not random.
            with lock:
                ATT["ffc"] = True
            self._send(b'{"ok":true}', "application/json")
        elif self.path.startswith("/palette"):
            # No soft reset and no dropped frame: the LUT is applied host-side
            # on the next frame, so the change is visible immediately. An
            # unknown name is rejected rather than silently falling back to the
            # default -- colourise() would swallow it and the UI would then show
            # a palette nobody selected.
            want = self.path.partition("=")[2].strip("/ ")
            if want in lepton_fix.PALETTES:
                with lock:
                    desired["palette"] = want
                    state["palette"] = want
                self._send(b'{"ok":true}', "application/json")
            else:
                self._send(json.dumps({"ok": False, "error": "unknown palette",
                                       "have": sorted(lepton_fix.PALETTES)}
                                      ).encode(), "application/json")
        elif self.path.startswith("/state.json"):
            with lock:
                body = json.dumps(state).encode()
            self._send(body, "application/json")
        else:
            self._send(PAGE.encode(), "text/html; charset=utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--quality", type=int, default=60)
    ap.add_argument("--camera", choices=CAMERAS, default="rgb")
    ap.add_argument("--palette", choices=sorted(lepton_fix.PALETTES), default="blackhot")
    ap.add_argument("--agc-damping", type=int, default=lepton_fix.AGC_DAMPING,
                    help="0-255 temporal damping on the sensor's AGC. Ships as 0, "
                         "which is the flicker; 200 measured best")
    ap.add_argument("--imu", action="store_true", default=True,
                    help="sample the N6 IMU in the same round trip as each frame")
    ap.add_argument("--no-imu", dest="imu", action="store_false")
    ap.add_argument("--record", action="store_true",
                    help="record this session under data/recordings/<timestamp>/")
    ap.add_argument("--record-root", default=recorder_mod.DEFAULT_ROOT,
                    help="where session directories are created (default: %(default)s)")
    ap.add_argument("--save-frames", choices=("none", "thermal", "rgb", "both"),
                    default="none",
                    help="also store frame bytes, RAW as received. thermal is "
                         "168 KB/s (606 MB/h), rgb about 230 KB/s (830 MB/h)")
    ap.add_argument("--record-max-mb", type=int, default=2048,
                    help="stop storing frame bytes past this; timing keeps logging")
    ap.add_argument("--no-regain", action="store_true",
                    help="skip reclaiming the range the dead rows occupied")
    ap.add_argument("--no-fusion", action="store_true",
                    help="skip thermal x radar fusion overlay (detection, "
                         "azimuth association, uncertainty) on the thermal view")
    ap.add_argument("--no-ai", action="store_true",
                    help="skip the learned person detector on thermal frames "
                         "(runs in its own thread; merged with the warm-blob "
                         "detector, never replacing it)")
    ap.add_argument("--no-repair", action="store_true",
                    help="show the dead rows instead of interpolating them")
    ap.add_argument("--cfg",
                    help="send this IWR1843 .cfg over the radar CLI port before "
                         "streaming, and record it in the session. Without it "
                         "the radar runs whatever was last pushed into it by "
                         "some other tool and the session cannot say what that "
                         "was -- which is how the 20260728 recording ended up "
                         "with an unknown configuration.")
    ap.add_argument("--detect-dead", action="store_true", default=True)
    ap.add_argument("--no-detect-dead", dest="detect_dead", action="store_false",
                    help="skip runtime detection, use the map measured on this module")
    args = ap.parse_args()

    # A map measured on a previous run is still valid: it describes the sensor,
    # not the session. Reload it rather than making someone re-pan every launch.
    if os.path.exists(FPN_PATH):
        try:
            FPN["map"] = np.load(FPN_PATH)
            FPN["note"] = "loaded %s (sd %.2f DN)" % (os.path.basename(FPN_PATH),
                                                      float(FPN["map"].std()))
        except Exception as e:                      # noqa: BLE001
            FPN["note"] = "could not load %s: %s" % (FPN_PATH, e)
        print("pixel cal  : %s" % FPN["note"])
    state["fpn"] = FPN["note"]

    desired["camera"] = args.camera
    desired["palette"] = args.palette
    state["palette"] = args.palette

    cam = by_id("*MicroPython*if00")
    radar_data = by_id("*XDS110*if03")
    print("openmv     : %s" % cam)
    print("radar DATA : %s" % radar_data)
    print("thermal    : AGC damping %d, %s, dead-row repair %s"
          % (args.agc_damping, args.palette, "off" if args.no_repair else "on"))

    radar_cfg = send_radar_cfg(args.cfg)
    if radar_cfg is None and args.record:
        print("radar cfg  : NOT SENT -- this session will not know which chirp "
              "config produced it. Pass --cfg to fix that.")

    global REC
    if args.record:
        REC = recorder_mod.Recorder(
            root=args.record_root, save_frames=args.save_frames,
            max_mb=args.record_max_mb,
            meta={"openmv_port": cam, "radar_port": radar_data,
                  "radar_cfg": radar_cfg,
                  "camera": args.camera, "quality": args.quality,
                  "palette": args.palette, "agc_damping": args.agc_damping,
                  "repair": not args.no_repair, "regain": not args.no_regain,
                  "imu": args.imu,
                  "dead_rows": list(lepton_fix.DEAD_ROWS),
                  "offset_rows": list(lepton_fix.OFFSET_ROWS),
                  "ticks_period_us": frame_clock.TICKS_PERIOD})
        print("recording  : %s  (frames: %s)" % (REC.dir, args.save_frames))
    else:
        print("recording  : off  (pass --record to collect a session)")

    if cam:
        threading.Thread(target=camera_thread, args=(cam, args), daemon=True).start()
    if radar_data:
        threading.Thread(target=radar_thread, args=(radar_data,), daemon=True).start()
    if not args.no_ai and not args.no_fusion:
        threading.Thread(target=ai_worker, daemon=True).start()

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print("\nserving on http://0.0.0.0:%d  (Ctrl-C to stop)" % args.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        if REC is not None:
            REC.close()
            print("\nrecording closed: %s" % REC.dir)


if __name__ == "__main__":
    main()

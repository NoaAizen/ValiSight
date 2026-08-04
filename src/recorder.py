"""On-disk recording layout for ValiSight capture sessions.

One directory per session, created when recording starts:

    data/recordings/20260728_154812/
        meta.json          what the rig was, once, at session start
        frames.jsonl       one line per frame: seq, sensor, mono_us, host_wall, bytes, file
        imu.jsonl          one line per IMU sample, stamped on the same mono_us axis
        radar.jsonl        one line per radar report
        thermal/000001.bin raw 19200-byte Lepton frames, exactly as the sensor gave them
        rgb/000001.jpg     board-encoded JPEG, byte-for-byte as received

WHAT GETS STORED, AND WHY IT IS THE RAW FRAME

thermal/*.bin holds the sensor's bytes BEFORE destripe, repair, regain and the
palette. That is deliberate. The repair pipeline is still being tuned, and a
recording of its output would bake today's parameters into the dataset
permanently -- you could never re-run a changed defect map over old data. The
raw frame plus meta.json is enough to reproduce any rendering exactly, and
lepton_fix is pure enough to replay it offline.

SIZE

Raw thermal is 19200 B/frame at 8.77 fps = 168 KB/s = 606 MB/hour. RGB JPEG is
about 6 KB/frame at 38 fps = 230 KB/s = 830 MB/hour. Frame recording is
therefore opt-in and capped; the JSONL streams are tiny (~100 B/frame) and
always written, so a long session can log timing and IMU without filling the
disk.

The JSONL files are append-only and flushed per line, so a session killed with
Ctrl-C or a yanked cable still leaves everything up to the last frame readable.
"""
import json
import os
import time

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "data", "recordings")


class Recorder:
    def __init__(self, root=None, save_frames="none", max_mb=2048, meta=None):
        self.root = root or DEFAULT_ROOT
        self.save_frames = save_frames        # none | thermal | rgb | both
        self.max_bytes = max_mb * 1024 * 1024
        self.written = 0
        self.capped = False
        self.session = time.strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(self.root, self.session)
        os.makedirs(self.dir, exist_ok=True)
        for sub in ("thermal", "rgb"):
            if self._wants(sub):
                os.makedirs(os.path.join(self.dir, sub), exist_ok=True)
        self._f = {}
        self.write_meta(meta or {})

    def _wants(self, sensor):
        return self.save_frames in ("both", sensor)

    def _stream(self, name):
        if name not in self._f:
            self._f[name] = open(os.path.join(self.dir, name + ".jsonl"), "a")
        return self._f[name]

    def write_meta(self, meta):
        """Idempotent on the started-* stamps: write_meta may be re-called
        mid-session (facts measured off the stream, e.g. actual frame dims,
        get stamped once known), and a re-call must not shift session start."""
        meta = dict(meta)
        if not hasattr(self, "_started"):
            self._started = (time.time(), time.strftime("%Y-%m-%dT%H:%M:%S"))
        meta.update(session=self.session,
                    started_host_wall=self._started[0],
                    started_iso=self._started[1],
                    save_frames=self.save_frames,
                    note="mono_us is the N6 monotonic axis; host_wall is the "
                         "Jetson clock and is for correlation only")
        with open(os.path.join(self.dir, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)

    def frame(self, rec, payload=None):
        """Log one frame; optionally store its bytes. rec comes from frame_clock.stamp."""
        rec = dict(rec)
        sensor = rec.get("sensor")
        if payload is not None and self._wants(sensor) and not self.capped:
            if self.written + len(payload) > self.max_bytes:
                # Stop writing frames but keep logging timing. Say so once --
                # a recording that silently stops is worse than a short one.
                self.capped = True
                self._stream("frames").write(json.dumps(
                    {"event": "frame_cap_reached", "bytes_written": self.written,
                     "host_wall": time.time()}) + "\n")
            else:
                ext = "bin" if sensor == "thermal" else "jpg"
                name = os.path.join(sensor, "%06d.%s" % (rec["seq"], ext))
                with open(os.path.join(self.dir, name), "wb") as fh:
                    fh.write(payload)
                self.written += len(payload)
                rec["file"] = name
        fh = self._stream("frames")
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

    def imu(self, mono_us, accel_mg, gyro_mdps, epoch=0):
        fh = self._stream("imu")
        fh.write(json.dumps({"epoch": epoch, "mono_us": mono_us,
                             "accel_mg": list(accel_mg),
                             "gyro_mdps": list(gyro_mdps)}) + "\n")
        fh.flush()

    def radar(self, rec):
        """1 Hz aggregate counters. Fine for the dashboard, useless for odometry."""
        fh = self._stream("radar")
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

    def radar_frame(self, rec):
        """One line per radar frame, raw and ungated.

        This is the stream odometry is developed against, so it stores the
        parser's tuples verbatim -- no gating, no rounding, no clustering. It
        carries the radar's own `cycles` (measured 200.000 MHz, 9 us jitter)
        alongside the host arrival stamp, because the former dates the
        measurement and the latter only dates its delivery.

        ~8.5 points/frame at 10 Hz is roughly 40 KB/s = 140 MB/hour: small
        enough to leave on for every session, unlike the frame payloads.
        """
        fh = self._stream("radar_frames")
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

    def clock_sync(self, rec):
        """One line per REPL clock ping, for the host <-> N6 cross-clock fit."""
        fh = self._stream("clock_sync")
        fh.write(json.dumps(rec) + "\n")
        fh.flush()

    def close(self):
        for fh in self._f.values():
            try:
                fh.close()
            except Exception:
                pass
        self._f.clear()

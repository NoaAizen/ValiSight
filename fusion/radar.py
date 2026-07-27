"""IWR1843 radar ingest for the fusion package.

Wraps the tested iwr1843_uart.RadarReader over pyserial in a background
thread, keeps a sliding window of points (sparse static returns need
aggregation before clustering), and reuses radar_classify_n6.classify_frame
for the coarse radar class. The radar is light-independent and penetrates
fog; thermal covers smoke/dust/darkness; fusion covers both (see the
radar-physics corpus — neither modality alone covers all degraded
visibility). Check alive() to know the link is actually up.
"""
import collections
import threading
import time

from iwr1843_uart import RadarReader, send_config, find_com_ports
from radar_classify_n6 import classify_frame
from core.timesync import sensor_time_from_frame

WINDOW_S = 0.5                  # sliding aggregation window
# configs/iwr1843_live.cfg frameCfg period is 100 ms -> 10 Hz. Sample time is
# reconstructed from frame_no at this cadence, NOT from UART arrival time.
FRAME_PERIOD_S = 0.1


class RadarSource(threading.Thread):
    """Background DATA-UART reader.

    latest_points() -> list of (x, y, z, doppler[, snr, noise]) from the last
    WINDOW_S seconds; clusters() -> classify_frame() over that window.
    """

    def __init__(self, data_port, window_s=WINDOW_S,
                 frame_period_s=FRAME_PERIOD_S):
        super().__init__(daemon=True)
        import serial
        self.ser = serial.Serial(data_port, 921600, timeout=0.05)
        self.reader = RadarReader()
        self.window_s = window_s
        self.frame_period_s = frame_period_s
        self.lock = threading.Lock()
        self.window = collections.deque()       # (sample_time_s, points)
        self.frames_rx = 0
        self.frames_dropped = 0                 # detected via frame_no gaps
        # sample-time reconstruction state (monotonic anchor + frame index)
        self._anchor = None
        self._first_frame = None
        self._last_frame = None
        self.running = True

    @classmethod
    def open(cls, cfg_port=None, data_port=None, cfg_path=None,
             send_cfg=True, window_s=WINDOW_S):
        """Auto-detect COM ports, optionally push the chirp config, start
        reading. Returns the running RadarSource."""
        import serial
        if not (cfg_port and data_port):
            auto_cfg, auto_data = find_com_ports()
            cfg_port = cfg_port or auto_cfg
            data_port = data_port or auto_data
        if not data_port:
            raise RuntimeError("radar DATA COM port not found; plug in the "
                               "IWR1843 or pass data_port explicitly")
        if send_cfg and cfg_path:
            if not cfg_port:
                raise RuntimeError("CONFIG COM port not found; pass cfg_port "
                                   "or send_cfg=False")
            cs = serial.Serial(cfg_port, 115200, timeout=0.3)
            try:
                send_config(cs, cfg_path)
            finally:
                cs.close()          # a rejected command must not leak the port
        src = cls(data_port, window_s)
        src.start()
        return src

    def run(self):
        while self.running:
            try:
                data = self.ser.read(4096)
            except OSError:            # SerialException is an OSError
                # USB unplug / port closed mid-read: die loudly, not silently —
                # alive() goes False once the window empties
                if self.running:
                    print("Radar UART read failed - radar thread stopped "
                          "(unplugged?)")
                self.running = False
                break
            for fr in self.reader.feed(data):
                arrival = time.monotonic()       # monotonic, never wall clock
                fno = fr["frame"]
                # (re)anchor on the first frame or a frame_no reset (sensorStart)
                if self._anchor is None or \
                        (self._last_frame is not None and fno < self._last_frame):
                    self._anchor = arrival
                    self._first_frame = fno
                    self._last_frame = None
                # count dropped frames from gaps in the radar's own frame index
                if self._last_frame is not None and fno > self._last_frame + 1:
                    self.frames_dropped += fno - self._last_frame - 1
                self._last_frame = fno
                # sample time from the frame index, so UART jitter is not baked
                # into the timestamp (the auditor's #1 failure)
                sample_t = sensor_time_from_frame(
                    fno, self.frame_period_s, self._anchor, self._first_frame)
                snr = fr["snr"] or []
                noise = fr.get("noise") or []
                pts = [p + (snr[i] if i < len(snr) else None,
                            noise[i] if i < len(noise) else None)
                       for i, p in enumerate(fr["points"])]
                with self.lock:
                    self.frames_rx += 1
                    self.window.append((sample_t, pts))
                    while (self.window
                           and sample_t - self.window[0][0] > self.window_s):
                        self.window.popleft()

    def latest_points(self):
        """All points inside the sliding window, newest last."""
        cutoff = time.monotonic() - self.window_s
        with self.lock:
            return [p for ts, pl in self.window if ts >= cutoff for p in pl]

    def clusters(self, include_points=False):
        """classify_frame() over the current window -> labelled clusters."""
        return classify_frame(self.latest_points(),
                              include_points=include_points)

    def alive(self):
        """True when frames arrived recently (radar link is healthy)."""
        with self.lock:
            return bool(self.window) and \
                time.monotonic() - self.window[-1][0] < 1.0

    def health(self):
        """Link telemetry: frames received, frames dropped (from frame_no
        gaps), and current window occupancy. Drops are reported, not hidden."""
        with self.lock:
            return {"frames_rx": self.frames_rx,
                    "frames_dropped": self.frames_dropped,
                    "window_frames": len(self.window)}

    def stop(self):
        self.running = False
        try:
            self.ser.close()
        except Exception:
            pass

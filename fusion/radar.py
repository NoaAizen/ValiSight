"""IWR1843 radar ingest for the fusion package.

Wraps the tested iwr1843_uart.RadarReader over pyserial in a background
thread, keeps a sliding window of points (sparse static returns need
aggregation before clustering), and reuses radar_classify_n6.classify_frame
for the coarse radar class. The radar works in any light and through
smoke/fog — it is the sensor that is ALWAYS on.
"""
import collections
import threading
import time

from iwr1843_uart import RadarReader, send_config, find_com_ports
from radar_classify_n6 import classify_frame

WINDOW_S = 0.5                  # sliding aggregation window


class RadarSource(threading.Thread):
    """Background DATA-UART reader.

    latest_points() -> list of (x, y, z, doppler[, snr, noise]) from the last
    WINDOW_S seconds; clusters() -> classify_frame() over that window.
    """

    def __init__(self, data_port, window_s=WINDOW_S):
        super().__init__(daemon=True)
        import serial
        self.ser = serial.Serial(data_port, 921600, timeout=0.05)
        self.reader = RadarReader()
        self.window_s = window_s
        self.lock = threading.Lock()
        self.window = collections.deque()       # (timestamp, points)
        self.frames_rx = 0
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
            send_config(cs, cfg_path)
            cs.close()
        src = cls(data_port, window_s)
        src.start()
        return src

    def run(self):
        while self.running:
            data = self.ser.read(4096)
            for fr in self.reader.feed(data):
                now = time.time()
                snr = fr["snr"] or []
                noise = fr.get("noise") or []
                pts = [p + (snr[i] if i < len(snr) else None,
                            noise[i] if i < len(noise) else None)
                       for i, p in enumerate(fr["points"])]
                with self.lock:
                    self.frames_rx += 1
                    self.window.append((now, pts))
                    while (self.window
                           and now - self.window[0][0] > self.window_s):
                        self.window.popleft()

    def latest_points(self):
        """All points inside the sliding window, newest last."""
        cutoff = time.time() - self.window_s
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
                time.time() - self.window[-1][0] < 1.0

    def stop(self):
        self.running = False
        try:
            self.ser.close()
        except Exception:
            pass

"""סעיף 1 — הרדאר האמיתי (IWR1843, UART) מאחורי RadarFeed של יעל.

RadarFeed (board/mmwave/radar_feed.py) כבר מממש 1.1/1.2/1.3/3.1/3.2, אבל עד
היום ניזון מסימולציה. כאן: LiveRadar פותח את יציאת ה-DATA (921600), מריץ את
mmwave_parser (FrameSync + parse_frame) בחוט רקע, חותם כל פריים בשעון
המשותף ברגע שהוא נסגר, ודוחף ל-ParsedFrameSource. אם הרדאר שותק (הופעל
עכשיו) — דוחף את ה-cfg דרך יציאת ה-CLI (115200), פקודה-פקודה עם המתנה
ל-Done (מתכון של iwr1843_uart.send_config, שאומת בשטח).

חשוב ליעל: ב-cfg שלנו `clutterRemoval -1 0` — הרדאר עצמו לא מסיר סטטיים,
אז הקירות באמת מגיעים. עם clutterRemoval=1 הם היו נעלמים עוד לפני התוכנה.
"""
import glob
import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_MMW = os.path.join(os.path.dirname(_HERE), "board", "mmwave")
if _MMW not in sys.path:
    sys.path.insert(0, _MMW)
import mmwave_parser as mp                       # noqa: E402
from radar_feed import RadarFeed, ParsedFrameSource  # noqa: E402
from .clock import shared_clock_ms               # noqa: E402

DEFAULT_CFG = os.path.join(_MMW, "radar_10hz.cfg")


def find_port(tail):
    hits = sorted(glob.glob("/dev/serial/by-id/*XDS110*" + tail))
    return os.path.realpath(hits[0]) if hits else None


def send_config(cli_serial, cfg_path, timeout=2.0, retries=2):
    """שולח cfg שורה-שורה ומחכה ל-Done לכל פקודה. מחזיר [(פקודה, תשובה)]."""
    def send_line(line):
        cli_serial.reset_input_buffer()
        cli_serial.write((line + "\n").encode())
        t0, resp = time.time(), ""
        while time.time() - t0 < timeout:
            n = cli_serial.in_waiting
            if n:
                resp += cli_serial.read(n).decode("ascii", "ignore")
                if "Done" in resp or "Error" in resp or "Ignored" in resp:
                    break
            else:
                time.sleep(0.01)
        return resp.strip()
    log = []
    for line in open(cfg_path):
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        resp = send_line(line)
        for _ in range(retries):
            if "Done" in resp or "Ignored" in resp:
                break
            resp = send_line(line)
        log.append((line, resp))
    return log


class LiveRadar:
    """חוט רקע: UART -> פריימים מפוענחים -> ParsedFrameSource -> RadarFeed."""

    def __init__(self, data_port=None, cli_port=None, cfg=DEFAULT_CFG,
                 clock_ms=shared_clock_ms, lever_arm_m=None, autostart=True):
        self.data_port = data_port or find_port("if03")
        self.cli_port = cli_port or find_port("if00")
        self.cfg = cfg
        self.clock_ms = clock_ms
        self.source = ParsedFrameSource()
        self.feed = RadarFeed(source=self.source, clock_ms=clock_ms,
                              lever_arm_m=lever_arm_m)
        self.status = "init"
        self.frames = 0
        self.resyncs = 0
        self.frames_lost = 0            # frame_number jumps seen ON THE WIRE (not "unread by the consumer")
        self._lost_since_poll = 0
        self._last_frame_no = None
        self.config_log = None
        self._stop = threading.Event()
        self._thread = None
        if autostart:
            self.start()

    def start(self):
        if self._thread:
            return self
        self._thread = threading.Thread(target=self._run, daemon=True, name="LiveRadar")
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def _run(self):
        import serial
        if not self.data_port:
            self.status = "no DATA port (XDS110 if03)"
            return
        try:
            ser = serial.Serial(self.data_port, 921600, timeout=0.02)
        except Exception as e:                    # noqa: BLE001
            self.status = "open failed: %s" % e
            return
        ser.reset_input_buffer()
        fs = mp.FrameSync()
        t0 = time.time()
        configured = False
        self.status = "listening"
        while not self._stop.is_set():
            n = ser.in_waiting
            chunk = ser.read(n if n else 1)
            if chunk:
                for fr in fs.feed(chunk):
                    ts = self.clock_ms()          # arrival = frame closed on the wire
                    try:
                        parsed = mp.parse_frame(fr)
                    except ValueError:
                        continue
                    fn = parsed["header"]["frame_number"]
                    if self._last_frame_no is not None and fn != self._last_frame_no + 1:
                        lost = (fn - self._last_frame_no - 1) if fn > self._last_frame_no else 1
                        self.frames_lost += lost
                        self._lost_since_poll += lost
                    self._last_frame_no = fn
                    self.source.push(parsed, ts)
                    self.frames += 1
                    self.status = "ok"
            self.resyncs = fs.resync_count
            if not self.frames and not configured and time.time() - t0 > 4.0:
                configured = True
                self._configure(serial)
        ser.close()

    def _configure(self, serial):
        if not self.cli_port or not os.path.exists(self.cfg):
            self.status = "silent, no CLI port / cfg"
            return
        self.status = "pushing cfg"
        try:
            with serial.Serial(self.cli_port, 115200, timeout=1) as cs:
                self.config_log = send_config(cs, self.cfg)
            bad = [c for c, r in self.config_log if "Done" not in r and "Ignored" not in r]
            self.status = "cfg sent" if not bad else "cfg errors: %s" % bad[:2]
        except Exception as e:                    # noqa: BLE001
            self.status = "cfg failed: %s" % str(e)[:60]

    # --- Yael's calls, delegated ------------------------------------------
    def radar_detections_all(self):
        return self.feed.radar_detections_all()

    def radar_ego_velocity(self):
        return self.feed.radar_ego_velocity()

    def sensor_health_flags(self):
        f = self.feed.sensor_health_flags()
        # RadarFeed's frame_gap compares consecutive *polls*; a slow consumer sees a
        # "gap" that is only its own polling. Report wire loss instead.
        f["frame_gap"] = self._lost_since_poll > 0
        f["frames_lost_total"] = self.frames_lost
        self._lost_since_poll = 0
        f["link"] = self.status
        f["frames_total"] = self.frames
        f["uart_resyncs"] = self.resyncs
        return f

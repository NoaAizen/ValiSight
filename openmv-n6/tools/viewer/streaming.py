"""Board stream protocol, framing and serial reader supervision."""
from pathlib import Path
import sys
import threading
import time

import serial

import capture
from .constants import OUT_W, OUT_H, TH_W, TH_H, HEAP_FLOOR, LEPTON_SAFE_GAP_MS

# Sent once. The raw REPL keeps globals between submissions, so the sensors are
# brought up a single time and the streaming batches below reuse them.
SETUP_CODE = capture._BRINGUP + (
    Path(__file__).resolve().parents[1] / "board" / "templates" / "live_stream.py.tmpl"
).read_text(encoding="utf-8")

# Sent repeatedly. Deliberately bounded: an unbounded loop on the board keeps
# writing into a port nobody reads if this host dies, the CDC RX backs up, and
# the board wedges hard enough to need a physical replug. A batch self-terminates.
BATCH_CODE = "stream(%d, %d)\n"

# How often to ask the board what its heap looks like while there is ample room.
# gc.mem_free() walks the complete 25MB heap and blocks the MicroPython/TinyUSB
# scheduler for 227ms.  At every 10th batch that pause landed every ~23s and the
# live hardware lost 512-byte CDC packets at the same cadence.  With five frames
# per batch, every 400th batch is ~3.8 minutes: still almost ten observations
# inside the deliberately huge
# 36-minute 4MB safety margin.  _want_heap() tightens to every third/every batch
# as that margin closes, where memory safety correctly wins over frame cadence.
HEAP_EVERY = 400


# ---------------------------------------------------------------- board stream


def _board_error(raw):
    """A real exception from the board, or a payload byte that happens to be 0x04?

    The raw REPL ends stdout with \\x04, so a line starting with one is how a
    traceback announces itself. But 0x04 occurs constantly inside JPEG and
    thermal data, and the moment framing slips, _line() starts handing back
    payload. Taking that at face value invents a board fault that never
    happened - and sends you debugging firmware that is working correctly.
    """
    if b"Traceback" in raw:
        return True
    if not raw.startswith(b"\x04"):
        return False
    body = raw[1:].strip()
    return bool(body) and all(c == 9 or 32 <= c < 127 for c in body)


class _PlannedRestart(Exception):
    """Not a failure: the supervisor is standing the board back up on purpose."""


class _DroppedBatch(Exception):
    """The board ended a batch while a promised payload was still incomplete.

    This is intentionally distinct from a dead serial link.  ``stream()`` gives
    up after three CDC no-progress timeouts to keep the Lepton inside its tested
    service-gap envelope, then returns normally to the raw REPL.  Restarting the
    cameras for that clean return turns one lost frame into a 20 second outage.
    The prompt is proof that the interpreter and both sensors are still alive,
    so the reader can discard this batch and submit the next one immediately.
    """

    def __init__(self, kind, received, expected):
        self.kind = kind
        self.received = received
        self.expected = expected
        super().__init__("%s short read %d/%d" % (kind, received, expected))


# A bounded raw-REPL submission ends with the line written by stream(), followed
# by MicroPython's stdout terminator and prompt.  Newline translation differs
# between ports, so accept both forms.  Detection is restricted to an exact
# BUFFER SUFFIX while fewer than the promised bytes are present; the prompt
# cannot have legitimate payload after it, and the long suffix makes mistaking
# binary image data for control framing vanishingly unlikely.
_BATCH_FOOTERS = (b"#BATCH\r\n\x04\x04>", b"#BATCH\n\x04\x04>")


class Streamer(threading.Thread):
    """Reads framed board output as fast as the port will give it.

    Drains with in_waiting rather than a fixed read size. A fixed read blocks for
    the whole port timeout collecting bytes it may never get, which back-pressures
    the board's CDC; a blocked write there starves TinyUSB's tud_task (serviced
    from the MicroPython scheduler, not an ISR) and the board falls off the bus.
    That is the failure this project spent a long time chasing.
    """

    def __init__(self, port, pipeline, quality, state, work, batch=5, raw16=False):
        super().__init__(daemon=True)
        self.port, self.pipe, self.quality, self.state = port, pipeline, quality, state
        # Ask the board for the Lepton's own 16-bit words instead of the 8-bit
        # plane it derives from them. See --raw16, and lepton_copy_raw() in the
        # firmware for what is being kept.
        self.raw16 = raw16
        # Range policy for the board's Lepton. None = auto-range (the default,
        # a percentile clip off one early frame); a (tmin, tmax) pair pins the
        # window instead. Pinning exists because auto-range samples ONCE, and
        # the sensor's output drifts for minutes after bring-up: measured here
        # 2026-08-16, a window chosen at start-up had 56.6% of the frame pinned
        # at its floor immediately and 100.0% pinned four minutes later, i.e.
        # the whole scene had fallen out of the bottom of it. A pinned wide
        # window is how you SEE that drift, and how a calibration session gets
        # a window that is still right at the end of it.
        self.fixed_range = None
        self.work = work
        self.batch = batch
        self.buf = bytearray()
        self.stop = threading.Event()
        # Failure forensics. A stall reports as a bare timeout, and the counters
        # in state are not enough to tell the two causes apart: the board going
        # quiet mid-write looks identical to this host losing count of the #F
        # headers and waiting for frames the board already finished sending.
        # What separates them is the residual buffer and what was owed at the
        # time, so keep both current.
        self.last_line, self.last_line_t = b"", 0.0
        self.headers, self.pending = 0, 0

    def _fill(self, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            n = self.s.in_waiting
            chunk = self.s.read(n if n else 1)
            if chunk:
                self.buf += chunk
                return True
        return False

    def _line(self, timeout=10.0):
        """One framed line. The timeout is a parameter because the two callers
        want very different limits: streaming lines arrive every ~115ms, while
        the one-off bring-up takes ~10s to reach #READY. The raw REPL acks a
        submission with OK *before* running it, so the ack tells you nothing
        about how long the code will take."""
        while True:
            i = self.buf.find(b"\n")
            if i >= 0:
                out = bytes(self.buf[:i])
                del self.buf[:i + 1]
                self.last_line, self.last_line_t = out, time.time()
                return out
            if not self._fill(timeout):
                raise TimeoutError("no line from board")

    def _exact(self, n, kind="payload"):
        while len(self.buf) < n:
            for footer in _BATCH_FOOTERS:
                if self.buf.endswith(footer):
                    received = len(self.buf) - len(footer)
                    # Keep the raw-REPL prompt for the normal pending==0 path.
                    # It will consume it and submit the next batch immediately.
                    del self.buf[:received + len(footer) - 3]
                    raise _DroppedBatch(kind, received, n)
            if not self._fill():
                raise TimeoutError("short read %d/%d" % (len(self.buf), n))
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out

    @staticmethod
    def _dump(b, n=32, tail=96):
        """Head of the residual buffer as hex plus repr, then its tail as repr.

        Both encodings for the head, because what is left in there is either text
        or binary and you do not know which in advance: b'\\x04\\x04>' is legible
        in repr and noise in hex, the tail of a half-read frame payload is the
        other way round.

        The tail is here because this function once threw away the diagnosis. A
        board exception arrived, the port died mid-traceback, and the 174 bytes
        still sitting in the buffer held the one line worth having - the exception
        type and its message, which a traceback prints LAST. Dumping only the head
        printed `File "<stdin>", line 1, in <module>`, the outer frame, which
        names nothing at all. The answer was already on this host and got
        truncated away, so the tail gets the generous allowance: a MicroPython
        exception line runs to ~70 characters before the prompt bytes.
        """
        head = bytes(b[:n])
        out = "%s %r" % (" ".join("%02x" % c for c in head), head)
        rest = b[n:]
        if rest:
            end = bytes(rest[-tail:])
            skipped = len(rest) - len(end)
            out += " ...%s%r" % ("[%d more]" % skipped if skipped else "", end)
        return out

    def run(self):
        """Supervisor. Streaming is a session, not a one-shot, and every failure
        mode found so far is cured by standing the bring-up back up:

          - a wedged Lepton needs a fresh csi.CSI() object, which only a new
            SETUP_CODE submission creates. Waiting, framesize() and a full soft
            re-init were all measured against a wedged part and all raised.
          - a lost stream, a resync storm, a board that fell off USB: same cure.
          - the planned heap restart above: same cure, taken early and on purpose.

        Previously this recorded the exception and let the thread end, so the
        first hiccup ended the session silently and the browser kept showing the
        last frame forever.
        """
        backoff = 1.0
        while not self.stop.is_set():
            planned = False
            try:
                self._run()
                return                          # asked to stop, cleanly
            except _PlannedRestart as e:
                planned = True
                self.state["last_restart"] = str(e)
            except Exception as e:
                self._record_failure(e)
                self.state["last_restart"] = type(e).__name__
            finally:
                # A restart outlasts STALE_S by a wide margin - ~3s draining the
                # port plus ~10s of bring-up - so without this the health panel
                # would call the stream dead every time the defence works. The
                # panel is only worth having if it distinguishes "recovering" from
                # "broken".
                self.state["restarting_since"] = time.time()
                self.release()

            if self.stop.is_set():
                return
            self.state["restarts"] = self.state.get("restarts", 0) + 1
            # A planned restart is not a fault and must not be rate-limited into
            # a stutter; an unplanned one backs off so a genuinely dead board is
            # not hammered at full speed.
            if planned:
                backoff = 1.0
            else:
                time.sleep(backoff)
                backoff = min(backoff * 2, 15.0)
            self._reset_for_restart()

    def _reset_for_restart(self):
        """Everything that must not survive into the next session."""
        self.buf = bytearray()
        self.pending, self.headers = 0, 0
        self.last_line, self.last_line_t = b"", 0.0
        self.s = None
        self.state["ready"] = False
        self.state.pop("heap_free", None)
        # SETUP_CODE recreates the board clock and resets _t_prev to zero.  Its
        # first delta is therefore board uptime, not a thermal service gap.  Do
        # not let rolling values from the previous clock survive and make that
        # first sample look like a multi-hour Lepton starvation event.
        for key in ("dt_window", "skew_window", "torn_window", "starved",
                    "last_starve_ms", "stalls", "last_stall_t",
                    "board_session_frames"):
            self.state.pop(key, None)

    def _record_failure(self, e):
        # The counters first, then the evidence. pending is the one that decides:
        # pending > 0 with the buffer holding the raw-REPL end marker means the
        # batch finished and this host is owed frames that were already sent - a
        # counting bug here, not a board fault. A partial line with pending > 0
        # means the board stopped mid-write.
        since = ("%.1fs" % (time.time() - self.last_line_t)
                 if self.last_line_t else "never")
        self.state["error"] = (
            "%s: %s (frames=%d, headers=%d, pending=%d, batches=%d, buf=%d)"
            " buf[%s] last[%r] +%s" % (
                type(e).__name__, e, self.state.get("frames", 0),
                self.headers, self.pending, self.state.get("batches", 0),
                len(self.buf), self._dump(self.buf), self.last_line[:64],
                since))

    def _attention(self, settle=8.0):
        """Take control of a board that may be in the middle of a batch.

        The old sequence was two ctrl-Cs and a 0.3s sleep, which assumes the board
        is sitting at a prompt. It very often is not: a host killed mid-session
        (a timeout, a ctrl-C, a crash) never runs release(), so the board is left
        streaming 26KB frames into a port nobody drains. Its write blocks, it
        cannot reach the point where it would notice the interrupt, and the next
        session then waits 30s for a prompt that cannot come while the buffer
        fills with a previous run's payload. That is a restart failing to restart,
        which is the one thing a supervisor may not do.

        So: the same read-first-then-interrupt dance release() already documents.
        Drain whatever is in flight so the board's write can complete, keep
        interrupting, and only enter the raw REPL once the port has gone quiet.

        A ctrl-D soft reboot was tried here as well, to get a genuinely fresh
        interpreter rather than one still holding the previous session's CSI
        objects. It desynchronised the raw-REPL handshake - the next submission
        came back as `NameError: name 'c' isn't defined`, a fragment of its own
        source - so it is deliberately not done. The heap side of that problem is
        handled instead by the gc.collect() at the top of capture._BRINGUP, which
        is safe there because no CSI object exists yet.
        """
        deadline = time.time() + settle
        quiet = 0
        while time.time() < deadline:
            n = self.s.in_waiting
            if n:
                self.s.read(n)
                quiet = 0
            else:
                quiet += 1
                if quiet >= 3:          # ~0.3s with nothing in flight
                    break
            self.s.write(b"\r\x03\x03")
            time.sleep(0.1)

        self.s.reset_input_buffer()
        self.buf = bytearray()
        self.s.write(b"\x01")           # raw REPL
        time.sleep(0.3)
        self.s.read_all()

    def release(self):
        """Read first, then interrupt. The board can only act on a ctrl-C once its
        own pending writes have somewhere to go."""
        s = getattr(self, "s", None)
        if s is None:
            return
        try:
            deadline = time.time() + 3
            while time.time() < deadline:
                nn = s.in_waiting
                s.read(nn if nn else 1)
                try:
                    s.write(b"\x03")
                except Exception:
                    pass
            s.write(b"\x02")
        except Exception:
            pass
        finally:
            s.close()

    def _traceback(self, first):
        """The whole traceback, not just the line that tripped the check.

        The one line worth having is the last one - the exception type and its
        message - and it arrives several lines after the marker. Reporting only
        the first names a file and says nothing about what went wrong, which is
        how a board fault reads as a mystery.

        Reading it must never be allowed to fail the report. Whatever stalls the
        interpreter enough to raise on the board is also what starves TinyUSB's
        tud_task, so the CDC very often dies in the middle of these very bytes -
        that is the common case here, not an edge case. Letting the read error
        propagate substitutes it for the board fault, and the session is then
        reported as `SerialException: device disconnected or multiple access on
        port?`, which sends you to check the cable while the real exception is
        sitting unread in the buffer. Keep whatever arrived and mark it partial.
        """
        deadline = time.time() + 3
        cut = ""
        while b"\x04\x04>" not in self.buf and time.time() < deadline:
            try:
                if not self._fill(0.5):
                    cut = "<no more output>"
                    break
            except Exception as e:
                cut = "<port died mid-traceback: %s: %s>" % (type(e).__name__, e)
                break
        i = self.buf.find(b"\x04\x04>")
        end = (i + 3) if i >= 0 else len(self.buf)
        rest = bytes(self.buf[:i if i >= 0 else len(self.buf)])
        del self.buf[:end]
        out = [first.lstrip("\x04")]
        out += [ln.strip() for ln in rest.decode("utf-8", "replace").splitlines()]
        if cut and i < 0:
            out.append(cut)
        return " | ".join(ln for ln in out if ln)

    def _resync(self, why):
        """Abandon this batch; the loop will start a clean one.

        Reading on past a line that is not a header is what turns one lost chunk
        into a dead stream. Bytes go missing on the wire, _exact() over-reads
        into the next frame, and from then on _line() finds newlines inside
        payloads forever - pending never returns to 0, so no further batch is
        ever submitted and a healthy board looks like it went quiet. A batch
        self-terminates, so its prompt is always still coming; drop the count
        and let the top of the loop wait for it.
        """
        self.state["resyncs"] = self.state.get("resyncs", 0) + 1
        self.state["last_resync"] = why
        self.state["last_resync_t"] = time.time()
        self.pending = 0

    def _range(self, lo, hi):
        """The board picks its own range by auto-ranging the scene, and it is the
        only thing that turns an 8-bit code back into a temperature. Push it into
        the pipeline the moment the board reports it, or every reading is scaled
        by the ratio of the assumed range to the real one - wrong by tens of
        degrees while looking entirely reasonable."""
        self.state["range"] = (lo, hi)
        self.pipe.set_range(lo, hi)

    def _await(self, token, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            i = self.buf.find(token)
            if i >= 0:
                del self.buf[:i + len(token)]
                return
            self._fill(0.5)
        raise TimeoutError("board never sent %r" % token)

    def _submit(self, code, timeout=45.0):
        """Hand one unit of work to the raw REPL and wait for its ack.

        Generous by default: the one-off bring-up runs the Lepton's 2s VoSPI
        settle plus a histogram pass in interpreted Python and measured 10.3s
        end to end. A 10s limit sat just under that and looked exactly like a
        dead board.
        """
        self.s.write(code.encode() + b"\x04")
        self._await(b"OK", timeout)

    def _want_heap(self, batch_no):
        """Should this batch carry a heap reading?

        The reading costs 227ms - gc.mem_free() walks the whole 25MB heap - so it
        cannot be taken every batch. But a fixed rate is the wrong shape for what
        it is watching: the danger is the automatic collector firing on an
        exhausted heap, which wedges the Lepton permanently, and how urgent that
        is depends entirely on how much room is left.

        So the rate follows the margin. Far from the floor, every 400th batch
        (~3.8min) is plenty against a drain measured in hours. Close to it, the
        assumption that the drain rate is the one that was measured is exactly
        what should not be relied on - a leak this code does not know about, or a
        scene that makes the loop allocate more, would be invisible until the
        collector had already fired. Near the floor the reading is worth its
        227ms every batch.
        """
        free = self.state.get("heap_free")
        if free is None:
            return True                      # first batch of a session: establish it
        if free < HEAP_FLOOR * 2:
            return True                      # inside 4MB of the trigger: watch closely
        if free < HEAP_FLOOR * 4:
            return batch_no % 3 == 0
        return batch_no % HEAP_EVERY == 0

    def _run(self):
        # TIOCEXCL is the serial equivalent of owning this stream. A second
        # opener toggling DTR makes TinyUSB discard queued 512-byte packets; it
        # must fail to open instead of corrupting a live frame silently.
        self.s = serial.Serial(self.port, 115200, timeout=0.2,
                               write_timeout=10, exclusive=True)
        tmin, tmax = self.fixed_range if self.fixed_range else (-10, 140)
        setup = (SETUP_CODE
                 .replace("__TMIN__", str(tmin)).replace("__TMAX__", str(tmax))
                 .replace("__AUTORANGE__", str(self.fixed_range is None))
                 .replace("__RAW__", str(self.raw16))
                 .replace("__Q__", str(self.quality)))
        self._attention()
        self._submit(setup)

        # Drain the bring-up before batching starts. pending begins at 0, so
        # without this the loop would wait for the raw-REPL prompt while the
        # sensors are still coming up and time out on a board that is fine.
        while True:
            line = self._line(timeout=60.0).decode("utf-8", "replace").strip()
            if line.startswith("\x04") or "Traceback" in line:
                raise RuntimeError("board: " + line.lstrip("\x04"))
            if line.startswith("#READY"):
                p = line.split()
                self._range(int(p[5]), int(p[6]))
                self.state["ready"] = True
                self.state["board_session_frames"] = 0
                # Frames are flowing again, so the error that got us here is
                # history. Leaving it set would pin the health panel red for the
                # rest of the session and train the user to ignore it.
                self.state.pop("error", None)
                self.state.pop("restarting_since", None)
                break

        self.pending = 0
        started = False
        t_prev, n = time.time(), 0      # n counts frames, for fps - do not reuse
        batch_no = 0
        while not self.stop.is_set():
            if self.pending == 0:
                # every submission ends with \x04\x04> - consume the prompt
                # before handing over the next one. That prompt is also the only
                # dependable sign that a batch finished: #BATCH is written just
                # ahead of it, so _await eats the line before the loop below can
                # ever see it - count the prompt, not the line. Match all three
                # bytes, not a bare '>': after a resync the buffer still holds
                # payload, and 0x3e is a perfectly ordinary byte inside a JPEG.
                self._await(b"\x04\x04>", 30.0)
                if started:
                    self.state["batches"] = self.state.get("batches", 0) + 1
                self._submit(
                    BATCH_CODE % (self.batch, 1 if self._want_heap(batch_no) else 0),
                    timeout=20.0)
                batch_no += 1
                started = True
                self.pending = self.batch
            raw = self._line()
            line = raw.decode("utf-8", "replace").strip()
            if _board_error(raw):
                raise RuntimeError("board: " + self._traceback(line))
            if line.startswith("#HEAP"):
                free = int(line.split()[1])
                self.state["heap_free"] = free
                if free < HEAP_FLOOR:
                    raise _PlannedRestart("heap down to %.1fMB" % (free / (1 << 20)))
                continue
            if line.startswith("#BATCH"):
                self.pending = 0
                continue
            if line.startswith("#READY"):
                p = line.split()
                self._range(int(p[5]), int(p[6]))
                self.state["ready"] = True
                self.pending = 0
                continue
            if not line.startswith("#F "):
                self._resync("not a header: %r" % raw[:24])
                continue

            f = line.split()
            jlen, tlen, torn = (int(v) for v in f[1:4])
            # The two board clocks are optional in the parse, not because any
            # firmware omits them - the host submits the code that writes them -
            # but because a resync leaves payload in the buffer and a line that
            # happens to start "#F " must not take the whole stream down on an
            # index error.
            dt_ms = int(f[4]) if len(f) > 4 else None
            skew_ms = int(f[5]) if len(f) > 5 else None
            stalled = int(f[6]) if len(f) > 6 else None
            # Bytes can go missing after the board has already counted them as
            # sent - a DTR toggle from anything else opening the port makes the
            # firmware discard what is queued, and no board-side check can see
            # it. So the lengths here are not trustworthy just because the board
            # meant well. A negative jlen would be worse than a wrong one:
            # _exact's guard is vacuous for it and del buf[:-n] throws the
            # buffer away.
            # Two legal thermal sizes, and the header is what distinguishes
            # them: one byte per pixel is the board's AGC'd 8-bit plane, two is
            # the 16-bit radiometric one. Nothing else has to be negotiated -
            # the length already says which arrived.
            if not (0 < jlen <= OUT_W * OUT_H
                    and tlen in (TH_W * TH_H, 2 * TH_W * TH_H)):
                self._resync("implausible header %r" % line[:40])
                continue
            self.headers += 1
            self.pending -= 1
            try:
                jpg = self._exact(jlen, "jpeg")
                thermal = self._exact(tlen, "thermal")
            except _DroppedBatch as e:
                self.state["batch_drops"] = self.state.get("batch_drops", 0) + 1
                self.state["last_batch_drop_t"] = time.time()
                self.state["link_recovering_since"] = time.time()
                self.state["link_recovering_reason"] = str(e)
                self._resync("board ended batch during %s" % e)
                print("batch dropped (continuing without sensor restart): %s" % e,
                      file=sys.stderr)
                continue

            # Hand off and go straight back to the port. Nothing that decodes,
            # fuses, composes or encodes belongs on this thread: every
            # millisecond spent here is a millisecond the CDC is not being
            # drained, and the board's out.write() discards the tail of a frame
            # it has already announced once no progress is made for 500ms.
            # Measured inline cost was 18ms fused / 28ms in the edges view
            # against a 114ms frame, so this is headroom rather than a rescue -
            # but the detector is 68-99ms and would not have fitted at all.
            self.work.put((jpg, thermal))

            n += 1
            now = time.time()
            self.state["frames"] = self.state.get("frames", 0) + 1
            self.state["board_session_frames"] = (
                self.state.get("board_session_frames", 0) + 1)
            self.state["last_frame_t"] = now
            self.state.pop("link_recovering_since", None)
            self.state.pop("link_recovering_reason", None)

            # Short rolling windows rather than totals: the panel is meant to
            # report the state of the board now, and a fault that cleared ten
            # minutes ago should stop being red.
            # Tearing is read off the board's own header, so it belongs to this
            # thread. rows_rebuilt is a property of the fusion pass and is
            # recorded by the worker, which is the only thread that knows when
            # one has actually run.
            tw = self.state.setdefault("torn_window", [])
            tw.append(bool(torn))
            del tw[:-60]

            # The board's own cadence, kept separate from the host's fps. They
            # answer different questions and this project has already been
            # confused by conflating them: "8.8 fps" from arrival times says the
            # link is keeping up, and says nothing about whether the sensors are.
            #
            # The first interval of every batch is discarded. _t_prev survives
            # between submissions - the raw REPL keeps globals - so the gap it
            # measures spans the host's round trip to submit the next batch, not
            # a thermal frame period. Same on the very first frame, where
            # _t_prev is still 0 and the difference is the whole uptime.
            if dt_ms is not None and self.pending < self.batch - 1 and 0 < dt_ms < 60000:
                dw = self.state.setdefault("dt_window", [])
                dw.append(dt_ms)
                del dw[:-120]
                # An FFC parks snapshot() in the driver for 1824ms (measured
                # 2026-08-06, three in ten minutes). It is not a fault and must
                # not be averaged in with the 114ms frames, or the cadence reads
                # as chronically slow for a minute after every shutter event.
                if dt_ms > 1000:
                    self.state["last_ffc_t"] = now
                    self.state["ffcs"] = self.state.get("ffcs", 0) + 1
            if skew_ms is not None and 0 <= skew_ms < 60000:
                sw = self.state.setdefault("skew_window", [])
                sw.append(skew_ms)
                del sw[:-120]

            # Host load reaching the sensor. Counted rather than averaged: one
            # stall is 500ms the Lepton went unserviced, and the part is only
            # measured safe out to a 2500ms gap. This is the number that says
            # whether adding work on this host - a detector, a browser, anything
            # that stops draining the port - has started to cost the thermal
            # side, and it is the only such signal the board can give.
            if stalled:
                self.state["stalls"] = self.state.get("stalls", 0) + stalled
                self.state["stalls_total"] = self.state.get("stalls_total", 0) + stalled
                self.state["last_stall_t"] = now
            # frames > 1: the first interval of a session spans whatever came
            # before the viewer attached - board idle, bring-up, a previous
            # viewer's death - measured on the board clock, which survives
            # attach. That time was not this session's draining and one such
            # count would hold the panel red for the whole run (2026-08-23:
            # worst 21737302ms = six idle hours, on a link running at 8.77fps).
            if dt_ms is not None and dt_ms > LEPTON_SAFE_GAP_MS \
                    and self.state.get("board_session_frames", 0) > 1:
                self.state["starved"] = self.state.get("starved", 0) + 1
                self.state["starved_total"] = self.state.get("starved_total", 0) + 1
                self.state["last_starve_ms"] = dt_ms

            if now - t_prev >= 1.0:
                self.state["fps"] = n / (now - t_prev)
                self.state["torn"] = torn
                n, t_prev = 0, now



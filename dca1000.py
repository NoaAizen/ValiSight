"""
DCA1000EVM (FPGA raw-data capture card) Ethernet interface.

The DCA1000 sits on the IWR1843BOOST 60-pin LVDS connector and streams the
radar's raw ADC samples to the PC over UDP.  This module speaks its config
protocol (UDP :4096) and receives the raw stream (UDP :4098) — no mmWave
Studio needed.

Network setup (EEPROM factory defaults):
  PC NIC   192.168.33.30 / 255.255.255.0   (static!)
  FPGA     192.168.33.180, config port 4096, data port 4098

Radar side: the OOB demo must enable the LVDS HW session so ADC data actually
reaches the FPGA:   lvdsStreamCfg -1 0 1 0
(live_radar_camera.py --dca rewrites that line automatically.)

Config packet format (both directions, little-endian):
  u16 0xA55A | u16 cmd | u16 data_len | data | u16 0xEEAA
Response data is a u16 status (0 = success); READ_FPGA_VERSION packs the
version into that word instead.

Data packets on :4098:
  u32 seq_no (1-based) | u48 byte_count (bytes sent BEFORE this packet) | raw
byte_count lets us zero-fill drops so frame boundaries stay aligned.

Usage:
    python dca1000.py --test                    # connect + FPGA version
    python dca1000.py --capture out.bin --seconds 5   # raw stream to file
"""
import argparse
import os
import socket
import struct
import threading
import time

FPGA_IP = "192.168.33.180"
PC_IP = "192.168.33.30"
CONFIG_PORT = 4096
DATA_PORT = 4098

HEADER = 0xA55A
FOOTER = 0xEEAA

CMD_RESET_FPGA = 0x01
CMD_RESET_AR_DEV = 0x02
CMD_CONFIG_FPGA = 0x03
CMD_CONFIG_EEPROM = 0x04
CMD_RECORD_START = 0x05
CMD_RECORD_STOP = 0x06
CMD_SYSTEM_CONNECT = 0x09
CMD_SYSTEM_STATUS = 0x0A          # async error report from the FPGA
CMD_CONFIG_PACKET = 0x0B
CMD_READ_FPGA_VERSION = 0x0E

CMD_NAMES = {v: k for k, v in list(globals().items()) if k.startswith("CMD_")}

# CONFIG_FPGA payload for xWR16xx/18xx (matches mmWave Studio):
#   raw logging | 2-lane LVDS | LVDS capture | ethernet stream | 16-bit | 30s
FPGA_CONFIG_DEFAULT = bytes([0x01, 0x02, 0x01, 0x02, 0x03, 0x1E])

PACKET_SIZE = 1470                # bytes of raw data per UDP packet
PACKET_DELAY_US = 25              # inter-packet gap (25us ~= 466 Mbps burst)


class DCA1000Error(RuntimeError):
    pass


class DCA1000:
    """Config-port client: connect, configure, start/stop recording."""

    def __init__(self, fpga_ip=FPGA_IP, pc_ip=PC_IP, timeout=2.0):
        self.addr = (fpga_ip, CONFIG_PORT)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind((pc_ip, CONFIG_PORT))
        except OSError as e:
            raise DCA1000Error(
                "Cannot bind %s:%d — is the NIC configured with the static "
                "IP, or is another capture tool running? (%s)"
                % (pc_ip, CONFIG_PORT, e))
        self.sock.settimeout(timeout)

    def _cmd(self, code, data=b"", retries=2):
        pkt = struct.pack("<3H", HEADER, code, len(data)) + data + \
            struct.pack("<H", FOOTER)
        last_err = None
        for _ in range(retries + 1):
            self.sock.sendto(pkt, self.addr)
            t0 = time.time()
            while time.time() - t0 < self.sock.gettimeout():
                try:
                    resp, _ = self.sock.recvfrom(2048)
                except socket.timeout:
                    last_err = "timeout"
                    break
                if len(resp) >= 8:
                    hdr, rcmd, status = struct.unpack_from("<3H", resp)
                    if hdr == HEADER and rcmd == code:
                        return status
                    # async status / stale reply — keep waiting for our echo
            last_err = last_err or "no matching response"
        raise DCA1000Error("%s: %s (FPGA at %s:%d — check Ethernet link and "
                           "that the DCA1000 is powered)"
                           % (CMD_NAMES.get(code, hex(code)), last_err,
                              self.addr[0], self.addr[1]))

    def connect(self):
        status = self._cmd(CMD_SYSTEM_CONNECT)
        if status != 0:
            raise DCA1000Error("SYSTEM_CONNECT failed (status %d)" % status)

    def fpga_version(self):
        """Returns 'major.minor' string; also proves the FPGA is alive."""
        w = self._cmd(CMD_READ_FPGA_VERSION)
        return "%d.%d" % (w & 0x7F, (w >> 7) & 0x7F)

    def configure(self, packet_size=PACKET_SIZE, delay_us=PACKET_DELAY_US):
        st = self._cmd(CMD_CONFIG_FPGA, FPGA_CONFIG_DEFAULT)
        if st != 0:
            raise DCA1000Error("CONFIG_FPGA failed (status %d)" % st)
        delay = int(delay_us * 1000 / 8)          # FPGA clock ticks
        st = self._cmd(CMD_CONFIG_PACKET,
                       struct.pack("<3H", packet_size, delay, 0))
        if st != 0:
            raise DCA1000Error("CONFIG_PACKET failed (status %d)" % st)

    def start_record(self):
        st = self._cmd(CMD_RECORD_START)
        if st != 0:
            raise DCA1000Error("RECORD_START failed (status %d)" % st)

    def stop_record(self):
        try:
            self._cmd(CMD_RECORD_STOP)
        except DCA1000Error:
            pass                                  # stopping is best-effort

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class RawCapture(threading.Thread):
    """Data-port receiver: reassembles the UDP stream into a .bin file.

    Dropped packets are zero-filled using the 48-bit byte counter, so offsets
    in the output file always equal 'bytes since record start' and frame
    boundaries stay computable.
    """

    def __init__(self, out_path, pc_ip=PC_IP):
        super().__init__(daemon=True)
        self.out_path = out_path
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)
        self.sock.bind((pc_ip, DATA_PORT))
        self.sock.settimeout(0.5)
        self.running = True
        self.packets = self.dropped = self.bytes = self.late = 0

    def run(self):
        with open(self.out_path, "wb") as f:
            while self.running:
                try:
                    pkt, _ = self.sock.recvfrom(4096)
                except socket.timeout:
                    continue
                except OSError:                   # socket closed in stop()
                    break
                if len(pkt) <= 10:
                    continue
                count = struct.unpack_from("<IHI", pkt)  # seq u32 + count u48
                byte_count = count[1] | (count[2] << 16)
                payload = pkt[10:]
                if byte_count > self.bytes:       # gap -> zero-fill the drop
                    f.write(b"\x00" * (byte_count - self.bytes))
                    self.dropped += 1
                    self.bytes = byte_count
                elif byte_count < self.bytes:
                    # late reordered packet: its slot was already zero-filled,
                    # so write it back into place instead of losing it (UDP
                    # reorder is routine at ~470 Mbps burst). True duplicates
                    # rewrite identical bytes — harmless.
                    n = min(len(payload), self.bytes - byte_count)
                    if n > 0:
                        f.seek(byte_count)
                        f.write(payload[:n])
                        f.seek(self.bytes)
                        self.late += 1
                        self.packets += 1
                    continue
                f.write(payload)
                self.bytes += len(payload)
                self.packets += 1

    def stop(self):
        self.running = False
        self.join(timeout=2.0)
        try:
            self.sock.close()
        except OSError:
            pass

    def stats(self):
        # drop_gaps counts zero-filled GAPS, not lost packets; gaps later
        # repaired by a reordered packet are counted in late_backfilled
        return {"packets": self.packets, "drop_gaps": self.dropped,
                "late_backfilled": self.late, "bytes": self.bytes}


def frame_bytes_from_cfg(cfg_path):
    """Raw ADC bytes per frame implied by a demo .cfg (complex16 samples).

    profileCfg gives numAdcSamples, channelCfg the RX count, frameCfg the
    chirps per frame.  Needed to slice adc_raw.bin into frames offline.
    """
    n_samples = n_rx = n_chirps = None
    with open(cfg_path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "profileCfg":
                n_samples = int(t[10])
            elif t[0] == "channelCfg":
                n_rx = bin(int(t[1])).count("1")
            elif t[0] == "frameCfg":
                n_chirps = (int(t[2]) - int(t[1]) + 1) * int(t[3])
    if None in (n_samples, n_rx, n_chirps):
        return None
    return n_samples * n_rx * n_chirps * 4        # 2 bytes I + 2 bytes Q


def parse_adc(bin_path, cfg_path):
    """adc_raw.bin -> complex64 array (n_frames, chirps, rx, samples).

    TI 2-lane complex layout: int16 groups of 4 = [I0, I1, Q0, Q1] for two
    consecutive samples (readDCA1000.m 'complex, 2 lanes' case). PC-side
    helper — needs numpy.
    """
    import numpy as np
    fb = frame_bytes_from_cfg(cfg_path)
    if not fb:
        raise ValueError("Could not derive frame size from %s" % cfg_path)
    raw = np.fromfile(bin_path, dtype=np.int16)
    n_frames = raw.size * 2 // fb
    if n_frames == 0:
        raise ValueError("Capture shorter than one frame (%d bytes, frame=%d)"
                         % (raw.size * 2, fb))
    raw = raw[: n_frames * fb // 2].reshape(-1, 4)
    iq = np.empty((raw.shape[0] * 2,), dtype=np.complex64)
    iq[0::2] = raw[:, 0] + 1j * raw[:, 2]
    iq[1::2] = raw[:, 1] + 1j * raw[:, 3]

    n_samples = n_rx = n_chirps = None
    with open(cfg_path) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            if t[0] == "profileCfg":
                n_samples = int(t[10])
            elif t[0] == "channelCfg":
                n_rx = bin(int(t[1])).count("1")
            elif t[0] == "frameCfg":
                n_chirps = (int(t[2]) - int(t[1]) + 1) * int(t[3])
    return iq.reshape(n_frames, n_chirps, n_rx, n_samples)


def main():
    ap = argparse.ArgumentParser(description="DCA1000EVM FPGA capture tool")
    ap.add_argument("--fpga-ip", default=FPGA_IP)
    ap.add_argument("--pc-ip", default=PC_IP)
    ap.add_argument("--test", action="store_true",
                    help="connect + read FPGA version, then exit")
    ap.add_argument("--capture", metavar="OUT.bin",
                    help="record the raw stream to this file")
    ap.add_argument("--seconds", type=float, default=5.0,
                    help="capture duration (radar must already be streaming "
                         "with lvdsStreamCfg -1 0 1 0)")
    args = ap.parse_args()

    dca = DCA1000(args.fpga_ip, args.pc_ip)
    try:
        dca.connect()
        ver = dca.fpga_version()
        print("FPGA connected, version %s" % ver)
        if args.test or not args.capture:
            return
        dca.configure()
        cap = RawCapture(args.capture, args.pc_ip)
        cap.start()
        dca.start_record()
        print("Recording %.1fs -> %s" % (args.seconds, args.capture))
        time.sleep(args.seconds)
        dca.stop_record()
        cap.stop()
        s = cap.stats()
        print("Done: %d packets, %d drop gaps (%d repaired late), %.2f MB"
              % (s["packets"], s["drop_gaps"], s["late_backfilled"],
                 s["bytes"] / 1e6))
        if s["bytes"] == 0:
            print("No data received — is the radar running with "
                  "lvdsStreamCfg -1 0 1 0 ?")
    finally:
        dca.close()


if __name__ == "__main__":
    main()

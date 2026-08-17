/*
 * ValiSight bridge, RECEIVER side — runs on the Jetson.
 *
 *   bridge_rx <serial-device | file | ->  [-o dump_dir] [-q]
 *
 * Reads the byte stream the N6 sender emits (bridge_protocol.h), reassembles
 * records in any chunking, verifies every CRC, and keeps honest counters:
 * per-type record counts, seq gaps (records the link lost), bad-CRC records,
 * resyncs (garbage skipped), and the sender's own drop count from HELLO.
 * Nothing is lost silently: every counter is printed every STATS_EVERY_MS
 * and at exit.
 *
 * With -o DIR each thermal/RGB frame is written as DIR/<seq>_<type>.bin
 * (payload only) and IMU samples appended to DIR/imu.csv — enough for the
 * fusion / Yael to consume offline; the live hand-off is the next step.
 *
 * Structure follows mmwave_stage1_raw_uart.c: a portable core (bridge_feed)
 * with no I/O, and a POSIX shell around it. The core is what test_bridge_rx_pty.py
 * exercises. Serial is opened O_RDONLY: this program can never transmit.
 */
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>
#include <zlib.h>
#include <sys/stat.h>

#include "bridge_protocol.h"

#define READ_CHUNK      65536
#define BUF_CAP         (BRIDGE_MAX_PAYLOAD + BRIDGE_HDR_LEN + READ_CHUNK)
#define STATS_EVERY_MS  2000

/* ------------------------------------------------------------- core */
typedef struct {
    uint8_t  *buf;
    size_t    len;
    /* counters */
    uint64_t  recs[4];        /* per type: hello, thermal, rgb, imu */
    uint64_t  other;          /* unknown type */
    uint64_t  bad_crc;
    uint64_t  resyncs;
    uint64_t  seq_gaps;       /* number of gap events */
    uint64_t  seq_lost;       /* records missing inside those gaps */
    uint64_t  sender_drops;   /* last value reported by HELLO */
    uint32_t  last_seq;
    int       have_seq;
    uint64_t  bytes;
    /* unwrapped timestamp of the last record, in ticks */
    int64_t   ts_unwrapped;   /* signed: an out-of-order record must not underflow */
    uint32_t  ts_last;
    int       have_ts;
    /* sink */
    const char *dump_dir;
    FILE      *imu_csv;
} bridge_rx_t;

static void rx_init(bridge_rx_t *r, const char *dump_dir)
{
    memset(r, 0, sizeof *r);
    r->buf = malloc(BUF_CAP);
    r->dump_dir = dump_dir;
    if (dump_dir) {
        char p[512];
        mkdir(dump_dir, 0755);
        snprintf(p, sizeof p, "%s/imu.csv", dump_dir);
        r->imu_csv = fopen(p, "a");
        if (r->imu_csv && ftell(r->imu_csv) == 0)
            fprintf(r->imu_csv, "seq,ts_ticks,ts_src,ax_mg,ay_mg,az_mg,gx_mdps,gy_mdps,gz_mdps\n");
    }
}

static uint32_t rd32(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static int32_t  rd32s(const uint8_t *p) { return (int32_t)rd32(p); }

static void rx_unwrap_ts(bridge_rx_t *r, const bridge_hdr_t *h)
{
    uint64_t period = bridge_ts_period(h->ts_src);
    if (!r->have_ts) { r->ts_unwrapped = (int64_t)h->ts; r->ts_last = h->ts; r->have_ts = 1; return; }
    int64_t d = (int64_t)h->ts - (int64_t)r->ts_last;
    if (d < -(int64_t)(period / 2)) d += (int64_t)period;      /* wrapped forward */
    else if (d > (int64_t)(period / 2)) d -= (int64_t)period;  /* out-of-order across wrap */
    r->ts_unwrapped += d;
    r->ts_last = h->ts;
}

static void rx_deliver(bridge_rx_t *r, const bridge_hdr_t *h, const uint8_t *payload)
{
    if (h->type < 4) r->recs[h->type]++; else r->other++;

    if (r->have_seq) {
        uint32_t expect = r->last_seq + 1u;
        if (h->seq != expect) { r->seq_gaps++; r->seq_lost += (uint32_t)(h->seq - expect); }
    }
    r->last_seq = h->seq; r->have_seq = 1;
    rx_unwrap_ts(r, h);

    if (h->type == BRIDGE_T_HELLO && h->len >= 8) {
        r->sender_drops = rd32(payload + 4);
    } else if (h->type == BRIDGE_T_IMU && h->len >= 24 && r->imu_csv) {
        fprintf(r->imu_csv, "%u,%lld,%u,%d,%d,%d,%d,%d,%d\n", h->seq,
                (long long)r->ts_unwrapped, h->ts_src,
                rd32s(payload), rd32s(payload + 4), rd32s(payload + 8),
                rd32s(payload + 12), rd32s(payload + 16), rd32s(payload + 20));
    } else if ((h->type == BRIDGE_T_THERMAL || h->type == BRIDGE_T_RGB) && r->dump_dir) {
        char p[512];
        snprintf(p, sizeof p, "%s/%010u_%s.bin", r->dump_dir, h->seq,
                 h->type == BRIDGE_T_THERMAL ? "thermal" : "rgb");
        FILE *f = fopen(p, "wb");
        if (f) { fwrite(payload, 1, h->len, f); fclose(f); }
    }
}

/* Feed a chunk of bytes; delivers every complete, CRC-valid record. */
static void bridge_feed(bridge_rx_t *r, const uint8_t *chunk, size_t n)
{
    if (r->len + n > BUF_CAP) {                 /* cannot happen with a sane sender; never overflow */
        r->resyncs++; r->len = 0;
    }
    memcpy(r->buf + r->len, chunk, n); r->len += n; r->bytes += n;

    size_t pos = 0;
    for (;;) {
        /* find magic */
        size_t i = pos;
        while (i + BRIDGE_MAGIC_LEN <= r->len && memcmp(r->buf + i, BRIDGE_MAGIC, BRIDGE_MAGIC_LEN) != 0) i++;
        if (i + BRIDGE_MAGIC_LEN > r->len) {     /* no magic: keep a 3-byte tail */
            size_t keep = r->len >= 3 ? 3 : r->len;
            if (i > pos) r->resyncs++;
            memmove(r->buf, r->buf + r->len - keep, keep); r->len = keep;
            return;
        }
        if (i > pos) r->resyncs++;
        pos = i;
        if (r->len - pos < BRIDGE_HDR_LEN) break;
        bridge_hdr_t h; bridge_hdr_parse(r->buf + pos, &h);
        if (h.len > BRIDGE_MAX_PAYLOAD) { r->bad_crc++; pos += 1; continue; }
        if (r->len - pos < BRIDGE_HDR_LEN + h.len) break;          /* wait for the rest */
        uint32_t c = crc32(0, r->buf + pos + 4, 16);
        c = crc32(c, r->buf + pos + BRIDGE_HDR_LEN, h.len);
        if (c != h.crc32) { r->bad_crc++; pos += 1; continue; }
        rx_deliver(r, &h, r->buf + pos + BRIDGE_HDR_LEN);
        pos += BRIDGE_HDR_LEN + h.len;
    }
    memmove(r->buf, r->buf + pos, r->len - pos); r->len -= pos;
}

static void rx_stats(const bridge_rx_t *r, double elapsed_s, FILE *out)
{
    fprintf(out, "[%7.1fs] hello %llu  thermal %llu (%.1f Hz)  rgb %llu (%.1f Hz)  imu %llu (%.1f Hz)  "
                 "| gaps %llu (lost %llu)  bad_crc %llu  resyncs %llu  sender_drops %llu  | %.1f KB/s\n",
            elapsed_s,
            (unsigned long long)r->recs[0],
            (unsigned long long)r->recs[1], elapsed_s > 0 ? r->recs[1] / elapsed_s : 0.0,
            (unsigned long long)r->recs[2], elapsed_s > 0 ? r->recs[2] / elapsed_s : 0.0,
            (unsigned long long)r->recs[3], elapsed_s > 0 ? r->recs[3] / elapsed_s : 0.0,
            (unsigned long long)r->seq_gaps, (unsigned long long)r->seq_lost,
            (unsigned long long)r->bad_crc, (unsigned long long)r->resyncs,
            (unsigned long long)r->sender_drops,
            elapsed_s > 0 ? r->bytes / elapsed_s / 1024.0 : 0.0);
    fflush(out);
}

/* ------------------------------------------------------------ shell */
static volatile sig_atomic_t stop = 0;
static void on_sig(int s) { (void)s; stop = 1; }

static int64_t mono_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000; }

static int open_input(const char *path)
{
    if (strcmp(path, "-") == 0) return 0;
    int fd = open(path, O_RDONLY | O_NOCTTY);
    if (fd < 0) { perror(path); return -1; }
    struct termios tio;
    if (tcgetattr(fd, &tio) == 0) {                /* a tty: raw mode, 115200 nominal (USB CDC ignores baud) */
        cfmakeraw(&tio);
        cfsetispeed(&tio, B115200); cfsetospeed(&tio, B115200);
        tio.c_cc[VMIN] = 0; tio.c_cc[VTIME] = 1;    /* 100 ms read timeout */
        tcsetattr(fd, TCSANOW, &tio);
    }
    return fd;
}

int main(int argc, char **argv)
{
    const char *path = NULL, *dump = NULL; int quiet = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-o") && i + 1 < argc) dump = argv[++i];
        else if (!strcmp(argv[i], "-q")) quiet = 1;
        else path = argv[i];
    }
    if (!path) { fprintf(stderr, "usage: %s <device|file|-> [-o dump_dir] [-q]\n", argv[0]); return 2; }
    int fd = open_input(path);
    if (fd < 0) return 1;

    signal(SIGINT, on_sig); signal(SIGTERM, on_sig);
    bridge_rx_t rx; rx_init(&rx, dump);
    uint8_t *chunk = malloc(READ_CHUNK);
    int64_t t0 = mono_ms(), last_stats = t0;
    for (;;) {
        if (stop) break;
        ssize_t n = read(fd, chunk, READ_CHUNK);
        if (n < 0) { if (errno == EINTR) continue; perror("read"); break; }
        if (n == 0) { if (fd != 0 && isatty(fd)) { /* timeout */ } else break; }   /* EOF on file/pipe */
        else bridge_feed(&rx, chunk, (size_t)n);
        int64_t now = mono_ms();
        if (!quiet && now - last_stats >= STATS_EVERY_MS) { rx_stats(&rx, (now - t0) / 1000.0, stdout); last_stats = now; }
    }
    rx_stats(&rx, (mono_ms() - t0) / 1000.0, stdout);
    printf("FINAL recs hello=%llu thermal=%llu rgb=%llu imu=%llu other=%llu gaps=%llu lost=%llu bad_crc=%llu resyncs=%llu sender_drops=%llu\n",
           (unsigned long long)rx.recs[0], (unsigned long long)rx.recs[1], (unsigned long long)rx.recs[2],
           (unsigned long long)rx.recs[3], (unsigned long long)rx.other, (unsigned long long)rx.seq_gaps,
           (unsigned long long)rx.seq_lost, (unsigned long long)rx.bad_crc, (unsigned long long)rx.resyncs,
           (unsigned long long)rx.sender_drops);
    fflush(stdout);
    if (rx.imu_csv) fclose(rx.imu_csv);
    free(rx.buf); free(chunk);
    if (fd > 0) close(fd);
    return 0;
}

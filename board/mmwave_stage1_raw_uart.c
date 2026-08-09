/*
 * mmwave_stage1_raw_uart.c — C port of mmwave_stage1_raw_uart.py.
 * Stage 1 of the N6<->AWR1843 TLV library bring-up.
 *
 * Purpose: wiring/baud sanity ONLY. No parsing. Reads raw bytes from the radar
 * DATA UART and reports whether the mmWave magic word appears at a stable rate.
 *
 * Gate (do not proceed to Stage 2 until BOTH hold):
 *   1. Magic word 02 01 04 03 06 05 08 07 repeats at a stable rate
 *      (should match the frame rate in your radar .cfg, e.g. ~10 Hz).
 *   2. No RX overruns: "bytes/s" stays steady and no gaps/garbage between
 *      magics beyond a sane frame size.
 *
 * If NO magic appears: wiring (DATA_TX -> RX), shared ground, baud, or the
 * radar isn't streaming (sensorStart not sent). Not a code problem — stop here.
 *
 * ---------------------------------------------------------------------------
 * Physical connections (from NOA_TASK2_FINDINGS.md, board is IWR1843BOOST):
 *
 *   signal      radar EVM           listener              notes
 *   ---------   ----------------    ------------------    --------------------
 *   DATA_TX     J5 pin 9            N6 UART3 RX (P5)      921600 8N1, 3.3V
 *   SYNC_OUT    J6 pin 18           N6 input-capture pin  one pulse per frame
 *   GND         any GND pin         any GND pin           MANDATORY, first wire
 *   power       P6 barrel 5V/3A     (dedicated supply)    press NRST (SW2) once
 *                                                         after power-up
 *
 *   Power-up order: radar FIRST, wait for PGOOD (J6 pin 14) to reach 3.3V,
 *   only then connect/enable any signal toward it (pins are not failsafe).
 *   Never drive J6 pin 7 (radar RX) while the XDS110 USB is attached.
 *
 *   No level shifter — both sides are 3.3V logic.
 *
 *   The DATA UART reaches the headers and the XDS110 USB in PARALLEL (no
 *   switch), which is why this same check can run on the Jetson over
 *   /dev/ttyACM* before the N6 is wired at all: same bytes, same gate.
 *   ttyACM numbering is NOT stable across replugs on this system — the DATA
 *   port is usually the *second* XDS110 port, verify with tools/radar_scan.py.
 * ---------------------------------------------------------------------------
 *
 * The scanner core (stage1_*) is dependency-free C99 in the same spirit as
 * fusion.c: portable to an OpenMV firmware module later, with main() below
 * swapped for a MicroPython binding that pumps the UART RX FIFO into
 * stage1_feed().
 *
 * Build (Jetson / any Linux):
 *   gcc -O2 -Wall -Wextra -o mmwave_stage1_raw_uart mmwave_stage1_raw_uart.c
 *
 * Run:
 *   ./mmwave_stage1_raw_uart /dev/ttyACM1            # XDS110 DATA port
 *   ./mmwave_stage1_raw_uart /dev/ttyUSB0 921600     # explicit baud
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------ portable core */

#define STAGE1_MAGIC_LEN   8
#define STAGE1_MAX_IVALS   50      /* rolling window of inter-magic intervals */
#define STAGE1_HEXDUMP     512     /* hexdump only the first N bytes */

static const uint8_t STAGE1_MAGIC[STAGE1_MAGIC_LEN] =
    { 0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07 };

typedef struct {
    /* carry of the last magic_len-1 bytes, so a magic split across two chunks
       is still counted — same tail trick as the .py */
    uint8_t  tail[STAGE1_MAGIC_LEN - 1];
    int      tail_len;

    uint64_t total_bytes;
    uint64_t magic_count;

    /* last N inter-magic intervals in ms, ring buffer */
    uint32_t ivals[STAGE1_MAX_IVALS];
    int      ival_head, ival_len;
    int64_t  last_magic_ms;        /* -1 until the first magic */

    /* per-stats-window counters */
    uint64_t window_bytes;
    uint64_t window_magics;

    int      dumped;               /* hexdump budget used */
} stage1_t;

static void stage1_init(stage1_t *s)
{
    memset(s, 0, sizeof(*s));
    s->last_magic_ms = -1;
}

static void stage1_hexdump(const uint8_t *data, int len, int base)
{
    for (int row = 0; row < len; row += 16) {
        printf("%06x ", base + row);
        for (int i = row; i < row + 16 && i < len; i++)
            printf(" %02x", data[i]);
        printf("\n");
    }
}

static void stage1_count_magic(stage1_t *s, int64_t now_ms)
{
    s->magic_count++;
    s->window_magics++;
    if (s->last_magic_ms >= 0) {
        s->ivals[s->ival_head] = (uint32_t)(now_ms - s->last_magic_ms);
        s->ival_head = (s->ival_head + 1) % STAGE1_MAX_IVALS;
        if (s->ival_len < STAGE1_MAX_IVALS) s->ival_len++;
    }
    s->last_magic_ms = now_ms;
}

/*
 * Feed one received chunk. now_ms must come from a monotonic clock; on the N6
 * that means the free-running hardware timer, NOT time.ticks_ms() carried past
 * its 17.9-minute wrap.
 *
 * The .py joins tail+chunk and searches the join; here the seam and the body
 * are scanned separately to avoid allocating. A magic that starts inside the
 * old tail is found by the seam scan; body_start then skips the bytes of the
 * chunk that seam match consumed, so nothing is counted twice.
 */
static void stage1_feed(stage1_t *s, const uint8_t *chunk, int len, int64_t now_ms)
{
    if (len <= 0) return;

    s->total_bytes += (uint64_t)len;
    s->window_bytes += (uint64_t)len;

    if (s->dumped < STAGE1_HEXDUMP) {
        int take = STAGE1_HEXDUMP - s->dumped;
        if (take > len) take = len;
        stage1_hexdump(chunk, take, s->dumped);
        s->dumped += take;
        if (s->dumped >= STAGE1_HEXDUMP)
            printf("stage1: hexdump limit reached, switching to stats only\n");
    }

    /* seam: old tail + first magic_len-1 bytes of the chunk */
    uint8_t seam[2 * (STAGE1_MAGIC_LEN - 1)];
    int seam_chunk = len < STAGE1_MAGIC_LEN - 1 ? len : STAGE1_MAGIC_LEN - 1;
    int seam_len = s->tail_len + seam_chunk;
    int body_start = 0;

    memcpy(seam, s->tail, (size_t)s->tail_len);
    memcpy(seam + s->tail_len, chunk, (size_t)seam_chunk);

    for (int i = 0; i + STAGE1_MAGIC_LEN <= seam_len && i < s->tail_len; i++) {
        if (memcmp(seam + i, STAGE1_MAGIC, STAGE1_MAGIC_LEN)) continue;
        stage1_count_magic(s, now_ms);
        body_start = i + STAGE1_MAGIC_LEN - s->tail_len;
        i += STAGE1_MAGIC_LEN - 1;
    }

    for (int i = body_start; i + STAGE1_MAGIC_LEN <= len; i++) {
        if (memcmp(chunk + i, STAGE1_MAGIC, STAGE1_MAGIC_LEN)) continue;
        stage1_count_magic(s, now_ms);
        i += STAGE1_MAGIC_LEN - 1;
    }

    /* carry the last magic_len-1 bytes into the next call */
    if (len >= STAGE1_MAGIC_LEN - 1) {
        memcpy(s->tail, chunk + len - (STAGE1_MAGIC_LEN - 1), STAGE1_MAGIC_LEN - 1);
        s->tail_len = STAGE1_MAGIC_LEN - 1;
    } else {
        int drop = s->tail_len + len - (STAGE1_MAGIC_LEN - 1);
        if (drop < 0) drop = 0;
        memmove(s->tail, s->tail + drop, (size_t)(s->tail_len - drop));
        memcpy(s->tail + s->tail_len - drop, chunk, (size_t)len);
        s->tail_len = s->tail_len - drop + len;
    }
}

static void stage1_stats(stage1_t *s, int64_t elapsed_ms)
{
    char rate[96];

    if (s->ival_len > 0) {
        uint32_t lo = 0xFFFFFFFFu, hi = 0;
        uint64_t sum = 0;
        for (int i = 0; i < s->ival_len; i++) {
            uint32_t v = s->ivals[i];
            sum += v;
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        snprintf(rate, sizeof(rate),
                 "avg frame interval %llu ms (jitter %u ms over last %d)",
                 (unsigned long long)(sum / (uint64_t)s->ival_len),
                 hi - lo, s->ival_len);
    } else {
        snprintf(rate, sizeof(rate), "no magic yet");
    }

    printf("stage1: %llu B/s | magics total %llu (+%llu) | %s\n",
           (unsigned long long)(s->window_bytes * 1000 / (uint64_t)elapsed_ms),
           (unsigned long long)s->magic_count,
           (unsigned long long)s->window_magics, rate);
    fflush(stdout);

    s->window_bytes = 0;
    s->window_magics = 0;
}

/* --------------------------------------------------------------- POSIX shell
 *
 * Everything below replaces MicroPython's UART object for a Linux host. On the
 * N6 firmware build this file's core stays, and main() is swapped for a
 * MicroPython binding.
 */
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#define STATS_EVERY_MS 2000
#define READ_CHUNK     4096

static int64_t mono_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static speed_t baud_const(long baud)
{
    switch (baud) {
    case 115200:  return B115200;
    case 230400:  return B230400;
    case 460800:  return B460800;
    case 921600:  return B921600;
    case 1000000: return B1000000;
    default:      return 0;
    }
}

static int open_uart(const char *dev, long baud)
{
    int fd = open(dev, O_RDONLY | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) {
        fprintf(stderr, "stage1: open %s: %s\n", dev, strerror(errno));
        return -1;
    }

    struct termios tio;
    if (tcgetattr(fd, &tio) != 0) {
        fprintf(stderr, "stage1: tcgetattr: %s\n", strerror(errno));
        close(fd);
        return -1;
    }

    cfmakeraw(&tio);                                 /* raw: no line discipline */
    tio.c_cflag &= ~(CSTOPB | PARENB | CRTSCTS);     /* 8N1, no flow control */
    tio.c_cflag |= CLOCAL | CREAD;
    tio.c_cc[VMIN] = 0;                              /* non-blocking reads */
    tio.c_cc[VTIME] = 0;

    speed_t sp = baud_const(baud);
    if (sp == 0) {
        fprintf(stderr, "stage1: unsupported baud %ld\n", baud);
        close(fd);
        return -1;
    }
    cfsetispeed(&tio, sp);
    cfsetospeed(&tio, sp);

    if (tcsetattr(fd, TCSANOW, &tio) != 0) {
        fprintf(stderr, "stage1: tcsetattr: %s\n", strerror(errno));
        close(fd);
        return -1;
    }
    tcflush(fd, TCIFLUSH);                           /* start clean */
    return fd;
}

int main(int argc, char **argv)
{
    const char *dev = argc > 1 ? argv[1] : "/dev/ttyACM1";
    long baud = argc > 2 ? strtol(argv[2], NULL, 10) : 921600;

    int fd = open_uart(dev, baud);
    if (fd < 0) return 1;

    printf("stage1: listening on %s @ %ld 8N1\n", dev, baud);
    printf("stage1: expecting magic");
    for (int i = 0; i < STAGE1_MAGIC_LEN; i++) printf(" %02x", STAGE1_MAGIC[i]);
    printf("\n");
    fflush(stdout);

    stage1_t s;
    stage1_init(&s);

    uint8_t chunk[READ_CHUNK];
    int64_t last_stats = mono_ms();

    for (;;) {
        ssize_t n = read(fd, chunk, sizeof(chunk));
        int64_t now = mono_ms();

        if (n > 0) {
            stage1_feed(&s, chunk, (int)n, now);
        } else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
            fprintf(stderr, "stage1: read: %s\n", strerror(errno));
            break;               /* device vanished — the recable gotcha */
        }

        if (now - last_stats >= STATS_EVERY_MS) {
            stage1_stats(&s, now - last_stats);
            last_stats = now;
        }

        if (n <= 0) {
            struct timespec ts = { 0, 2 * 1000 * 1000 };   /* 2 ms, as the .py */
            nanosleep(&ts, NULL);
        }
    }

    close(fd);
    return 1;
}
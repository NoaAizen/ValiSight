/*
 * ValiSight bridge protocol — N6 (cameras + IMU) -> Jetson, over USB VCP.
 *
 * ONE message per record. Every record is:   header (24 B) + payload (len B)
 * Little-endian everywhere. Byte layout of the header:
 *
 *   off  size  field     meaning
 *    0    4    magic     "VSB1"  — scan for this to (re)synchronise
 *    4    1    type      BRIDGE_T_*  (what the payload is)
 *    5    1    ts_src    BRIDGE_TS_* (which N6 clock stamped `ts`)
 *    6    2    reserved  0
 *    8    4    seq       global running number, +1 per record, wraps at 2^32.
 *                        A gap = records lost between sender and receiver.
 *   12    4    ts        N6 timestamp in the units of ts_src (wraps! unwrap
 *                        on the receiver, never compare raw values)
 *   16    4    len       payload length in bytes (0..BRIDGE_MAX_PAYLOAD)
 *   20    4    crc32     zlib CRC-32 over header bytes [4,20) + payload
 *
 * Payloads (all little-endian):
 *   THERMAL : u16 w, u16 h, u8 fmt(BRIDGE_PIX_*), u8 flags (bit0 FFC in progress, bit1 FFC known), then pixels
 *   RGB     : u16 w, u16 h, u8 fmt(BRIDGE_PIX_JPEG), u8 flags (0), then bytes
 *   IMU     : i32 ax,ay,az [milli-g]  i32 gx,gy,gz [milli-deg/s]
 *   HELLO   : u32 proto_ver, u32 sender_drops (records the N6 itself gave up
 *             on because USB was not draining), u32 imu_overflow (IMU samples
 *             the on-board ring could not hold — happens when a USB stall
 *             blocks the sender), sent once at start and then every ~1 s as a
 *             heartbeat so silence is distinguishable from "nothing to send".
 *
 * A receiver that reads a bad crc drops that record and rescans for magic.
 * The same file is mirrored in bridge_protocol.py — keep them identical.
 */
#ifndef VALISIGHT_BRIDGE_PROTOCOL_H
#define VALISIGHT_BRIDGE_PROTOCOL_H

#include <stdint.h>

#define BRIDGE_PROTO_VER      1u
#define BRIDGE_MAGIC          "VSB1"
#define BRIDGE_MAGIC_LEN      4
#define BRIDGE_HDR_LEN        24
#define BRIDGE_MAX_PAYLOAD    (256u * 1024u)   /* sanity bound; frames are far smaller */

/* type */
#define BRIDGE_T_HELLO        0u
#define BRIDGE_T_THERMAL      1u
#define BRIDGE_T_RGB          2u
#define BRIDGE_T_IMU          3u

/* ts_src */
#define BRIDGE_TS_TICKS_US    0u   /* MicroPython time.ticks_us(): 1 us tick, wraps at 2^30 (~17.9 min) */
#define BRIDGE_TS_TIM2_500NS  1u   /* pyb TIM2 counter, prescaler 199 @400 MHz: 500 ns tick, wraps at 2^31 (~35.8 min) */

/* image fmt */
#define BRIDGE_PIX_GRAY8      0u
#define BRIDGE_PIX_JPEG       1u

typedef struct {
    uint8_t  type;
    uint8_t  ts_src;
    uint32_t seq;
    uint32_t ts;
    uint32_t len;
    uint32_t crc32;
} bridge_hdr_t;

/* Parse 24 header bytes (magic already verified by the caller). */
static inline void bridge_hdr_parse(const uint8_t *b, bridge_hdr_t *h)
{
    h->type   = b[4];
    h->ts_src = b[5];
    h->seq    = (uint32_t)b[8]  | ((uint32_t)b[9]  << 8) | ((uint32_t)b[10] << 16) | ((uint32_t)b[11] << 24);
    h->ts     = (uint32_t)b[12] | ((uint32_t)b[13] << 8) | ((uint32_t)b[14] << 16) | ((uint32_t)b[15] << 24);
    h->len    = (uint32_t)b[16] | ((uint32_t)b[17] << 8) | ((uint32_t)b[18] << 16) | ((uint32_t)b[19] << 24);
    h->crc32  = (uint32_t)b[20] | ((uint32_t)b[21] << 8) | ((uint32_t)b[22] << 16) | ((uint32_t)b[23] << 24);
}

/* Wrap period of each clock, in ticks, for unwrapping on the receiver. */
static inline uint64_t bridge_ts_period(uint8_t ts_src)
{
    return ts_src == BRIDGE_TS_TIM2_500NS ? (1ull << 31) : (1ull << 30);
}
/* Seconds per tick of each clock. */
static inline double bridge_ts_seconds_per_tick(uint8_t ts_src)
{
    return ts_src == BRIDGE_TS_TIM2_500NS ? 500e-9 : 1e-6;
}

#endif

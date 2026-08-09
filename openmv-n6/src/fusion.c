/*
 * See fusion.h for the pipeline overview.
 *
 * Everything here is integer. The scalar code in this file is the reference:
 * the Helium paths added later must match it bit for bit, and it is also what
 * runs on the host during algorithm work.
 */
#include "fusion.h"

#include <stdlib.h>
#include <string.h>

/*
 * Allocation. The host build uses libc; inside OpenMV firmware the buffers must
 * come from the MicroPython GC heap so they are traced through the owning object
 * rather than leaking on a soft reset.
 */
#ifndef FUSION_ALLOC
#ifdef FUSION_USE_MP_ALLOC
#include "umalloc.h"
/*
 * UMA_FAST asks for internal SRAM rather than the MicroPython GC heap, which on
 * the N6 lives in external SDRAM. That distinction dominates this pipeline: it
 * makes several full-frame passes over 640x400, so where the working set sits
 * matters more than the arithmetic. UMA_MAYBE degrades to slower memory instead
 * of failing outright, so a tight build still runs, just slower.
 */
#define FUSION_ALLOC(n)  uma_calloc((n), UMA_FAST | UMA_PERSIST | UMA_MAYBE)
#define FUSION_FREE(p)   do { if (p) { uma_free(p); } } while (0)
#else
#define FUSION_ALLOC(n)  calloc(1, (n))
#define FUSION_FREE(p)   free(p)
#endif
#endif

#define CLAMP255(v) ((v) < 0 ? 0 : ((v) > 255 ? 255 : (v)))

/* ------------------------------------------------------------------ box filter
 *
 * Separable running-sum box filter, borders handled by replication so the
 * window count is constant and no per-pixel normalisation table is needed.
 *
 * Overflow budget: the vertical pass accumulates (2r+1) horizontal sums, each
 * itself (2r+1) input samples. With squared 8-bit input (65025 max) that is
 * 65025*(2r+1)^2, which stays inside int32 for r <= 8. gf_radius is validated
 * against that in fusion_init().
 */
static void box_i32(const int32_t *src, int32_t *dst, int w, int h, int r, int32_t *tmp)
{
    const int win = 2 * r + 1;

    for (int y = 0; y < h; y++) {
        const int32_t *s = src + (size_t)y * w;
        int32_t *d = tmp + (size_t)y * w;
        int32_t acc = 0;

        for (int i = -r; i <= r; i++) {
            int xi = i < 0 ? 0 : (i >= w ? w - 1 : i);
            acc += s[xi];
        }
        for (int x = 0; x < w; x++) {
            d[x] = acc;
            int xa = x - r, xb = x + r + 1;
            xa = xa < 0 ? 0 : (xa >= w ? w - 1 : xa);
            xb = xb < 0 ? 0 : (xb >= w ? w - 1 : xb);
            acc += s[xb] - s[xa];
        }
    }

    for (int x = 0; x < w; x++) {
        int32_t acc = 0;
        for (int i = -r; i <= r; i++) {
            int yi = i < 0 ? 0 : (i >= h ? h - 1 : i);
            acc += tmp[(size_t)yi * w + x];
        }
        for (int y = 0; y < h; y++) {
            dst[(size_t)y * w + x] = acc;
            int ya = y - r, yb = y + r + 1;
            ya = ya < 0 ? 0 : (ya >= h ? h - 1 : ya);
            yb = yb < 0 ? 0 : (yb >= h ? h - 1 : yb);
            acc += tmp[(size_t)yb * w + x] - tmp[(size_t)ya * w + x];
        }
    }

    const int32_t n = (int32_t)win * win;
    const int32_t half = n / 2;
    for (size_t i = 0, e = (size_t)w * h; i < e; i++)
        dst[i] = (dst[i] + half) / n;   /* rounded mean */
}

/* Horizontal running sum of one row, borders replicated. */
static void hsum_row(const uint8_t *src, int w, int h, int y, int r, int32_t *dst)
{
    if (y < 0) y = 0;
    else if (y >= h) y = h - 1;

    const uint8_t *s = src + (size_t)y * w;
    int32_t acc = 0;

    for (int i = -r; i <= r; i++) {
        int xi = i < 0 ? 0 : (i >= w ? w - 1 : i);
        acc += s[xi];
    }
    for (int x = 0; x < w; x++) {
        dst[x] = acc;
        int xa = x - r, xb = x + r + 1;
        xa = xa < 0 ? 0 : (xa >= w ? w - 1 : xa);
        xb = xb < 0 ? 0 : (xb >= w ? w - 1 : xb);
        acc += s[xb] - s[xa];
    }
}

/*
 * Full-resolution box blur.
 *
 * Only (2r+2) rows of horizontal sums are ever live, held in a ring, so the
 * scratch is O(r*w) rather than O(w*h). At 640x400 the full-frame form would
 * have wanted 1MB of int32 - more than the fused frame itself, and far more
 * than is sane to hand a 4.2MB device.
 *
 * scratch must hold (2r+2)*w int32.
 */
void fusion_box_u8(const uint8_t *src, uint8_t *dst, int w, int h, int r, int32_t *scratch)
{
    const int win = 2 * r + 1;
    const int32_t n = (int32_t)win * win;
    const int32_t half = n / 2;

    int32_t *ring = scratch;             /* win * w */
    int32_t *col = scratch + (size_t)win * w;

    for (int i = 0; i < win; i++)
        hsum_row(src, w, h, i - r, r, ring + (size_t)(((i - r) % win + win) % win) * w);

    for (int x = 0; x < w; x++) col[x] = 0;
    for (int i = 0; i < win; i++) {
        const int32_t *row = ring + (size_t)i * w;
        for (int x = 0; x < w; x++) col[x] += row[x];
    }

    for (int y = 0; y < h; y++) {
        uint8_t *d = dst + (size_t)y * w;
        for (int x = 0; x < w; x++)
            d[x] = (uint8_t)((col[x] + half) / n);

        if (y + 1 >= h) break;

        /* Row y-r leaves the window and row y+1+r enters it. They differ by
           exactly win, so they share a ring slot - evict then refill in place. */
        int32_t *slot = ring + (size_t)(((y - r) % win + win) % win) * w;
        for (int x = 0; x < w; x++) col[x] -= slot[x];
        hsum_row(src, w, h, y + 1 + r, r, slot);
        for (int x = 0; x < w; x++) col[x] += slot[x];
    }
}

/* ------------------------------------------------------------------ decimate */

void fusion_decimate(const uint8_t *src, int sw, int sh, uint8_t *dst, int dw, int dh)
{
    const int fx = sw / dw, fy = sh / dh;
    const int n = fx * fy, half = n / 2;

    for (int y = 0; y < dh; y++) {
        for (int x = 0; x < dw; x++) {
            int32_t acc = 0;
            const uint8_t *p = src + (size_t)(y * fy) * sw + (size_t)(x * fx);
            for (int j = 0; j < fy; j++, p += sw)
                for (int i = 0; i < fx; i++)
                    acc += p[i];
            dst[(size_t)y * dw + x] = (uint8_t)((acc + half) / n);
        }
    }
}

/* ------------------------------------------------------------------ thermal prep
 *
 * Three artefacts the Lepton hands us, all cheap to remove and all visible on
 * real frames before anything else in the pipeline can be judged.
 *
 * Saturated rows: fixed rows of this unit come back pinned at 255. See the
 * deadrow_flat/deadrow_lift comment in fusion.h for why that is a sensor defect
 * rather than a link problem, and why it cannot be left in.
 *
 * Banding: the 3.5 delivers 120 rows as four 30-row VoSPI segments, and the
 * segments carry slightly different DC offsets. The seam is a constant step, not
 * a content change, so it is estimated from the rows either side of each
 * boundary and subtracted cumulatively. Anchoring to the mean offset keeps the
 * frame's overall level - and therefore its temperature mapping - unchanged.
 *
 * Bad pixels: isolated stuck-high/low elements. Replaced by the median of their
 * four neighbours, but only when they differ from it by more than a threshold,
 * so genuine small hot targets survive.
 */
/*
 * Divide rounding to nearest, for a positive divisor and either sign of
 * numerator. C99 truncates toward zero, which biases every quotient toward the
 * origin; where the quotients are then accumulated (the VoSPI seam offsets) the
 * bias compounds instead of averaging out.
 */
static int32_t rdiv(int32_t num, int32_t den)
{
    return num >= 0 ? (num + den / 2) / den : -((-num + den / 2) / den);
}

static uint8_t med4(int a, int b, int c, int d)
{
    int t;
    if (a > b) { t = a; a = b; b = t; }
    if (c > d) { t = c; c = d; d = t; }
    if (a > c) { t = a; a = c; c = t; }
    if (b > d) { t = b; b = d; d = t; }
    return (uint8_t)((b + c) / 2);
}

/*
 * Replace rows that carry no scene with a blend of the nearest live rows.
 * Operates in place on dst; row_bad (h flags) records what was rebuilt so the
 * radiometric path can refuse to quote a reconstructed pixel.
 *
 * See the deadrow_flat/deadrow_lift comment in fusion.h for what the test is and
 * why it is not a brightness threshold.
 *
 * Runs before the deband and the bad-pixel pass, both of which it would
 * otherwise poison: a dead row landing on a VoSPI seam reads as a ~200-code DC
 * step and gets subtracted from a whole 30-row segment - which is worse than the
 * stripe, because a bright line is obvious and a segment 35C too cold is not -
 * and a dead row makes every one of its pixels agree with its neighbours' median,
 * so the bad-pixel filter walks straight past it.
 */
static void repair_rows(const fusion_t *f, uint8_t *dst, uint8_t *row_bad, int *nrebuilt)
{
    const int w = f->cfg.th_w, h = f->cfg.th_h;
    const int flat = f->cfg.deadrow_flat;
    const int lift = f->cfg.deadrow_lift;

    if (row_bad) memset(row_bad, 0, (size_t)h);
    *nrebuilt = 0;
    if (flat <= 0 || lift <= 0) return;

    uint8_t bad[512];
    if (h > (int)sizeof(bad)) return;

    /*
     * Frame median, by histogram. The reference has to be the scene's own level,
     * not a constant: the dead rows keep a fixed offset while the frame's level
     * drifts underneath them, so any absolute threshold is right for part of a
     * sequence and wrong for the rest.
     *
     * The dead rows bias the median not at all - they are ~12% of the frame and
     * all at the top end, and the 50th percentile does not notice.
     */
    int32_t hist[256];
    memset(hist, 0, sizeof(hist));
    for (size_t i = 0, e = (size_t)w * h; i < e; i++) hist[dst[i]]++;

    const int32_t half = (int32_t)w * h / 2;
    int32_t acc = 0, median = 0;
    for (int v = 0; v < 256; v++) {
        acc += hist[v];
        if (acc > half) { median = v; break; }
    }

    int nbad = 0;
    for (int y = 0; y < h; y++) {
        const uint8_t *row = dst + (size_t)y * w;
        int lo = 255, hi = 0;
        int32_t sum = 0;

        for (int x = 0; x < w; x++) {
            const int v = row[x];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
            sum += v;
        }
        bad[y] = (uint8_t)(hi - lo <= flat && (sum / w) - median >= lift);
        nbad += bad[y];
    }
    if (!nbad) return;

    /*
     * Safety valve. The defect is 14 rows out of 120; if a quarter of the frame
     * trips the test we are not looking at dead rows but at a scene that is
     * genuinely flat and hot over most of the field. Interpolating across that
     * would erase the hottest target in the frame and report the wall instead -
     * the one failure of this repair that could hide a fault rather than expose
     * one. Leave the frame alone and let the operator see it.
     */
    if (nbad * 4 > h) return;

    for (int y = 0; y < h; y++) {
        if (!bad[y]) continue;

        int a = -1, b = -1;
        for (int k = y - 1; k >= 0; k--) if (!bad[k]) { a = k; break; }
        for (int k = y + 1; k < h; k++)  if (!bad[k]) { b = k; break; }
        if (a < 0 && b < 0) continue;                   /* nothing left to rebuild from */

        uint8_t *row = dst + (size_t)y * w;
        if (a < 0 || b < 0) {
            memcpy(row, dst + (size_t)(a < 0 ? b : a) * w, (size_t)w);
        } else {
            /* Q8 weight toward the lower row; the sources are live rows, so
               copying from dst rather than a snapshot is safe. */
            const int wq = ((y - a) * 256) / (b - a);
            const uint8_t *ra = dst + (size_t)a * w;
            const uint8_t *rb = dst + (size_t)b * w;
            for (int x = 0; x < w; x++)
                row[x] = (uint8_t)((ra[x] * (256 - wq) + rb[x] * wq + 128) >> 8);
        }
        if (row_bad) row_bad[y] = 1;
        (*nrebuilt)++;
    }
}

/*
 * The DC step across one VoSPI segment boundary, in Q8 codes.
 *
 * Two rows either side - enough to see the step, short enough that a scene
 * gradient barely registers - but reduced with a median over columns rather
 * than a mean over all of them.
 *
 * The seam is a constant offset, so every column sees the same step plus its
 * own scene. Averaging lets one hot target crossing the boundary bias the
 * estimate, and that bias is then subtracted from all 30 rows of the segment:
 * a local feature becomes a frame-wide band. The median throws those columns
 * away for free. Measured seam steps on real captures are -5.12, +0.09 and
 * -0.69 codes against a typical row-to-row step of 1.23, so the correction is
 * doing real work and deserves a robust estimator.
 *
 * Read from dst, i.e. after the row repair: a dead row landing on a seam would
 * otherwise be measured as a ~200-code step. Every offset is computed before
 * any is applied, so this still sees unmodified data.
 *
 * Selected by counting rather than sorting - the per-column step is a
 * difference of two 2-pixel sums, so it is bounded to +-510 by construction and
 * needs no comparisons. The histogram is 2 KB of stack, live only for this
 * call; counts cannot overflow uint16 because fusion_init() caps th_w at 256.
 */
#define SEAM_STEP_MAX 510

static int32_t seam_step_q8(const uint8_t *dst, int w, int seam_row)
{
    uint16_t hist[2 * SEAM_STEP_MAX + 1];
    memset(hist, 0, sizeof(hist));

    const uint8_t *a1 = dst + (size_t)(seam_row - 2) * w;
    const uint8_t *a0 = dst + (size_t)(seam_row - 1) * w;
    const uint8_t *b0 = dst + (size_t)seam_row * w;
    const uint8_t *b1 = dst + (size_t)(seam_row + 1) * w;

    for (int x = 0; x < w; x++)
        hist[(b0[x] + b1[x]) - (a0[x] + a1[x]) + SEAM_STEP_MAX]++;

    int32_t acc = 0, mid = 0;
    for (int i = 0; i <= 2 * SEAM_STEP_MAX; i++) {
        acc += hist[i];
        if (acc * 2 > w) { mid = i - SEAM_STEP_MAX; break; }
    }
    /* mid is a difference of two-pixel sums, so the per-pixel step is mid/2 */
    return rdiv(mid * 256, 2);
}

void fusion_thermal_prep(fusion_t *f, const uint8_t *src, uint8_t *dst)
{
    const int w = f->cfg.th_w, h = f->cfg.th_h;
    const int seg = f->cfg.th_seg_rows;

    memcpy(dst, src, (size_t)w * h);
    repair_rows(f, dst, f->row_bad, &f->rows_rebuilt);

    if (seg > 0 && seg < h) {
        const int nseg = h / seg;
        /*
         * Segment offsets, Q8. They accumulate across seams, so the per-seam
         * estimate has to carry a fraction: rounded to whole codes the
         * truncation compounds, and by the last of four segments it reached 3
         * codes - 0.42C at the 0.141 C/LSB an auto-ranged session picks, applied
         * as a 30-row band in every frame and looking exactly like scene.
         */
        int32_t off[16];
        if (nseg <= (int)(sizeof(off) / sizeof(off[0]))) {
            off[0] = 0;
            for (int s = 1; s < nseg; s++)
                off[s] = off[s - 1] + seam_step_q8(dst, w, s * seg);

            int32_t mean = 0;
            for (int s = 0; s < nseg; s++) mean += off[s];
            mean = rdiv(mean, nseg);

            for (int s = 0; s < nseg; s++) {
                const int32_t d = rdiv(off[s] - mean, 256);
                if (!d) continue;
                for (int y = s * seg; y < (s + 1) * seg; y++) {
                    uint8_t *row = dst + (size_t)y * w;
                    for (int x = 0; x < w; x++) {
                        int v = row[x] - d;
                        row[x] = (uint8_t)CLAMP255(v);
                    }
                }
            }
        }
    }

    if (f->cfg.badpix_thresh > 0) {
        const int th = f->cfg.badpix_thresh;
        for (int y = 1; y < h - 1; y++) {
            const uint8_t *up = dst + (size_t)(y - 1) * w;
            uint8_t *cu = dst + (size_t)y * w;
            const uint8_t *dn = dst + (size_t)(y + 1) * w;
            for (int x = 1; x < w - 1; x++) {
                int m = med4(up[x], dn[x], cu[x - 1], cu[x + 1]);
                int diff = cu[x] - m;
                if (diff > th || diff < -th) cu[x] = (uint8_t)m;
            }
        }
    }
}

/* ------------------------------------------------------------------ warp */

/*
 * Install a warp table, rejecting anything that would sample outside the
 * thermal frame.
 *
 * This is not defensive tidying. fusion_warp_thermal() clamps x1/y1 - the
 * far corner of the bilinear tap - but takes x0/y0 straight from the table, so
 * a single entry of 0xFA00 (250.0 thermal px, in range for the type, not the
 * sentinel, and passing py_fusion.c's length-only check) reads hundreds of
 * bytes past the frame. The table is the one input that arrives from outside
 * the library, built by a separate toolchain, so it is exactly the input that
 * cannot be assumed well-formed.
 *
 * An out-of-range entry becomes FUSION_INVALID rather than being clamped to the
 * edge. A clamped coordinate is a plausible reading taken from the wrong pixel,
 * which is the failure the sentinel exists to prevent; no coverage is honest.
 *
 * Returns the number of entries rejected - 0 for a well-formed table. A nonzero
 * count means the calibration and this pipeline disagree about geometry, so it
 * is worth surfacing rather than absorbing.
 */
int fusion_set_warp(fusion_t *f, const uint16_t *warp)
{
    const size_t n = (size_t)f->cfg.low_w * f->cfg.low_h;
    /* x0 = qx >> 8 must land in [0, th_w-1], so qx < th_w << 8. The fractional
       part needs no guard: x1 is already clamped, so a full 0xFF frac at the
       last column interpolates the edge pixel with itself. */
    const uint32_t xlim = (uint32_t)f->cfg.th_w << 8;
    const uint32_t ylim = (uint32_t)f->cfg.th_h << 8;
    int rejected = 0;

    for (size_t i = 0; i < n; i++) {
        uint16_t qx = warp[2 * i], qy = warp[2 * i + 1];

        if (qx == FUSION_INVALID || qy == FUSION_INVALID) {
            qx = qy = FUSION_INVALID;               /* half-marked entry: both out */
        } else if ((uint32_t)qx >= xlim || (uint32_t)qy >= ylim) {
            qx = qy = FUSION_INVALID;
            rejected++;
        }
        f->warp[2 * i]     = qx;
        f->warp[2 * i + 1] = qy;
    }
    return rejected;
}

#if FUSION_ENABLE_HOMOGRAPHY
void fusion_set_homography(fusion_t *f, const double H[9])
{
    const int lw = f->cfg.low_w, lh = f->cfg.low_h;
    const double tw = f->cfg.th_w - 1, th = f->cfg.th_h - 1;

    for (int y = 0; y < lh; y++) {
        for (int x = 0; x < lw; x++) {
            double u = H[0] * x + H[1] * y + H[2];
            double v = H[3] * x + H[4] * y + H[5];
            double w = H[6] * x + H[7] * y + H[8];
            uint16_t *e = f->warp + 2 * ((size_t)y * lw + x);

            if (w == 0.0) {
                e[0] = e[1] = FUSION_INVALID;
                continue;
            }
            u /= w;
            v /= w;
            if (u < 0.0 || v < 0.0 || u > tw || v > th) {
                e[0] = e[1] = FUSION_INVALID;
            } else {
                e[0] = (uint16_t)(u * 256.0 + 0.5);
                e[1] = (uint16_t)(v * 256.0 + 0.5);
            }
        }
    }
}
#endif /* FUSION_ENABLE_HOMOGRAPHY */

void fusion_warp_thermal(const fusion_t *f, const uint8_t *thermal)
{
    const int lw = f->cfg.low_w, lh = f->cfg.low_h;
    const int tw = f->cfg.th_w, th = f->cfg.th_h;

    for (size_t i = 0, e = (size_t)lw * lh; i < e; i++) {
        uint16_t qx = f->warp[2 * i], qy = f->warp[2 * i + 1];

        if (qx == FUSION_INVALID || qy == FUSION_INVALID) {
            f->cover[i] = 0;
            f->t_reg[i] = 0;
            continue;
        }

        int x0 = qx >> 8, y0 = qy >> 8;
        int fxp = qx & 0xFF, fyp = qy & 0xFF;
        int x1 = x0 + 1 < tw ? x0 + 1 : tw - 1;
        int y1 = y0 + 1 < th ? y0 + 1 : th - 1;

        const uint8_t *r0 = thermal + (size_t)y0 * tw;
        const uint8_t *r1 = thermal + (size_t)y1 * tw;
        int top = r0[x0] * (256 - fxp) + r0[x1] * fxp;
        int bot = r1[x0] * (256 - fxp) + r1[x1] * fxp;

        f->t_reg[i] = (uint8_t)((top * (256 - fyp) + bot * fyp + 32768) >> 16);
        f->cover[i] = 1;
    }
}

/* ------------------------------------------------- temporal noise filter
 *
 * See fusion.h for the kernel and the reasoning. Everything here is integer:
 * the knee arrives as a temperature and is converted to Q8 codes against the
 * sensor range, which is the only place the two units meet.
 */

/*
 * Knee in Q8 code units, from a noise figure in milli-Celsius.
 *
 * The sensor maps [tmin,tmax] onto 0..255, so one code is span/255 milli-C and
 *
 *     knee_codes = noise_mc * 255 / span_mc
 *
 * At the 20..40C window this project uses that is 148 * 255 / 20000 = 1.89
 * codes; at the -10..140C default it is 0.25, i.e. below the quantisation step,
 * which is the correct answer - at that span the sensor's own quantisation is
 * coarser than its noise and there is nothing for a temporal filter to remove.
 *
 * Clamped to 32 codes. A knee wider than that would smooth across real scene
 * structure, and the only way to reach it is a misconfigured range - exactly the
 * mistake fusion_set_range() exists to make visible rather than silent.
 */
/*
 * The gate is set at FUSION_TEMPORAL_GATE times the noise figure, not at the
 * noise figure itself, and the difference is not cosmetic - the first version
 * of this filter ramped straight up from zero delta and measured 1.05x.
 *
 * The quantity being discriminated is the difference between two frames, which
 * for independent noise has sigma sqrt(2) times the per-frame figure. A gate at
 * 1 sigma therefore sits in the middle of the noise distribution: a typical
 * noise delta lands halfway up the ramp, gets a blend weight near 1/2 instead
 * of 1/8, and almost nothing is averaged. The gate has to clear the noise
 * distribution, not mark its centre. 3x the per-frame figure is ~2.1 sigma of
 * the difference, which passes the overwhelming majority of noise deltas
 * through to full smoothing.
 *
 * A generous gate is cheap here in a way that is worth being explicit about,
 * because the instinct from spatial filtering is the opposite: this filter is
 * purely temporal, so a static scene feature is preserved exactly however hard
 * it is smoothed - the IIR converges on its true value. What a wide gate costs
 * is temporal responsiveness, not spatial detail. At 8 frames and 8.77 fps a
 * genuine change below the gate surfaces over ~0.9s, which for inspection work
 * is not a cost at all.
 */
#define FUSION_TEMPORAL_GATE 3

static int32_t temporal_gate_q8(const fusion_t *f)
{
    const int32_t noise_mc = f->cfg.temporal_noise_mc;
    const int64_t span = (int64_t)f->tmax_mc - f->tmin_mc;

    if (noise_mc <= 0 || span <= 0) return 0;

    int64_t g = ((int64_t)noise_mc * FUSION_TEMPORAL_GATE * 255 * 256) / span;
    if (g > (32 << 8)) g = 32 << 8;
    return (int32_t)g;
}

/*
 * One pixel, three regimes:
 *
 *     d <= gate       noise      -> full smoothing at w_min
 *     gate < d < 2g   ambiguous  -> ramp w_min .. 256
 *     d >= 2*gate     motion     -> pass through
 *
 * The ramp exists so that a pixel on the edge of a moving target does not
 * alternate between fully smoothed and fully raw from frame to frame, which
 * reads as a shimmering outline.
 *
 * THE HISTORY IS Q8, NOT 8-BIT, AND THAT IS LOAD-BEARING.
 *
 * An IIR whose state is held at the same precision as its input stalls. The
 * per-frame update is (cur - prev) * w / 256; at w = 1/8 that is under half an
 * LSB for any |cur - prev| < 4, so it rounds to zero and the state never moves.
 * The filter freezes on the first frame it saw and returns a single noisy
 * sample forever.
 *
 * This is not a subtle degradation - it is invisible on a short run and total on
 * a long one, and it gets *worse* as the filter is asked to smooth harder. The
 * first version of this code measured 2.0x at 4 frames, 1.2x at 8 and exactly
 * 1.0x at 16 and 32, which is the signature: past w = 1/4 nothing but the frozen
 * first frame is left. Anyone porting this to a fixed-point DSP will meet the
 * same wall, so it is worth stating plainly rather than leaving in the type.
 *
 * At Q8 the smallest update that survives rounding is 1/256 of a code, three
 * orders below the noise floor the filter exists to attack.
 *
 * Widest intermediate is 256 * 65280 = 16.7M, comfortably inside int32.
 */
static inline int32_t temporal_blend_q8(int cur, int32_t prev_q8,
                                        int32_t gate_q8, int32_t w_min)
{
    const int32_t cur_q8 = (int32_t)cur << 8;
    const int32_t diff = cur_q8 - prev_q8;
    const int32_t d_q8 = diff < 0 ? -diff : diff;

    if (d_q8 >= 2 * gate_q8) return cur_q8;

    int32_t w = w_min;
    if (d_q8 > gate_q8)
        w = w_min + (((256 - w_min) * (d_q8 - gate_q8)) / gate_q8);

    return (w * cur_q8 + (256 - w) * prev_q8 + 128) >> 8;
}

/* Q8 state back to a pixel. */
static inline uint8_t temporal_round(int32_t v_q8)
{
    const int32_t v = (v_q8 + 128) >> 8;
    return (uint8_t)CLAMP255(v);
}

/* w_min from a frame count, clamped so that "1 frame" means "no filtering"
   rather than a divide that quietly rounds to heavy smoothing. */
static int32_t temporal_wmin(int frames)
{
    if (frames < 2) return 256;
    if (frames > 256) frames = 256;
    return 256 / frames;
}

void fusion_thermal_temporal(fusion_t *f)
{
    const size_t n = (size_t)f->cfg.th_w * f->cfg.th_h;
    const int32_t gate = temporal_gate_q8(f);
    const int32_t w_min = temporal_wmin(f->cfg.temporal_frames);

    f->temporal_moved = 0;

    if (gate <= 0 || w_min >= 256 || !f->t_prev) return;

    if (!f->t_prev_valid) {
        for (size_t i = 0; i < n; i++)
            f->t_prev[i] = (uint16_t)((int32_t)f->t_prep[i] << 8);
        f->t_prev_valid = 1;
        f->temporal_moved = (int)n;      /* the first frame is all "motion" */
        return;
    }

    for (size_t i = 0; i < n; i++) {
        const int cur = f->t_prep[i];
        const int32_t prev = f->t_prev[i];
        const int32_t v = temporal_blend_q8(cur, prev, gate, w_min);

        if (v == ((int32_t)cur << 8) && v != prev) f->temporal_moved++;

        f->t_prev[i] = (uint16_t)v;
        f->t_prep[i] = temporal_round(v);
    }
}

const uint8_t *fusion_y_temporal(fusion_t *f, const uint8_t *y)
{
    const size_t n = (size_t)f->cfg.out_w * f->cfg.out_h;
    const int32_t gate = (int32_t)f->cfg.y_temporal_knee << 8;
    const int32_t w_min = temporal_wmin(f->cfg.y_temporal_frames);

    if (gate <= 0 || w_min >= 256 || !f->y_prev || !f->y_out) return y;

    if (!f->y_prev_valid) {
        for (size_t i = 0; i < n; i++)
            f->y_prev[i] = (uint16_t)((int32_t)y[i] << 8);
        memcpy(f->y_out, y, n);
        f->y_prev_valid = 1;
        return f->y_out;
    }

    for (size_t i = 0; i < n; i++) {
        const int32_t v = temporal_blend_q8(y[i], f->y_prev[i], gate, w_min);
        f->y_prev[i] = (uint16_t)v;
        f->y_out[i] = temporal_round(v);
    }

    return f->y_out;
}

void fusion_temporal_reset(fusion_t *f)
{
    f->t_prev_valid = 0;
    f->y_prev_valid = 0;
    f->temporal_moved = 0;
}

/* ------------------------------------------------------------------ AGC
 *
 * Percentile stretch of the registered thermal plane, in place.
 *
 * Only covered pixels are counted - uncovered ones are held at 0 by
 * fusion_warp_thermal(), and letting a wide invalid border vote would drag the
 * low cut down to nothing and undo the stretch entirely.
 *
 * Runs on the low-res grid (16k pixels at the default config), so the histogram
 * pass is small compared to anything else in the pipeline.
 */
void fusion_agc(fusion_t *f)
{
    const size_t n = (size_t)f->cfg.low_w * f->cfg.low_h;
    const int permille = f->cfg.agc_permille;

    if (permille <= 0 || permille >= 500) return;

    int32_t hist[256];
    memset(hist, 0, sizeof(hist));

    int32_t total = 0;
    for (size_t i = 0; i < n; i++) {
        if (!f->cover[i]) continue;
        hist[f->t_reg[i]]++;
        total++;
    }
    if (total < 16) return;                 /* almost nothing registered */

    const int32_t cut = (int32_t)(((int64_t)total * permille) / 1000);

    int32_t acc = 0, lo = 0, hi = 255;
    for (int v = 0; v < 256; v++) {
        acc += hist[v];
        if (acc > cut) { lo = v; break; }
    }
    acc = 0;
    for (int v = 255; v >= 0; v--) {
        acc += hist[v];
        if (acc > cut) { hi = v; break; }
    }

    /*
     * Bound on the stretch, and the reason this AGC is safe to run at all.
     *
     * capture.py already floors the sensor's own range at NETD (255 codes over
     * no less than 12.75C), so one code is guaranteed to carry real signal.
     * Stretching a window of S codes to 255 multiplies both the signal and that
     * noise by 255/S. Holding S at 64 or more caps the amplification at 4x,
     * which keeps a flat wall reading as flat instead of torn - the exact
     * failure the capture-side floor was added to prevent.
     *
     * Below the floor the window is widened about its centre rather than the
     * stretch being abandoned, so a low-contrast scene still gets what
     * headroom it has earned.
     */
    const int32_t span_min = 64;
    if (hi - lo < span_min) {
        int32_t mid = (lo + hi) / 2;
        lo = mid - span_min / 2;
        hi = lo + span_min;
    }

    if (!f->agc_valid) {
        f->agc_lo = lo;
        f->agc_hi = hi;
        f->agc_valid = 1;
    } else {
        f->agc_lo = (3 * f->agc_lo + lo) / 4;
        f->agc_hi = (3 * f->agc_hi + hi) / 4;
    }

    lo = f->agc_lo;
    hi = f->agc_hi;
    if (hi - lo < 1) return;

    uint8_t lut[256];
    const int32_t span = hi - lo;
    for (int v = 0; v < 256; v++) {
        int32_t s = ((int32_t)(v - lo) * 255 + span / 2) / span;
        lut[v] = (uint8_t)CLAMP255(s);
    }

    for (size_t i = 0; i < n; i++)
        f->t_reg[i] = lut[f->t_reg[i]];
}

/* ------------------------------------------------------------------ guided filter
 *
 * He/Sun/Tang guided filter, fast variant: coefficients are solved on the
 * low-res grid and only a,b are carried to full resolution.
 *
 *   a = cov(I,p) / (var(I) + eps),  b = mean(p) - a*mean(I)
 *
 * a is held in Q16. cov and var are at most 65025, so the numerator is promoted
 * to int64 for the shift; that division runs on low_w*low_h pixels only.
 */
void fusion_guided(fusion_t *f)
{
    const int lw = f->cfg.low_w, lh = f->cfg.low_h;
    const size_t n = (size_t)lw * lh;
    const int r = f->cfg.gf_radius;
    const int32_t eps = f->cfg.gf_eps;

    int32_t *mean_i = f->a_q16;     /* borrowed until the final pass */
    int32_t *mean_p = f->b_q16;
    int32_t *acc = f->s1;
    int32_t *tmp = f->s2;

    /* mean_I */
    for (size_t i = 0; i < n; i++) acc[i] = f->y_low[i];
    box_i32(acc, mean_i, lw, lh, r, tmp);

    /* mean_p */
    for (size_t i = 0; i < n; i++) acc[i] = f->t_reg[i];
    box_i32(acc, mean_p, lw, lh, r, tmp);

    /* var_I = mean(I*I) - mean_I^2, reusing acc for the result */
    for (size_t i = 0; i < n; i++) acc[i] = (int32_t)f->y_low[i] * f->y_low[i];
    box_i32(acc, acc, lw, lh, r, tmp);
    for (size_t i = 0; i < n; i++) acc[i] -= mean_i[i] * mean_i[i];

    /* cov_Ip = mean(I*p) - mean_I*mean_p, in tmp's sibling buffer */
    int32_t *cov = f->s2 + n;
    int32_t *scratch = f->s2;
    for (size_t i = 0; i < n; i++) cov[i] = (int32_t)f->y_low[i] * f->t_reg[i];
    box_i32(cov, cov, lw, lh, r, scratch);
    for (size_t i = 0; i < n; i++) cov[i] -= mean_i[i] * mean_p[i];

    /* a, b
     *
     * a is a regression slope, so it is bounded in practice - but a
     * low-contrast guide window drives the denominator toward eps and would
     * let it explode. Clamping to +-4 keeps a*mean_I inside int32 (4<<16 * 255
     * = 66.8M) and costs nothing visually: slopes past 4 are noise amplifiers,
     * not signal.
     */
    for (size_t i = 0; i < n; i++) {
        int32_t denom = acc[i] + eps;
        if (denom < 1) denom = 1;

        /* covariance is signed and shifting a negative left is UB - multiply */
        int64_t a64 = (int64_t)cov[i] * 65536 / denom;
        if (a64 >  (4 << 16)) a64 =  (4 << 16);
        if (a64 < -(4 << 16)) a64 = -(4 << 16);

        int32_t a = (int32_t)a64;
        /* a is Q16 and mean_i is a plain integer, so a*mean_i is already Q16 */
        int32_t b = ((int32_t)mean_p[i] << 16) - a * mean_i[i];

        f->a_q16[i] = a;
        f->b_q16[i] = b;
    }

    /* smooth a,b - this is what stops the output banding at window edges */
    box_i32(f->a_q16, f->a_q16, lw, lh, r, f->s1);
    box_i32(f->b_q16, f->b_q16, lw, lh, r, f->s1);
}

/* ------------------------------------------------------------------ pipeline */

static void build_palettes(void);

/*
 * Output coordinate -> position in the low-res coefficient grid.
 *
 * Low-res pixel k is the box average of output columns [k*s, k*s+s-1], so its
 * centre sits at k*s + (s-1)/2, not at k*s. Ignoring that half-pixel offset
 * shifts the whole thermal layer by s/2 output pixels - 2px at s=4, which is a
 * third of the registration budget the 12mm baseline buys us.
 */
/* RGB888 or RGB565, chosen once per build of the pipeline rather than per pixel
   in spirit - the branch is on a config field the compiler hoists easily. */
static inline void store(const fusion_t *f, uint8_t *row8, uint16_t *row16,
                         int x, int r, int g, int b)
{
    if (f->cfg.out_rgb565) {
        row16[x] = (uint16_t)(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
    } else {
        row8[x * 3 + 0] = (uint8_t)r;
        row8[x * 3 + 1] = (uint8_t)g;
        row8[x * 3 + 2] = (uint8_t)b;
    }
}

static inline void grid_pos(int o, int s, int limit, int *idx, int *frac)
{
    int q = ((o * 2 - (s - 1)) * 256) / (2 * s);
    if (q < 0) q = 0;

    int i = q >> 8;
    if (i >= limit - 1) {
        *idx = limit - 1;
        *frac = 0;
    } else {
        *idx = i;
        *frac = q & 0xFF;
    }
}

size_t fusion_sizeof_state(void) { return sizeof(fusion_t); }
size_t fusion_sizeof_cfg(void)   { return sizeof(fusion_cfg_t); }

void fusion_default_cfg(fusion_cfg_t *cfg)
{
    cfg->out_w = 640;
    cfg->out_h = 400;
    cfg->low_w = 160;
    cfg->low_h = 100;
    cfg->th_w = 160;
    cfg->th_h = 120;
    cfg->gf_radius = 4;
    cfg->gf_eps = 200;
    cfg->detail_radius = 2;
    cfg->detail_gain = 200;     /* Q8 -> ~0.78 */
    cfg->detail_invert = 0;
    cfg->agc_permille = 0;      /* off: absolute tone by default, see fusion.h */
    cfg->th_seg_rows = 30;      /* Lepton 3.5: 120 rows in four segments */
    cfg->badpix_thresh = 40;
    /* Measured over 46 real frames: dead rows spread 0-14 codes and sit ~190
       above the frame median, ordinary rows spread 55-90 and sit within ~5.
       These two sit in the middle of that gap, nowhere near either edge. */
    cfg->deadrow_flat = 24;
    cfg->deadrow_lift = 64;
    /* 148 mK is the pessimistic end of the measured common-mode-removed spread
       (126-148), not the 33 mK NETD - see fusion.h. 8 frames buys ~3.9x. */
    cfg->temporal_noise_mc = 148;
    cfg->temporal_frames = 8;
    /* Visible filter off: it costs 256KB at 640x400 and only pays in the dark.
       See fusion.h for why that is opt-in on this part. */
    cfg->y_temporal_knee = 0;
    cfg->y_temporal_frames = 8;
    cfg->show_uncovered = 1;
    cfg->out_rgb565 = 0;
}

int fusion_init(fusion_t *f, const fusion_cfg_t *cfg)
{
    memset(f, 0, sizeof(*f));
    f->cfg = *cfg;
    build_palettes();

    if (cfg->gf_radius < 1 || cfg->gf_radius > 8)
        return -1;                                  /* see the box_i32 overflow note */
    if (cfg->out_w % cfg->low_w || cfg->out_h % cfg->low_h)
        return -1;                                  /* decimation must be integral */
    /*
     * The warp table is Q8 in a uint16, so a thermal coordinate tops out at
     * 255.996 px - the format cannot address a 320- or 640-wide core. Better to
     * refuse at init than to fit a bigger sensor one day and find the table
     * silently wrapping. Two is the minimum the bilinear tap needs.
     */
    if (cfg->th_w < 2 || cfg->th_h < 2 || cfg->th_w > 256 || cfg->th_h > 256)
        return -1;

    const size_t n = (size_t)cfg->low_w * cfg->low_h;
    const size_t full = (size_t)cfg->out_w * cfg->out_h;

    /* s2 is shared: guided-filter scratch (2n) and the full-res box ring.
       The ring is the bigger ask for wide frames with a large detail radius. */
    const size_t ring_need = (size_t)(2 * cfg->detail_radius + 2) * cfg->out_w;
    const size_t s2_len = (2 * n > ring_need) ? 2 * n : ring_need;

    f->warp  = FUSION_ALLOC(sizeof(uint16_t) * 2 * n);
    f->y_low = FUSION_ALLOC(n);
    f->t_prep = FUSION_ALLOC((size_t)cfg->th_w * cfg->th_h);
    f->t_reg = FUSION_ALLOC(n);
    f->cover = FUSION_ALLOC(n);
    f->a_q16 = FUSION_ALLOC(sizeof(int32_t) * n);
    f->b_q16 = FUSION_ALLOC(sizeof(int32_t) * n);
    f->s1    = FUSION_ALLOC(sizeof(int32_t) * n);
    f->s2    = FUSION_ALLOC(sizeof(int32_t) * s2_len); /* guided scratch + cov, or box ring */
    f->blur  = FUSION_ALLOC(full);
    f->row_bad = FUSION_ALLOC((size_t)cfg->th_h);

    /* Temporal histories, allocated only for the filters that are enabled. The
       visible one is the expensive half - 256KB at 640x400 against 19KB for the
       thermal - so an unconfigured build pays nothing for it. */
    if (cfg->temporal_noise_mc > 0 && cfg->temporal_frames > 1) {
        f->t_prev = FUSION_ALLOC(sizeof(uint16_t) * (size_t)cfg->th_w * cfg->th_h);
        if (!f->t_prev) { fusion_free(f); return -1; }
    }
    if (cfg->y_temporal_knee > 0 && cfg->y_temporal_frames > 1) {
        f->y_prev = FUSION_ALLOC(sizeof(uint16_t) * full);
        f->y_out  = FUSION_ALLOC(full);
        if (!f->y_prev || !f->y_out) { fusion_free(f); return -1; }
    }

    if (!f->warp || !f->y_low || !f->t_prep || !f->t_reg || !f->cover || !f->a_q16 ||
        !f->b_q16 || !f->s1 || !f->s2 || !f->blur || !f->row_bad) {
        fusion_free(f);
        return -1;
    }

    for (size_t i = 0; i < 2 * n; i++)
        f->warp[i] = FUSION_INVALID;

    /* capture.py's own defaults. Auto-range moves them every session, so
       anything that reads temperatures must call fusion_set_range(). */
    f->tmin_mc = -10000;
    f->tmax_mc = 140000;
    f->eps_q10 = 1024;          /* 1.0: what the sensor already assumes */
    f->refl_mc = 20000;         /* 20C ambient */

    f->palette = fusion_ironbow;
    return 0;
}

void fusion_free(fusion_t *f)
{
    FUSION_FREE(f->warp);
    FUSION_FREE(f->y_low);
    FUSION_FREE(f->t_prep);
    FUSION_FREE(f->t_reg);
    FUSION_FREE(f->cover);
    FUSION_FREE(f->a_q16);
    FUSION_FREE(f->b_q16);
    FUSION_FREE(f->s1);
    FUSION_FREE(f->s2);
    FUSION_FREE(f->blur);
    FUSION_FREE(f->row_bad);
    FUSION_FREE(f->t_prev);
    FUSION_FREE(f->y_prev);
    FUSION_FREE(f->y_out);
    memset(f, 0, sizeof(*f));
}

void fusion_set_palette(fusion_t *f, const uint8_t (*palette)[3])
{
    f->palette = palette ? palette : fusion_ironbow;
}

void fusion_process(fusion_t *f, const uint8_t *y, const uint8_t *thermal, uint8_t *out)
{
    const int ow = f->cfg.out_w, oh = f->cfg.out_h;
    const int lw = f->cfg.low_w, lh = f->cfg.low_h;
    const int sx = ow / lw, sy = oh / lh;
    const int gain = f->cfg.detail_gain;

    fusion_decimate(y, ow, oh, f->y_low, lw, lh);
    fusion_thermal_prep(f, thermal, f->t_prep);
    /* Before the warp, so the filter runs on the sensor grid where the noise is
       independent per pixel. After the warp it would be filtering interpolated
       samples, which are already correlated with their neighbours - the same
       reason the row repair runs here rather than downstream. */
    fusion_thermal_temporal(f);
    fusion_warp_thermal(f, f->t_prep);
    fusion_agc(f);
    fusion_guided(f);
    fusion_box_u8(y, f->blur, ow, oh, f->cfg.detail_radius, f->s2);

    /* t_prep and the warp are now consistent with what is about to be drawn, so
       a temperature query is answerable from here on. */
    f->have_frame = 1;

    for (int oy = 0; oy < oh; oy++) {
        int y0, fy;
        grid_pos(oy, sy, lh, &y0, &fy);
        int y1 = y0 + 1 < lh ? y0 + 1 : y0;

        const uint8_t *yrow = y + (size_t)oy * ow;
        const uint8_t *brow = f->blur + (size_t)oy * ow;
        uint8_t *orow = out + (size_t)oy * ow * 3;
        uint16_t *orow16 = (uint16_t *) out + (size_t)oy * ow;

        for (int ox = 0; ox < ow; ox++) {
            int x0, fx;
            grid_pos(ox, sx, lw, &x0, &fx);
            int x1 = x0 + 1 < lw ? x0 + 1 : x0;

            const size_t i00 = (size_t)y0 * lw + x0, i01 = (size_t)y0 * lw + x1;
            const size_t i10 = (size_t)y1 * lw + x0, i11 = (size_t)y1 * lw + x1;

            const int w00 = (256 - fx) * (256 - fy), w01 = fx * (256 - fy);
            const int w10 = (256 - fx) * fy,         w11 = fx * fy;

            int luma = yrow[ox];
            int detail = luma - brow[ox];
            int det = (detail * gain) >> 8;
            if (f->cfg.detail_invert) det = -det;

            if (f->cfg.show_uncovered &&
                !(f->cover[i00] | f->cover[i01] | f->cover[i10] | f->cover[i11])) {
                /* outside the thermal footprint: show the visible image plainly
                   rather than inventing a temperature for it */
                store(f, orow, orow16, ox, luma, luma, luma);
                continue;
            }

            int64_t a = ((int64_t)f->a_q16[i00] * w00 + (int64_t)f->a_q16[i01] * w01 +
                         (int64_t)f->a_q16[i10] * w10 + (int64_t)f->a_q16[i11] * w11) >> 16;
            int64_t b = ((int64_t)f->b_q16[i00] * w00 + (int64_t)f->b_q16[i01] * w01 +
                         (int64_t)f->b_q16[i10] * w10 + (int64_t)f->b_q16[i11] * w11) >> 16;

            int t_hi = (int)((a * luma + b) >> 16);
            t_hi = CLAMP255(t_hi);

            const uint8_t *pal = f->palette[t_hi];
            store(f, orow, orow16, ox,
                  CLAMP255(pal[0] + det), CLAMP255(pal[1] + det), CLAMP255(pal[2] + det));
        }
    }
}

/* ------------------------------------------------------------------ radiometry
 *
 * See fusion.h for what this path is and why it does not go through the AGC or
 * the guided filter.
 *
 * Everything here is integer, including the fourth root, so the measurement is
 * available on the target and not just on the host. The arithmetic is not in the
 * per-frame path - these are query functions - but pulling in soft-float on a
 * build whose whole point is that it has none would be a poor trade for a
 * handful of operations.
 */

#define FUSION_ZERO_C_MC  273150

/*
 * Clamp on absolute temperature, in deci-Kelvin. T^4 is 3.2e15 here and the
 * emissivity inversion multiplies it by 1024, which lands just inside int64.
 * 750K is well past anything the Lepton's high-gain mode can report anyway.
 */
#define FUSION_TMAX_DK    7500

/* Exact floor(sqrt(x)), restoring bit-by-bit. Two of these give the fourth
   root the emissivity inversion needs. */
static uint64_t isqrt64(uint64_t x)
{
    uint64_t r = 0, bit = (uint64_t)1 << 62;

    while (bit > x) bit >>= 2;
    while (bit) {
        if (x >= r + bit) {
            x -= r + bit;
            r = (r >> 1) + bit;
        } else {
            r >>= 1;
        }
        bit >>= 2;
    }
    return r;
}

/*
 * T^4, with T in deci-Kelvin, from a milli-Celsius temperature.
 *
 * The input is carried at 1/16 dK (6.25 mK) rather than at whole deci-Kelvin,
 * because the emissivity inversion amplifies any error here by roughly
 * (T_app/T_obj)^3 / eps - a factor of ~2.2 at eps 0.1. Rounding the input to a
 * whole deci-Kelvin cost 0.1C at eps 0.95 and 0.23C at eps 0.1, which is visible
 * quantisation on a live readout and was measured, not assumed.
 *
 * Squaring twice would overflow at that scale, so the intermediate is shifted
 * back down by the 2^8 that the Q4 input introduces - a relative loss of 2e-8,
 * six orders of magnitude below the sensor's own resolution.
 */
static int64_t fusion_radiance(int32_t milli_c)
{
    int64_t t_q = (((int64_t)milli_c + FUSION_ZERO_C_MC) * 4 + 12) / 25;  /* 1/16 dK */

    if (t_q < 16) t_q = 16;                                 /* 1 dK floor */
    if (t_q > (int64_t)FUSION_TMAX_DK * 16) t_q = (int64_t)FUSION_TMAX_DK * 16;

    const int64_t r2 = t_q * t_q;           /* 256 * T_dK^2 */
    return (r2 >> 8) * (r2 >> 8);           /* T_dK^4 */
}

void fusion_set_range(fusion_t *f, int32_t tmin_mc, int32_t tmax_mc)
{
    if (tmax_mc <= tmin_mc) return;     /* a zero or inverted span has no meaning */
    f->tmin_mc = tmin_mc;
    f->tmax_mc = tmax_mc;
}

void fusion_set_emissivity(fusion_t *f, int32_t eps_q10, int32_t refl_mc)
{
    if (eps_q10 > 1024) eps_q10 = 1024;
    if (eps_q10 < 51) eps_q10 = 51;     /* 0.05; see fusion.h */
    f->eps_q10 = eps_q10;
    f->refl_mc = refl_mc;
}

int32_t fusion_code_to_milli_c(const fusion_t *f, uint16_t code_q8)
{
    /* codes 0..255 span [tmin,tmax]; code_q8 is that in Q8, so 255*256 = full */
    const int64_t span = (int64_t)f->tmax_mc - f->tmin_mc;
    return (int32_t)(f->tmin_mc + (span * code_q8 + 32640) / 65280);
}

/*
 * Invert  L_meas = eps*L(T_obj) + (1-eps)*L(T_refl)  for T_obj.
 *
 * Radiance is taken as proportional to T^4. That is the Stefan-Boltzmann law for
 * the whole spectrum, and the Lepton only sees 8-14um, where the local exponent
 * is nearer 4.8 around room temperature - so this is an approximation, and worth
 * being explicit about rather than burying. It is the same one FLIR's own
 * simplified correction makes, and its error is second-order in (T_obj - T_refl)
 * while the error it removes is first-order in (1 - eps). At eps 0.9 with a 40C
 * object against a 20C room, the exponent choice moves the answer by a few
 * tenths of a degree; skipping the correction entirely moves it by ~2C. On the
 * shiny metal this exists for - eps 0.1 - the correction is worth tens of
 * degrees and its own inaccuracy is noise beside that.
 *
 * The honest summary for the UI: this turns a badly wrong number into a roughly
 * right one. It does not turn the Lepton into a laboratory instrument.
 */
int32_t fusion_apply_emissivity(const fusion_t *f, int32_t raw_mc)
{
    const int64_t eps = f->eps_q10;
    if (eps >= 1024) return raw_mc;                     /* nothing to correct */

    const int64_t r_app = fusion_radiance(raw_mc);
    const int64_t r_ref = fusion_radiance(f->refl_mc);

    int64_t r_obj = ((r_app - ((1024 - eps) * r_ref) / 1024) * 1024) / eps;
    if (r_obj < 1) r_obj = 1;   /* colder than the correction can express: the
                                   object would have to emit negative radiance,
                                   which means refl_mc is wrong, not the pixel */

    int64_t n = (int64_t)isqrt64(isqrt64((uint64_t)r_obj));
    if (n < 1) n = 1;

    /* One Newton step in fixed point turns the 0.1K lattice of the integer
       fourth root into ~1mK, so a hover readout does not visibly quantise:
       T ~= n + (R - n^4) / (4 n^3), evaluated in milli-Kelvin. */
    const int64_t n3 = n * n * n;
    const int64_t t_mk = n * 100 + ((r_obj - n3 * n) * 100) / (4 * n3);

    return (int32_t)(t_mk - FUSION_ZERO_C_MC);
}

/* Warp lookup at an output pixel. Fills Q8 thermal coords; 0 if uncovered.
 *
 * The four surrounding grid entries are interpolated when all four are valid,
 * which keeps the reading consistent with the picture (fusion_process does the
 * same bilinear over the same stencil). Where the stencil straddles the edge of
 * the thermal footprint the nearest valid entry is used instead: interpolating
 * against a FUSION_INVALID sentinel would silently drag the sample toward the
 * top-left of the thermal frame and report a temperature from somewhere else
 * entirely.
 */
static int temp_warp_at(const fusion_t *f, int ox, int oy, int *tx_q8, int *ty_q8)
{
    const int lw = f->cfg.low_w, lh = f->cfg.low_h;
    const int sx = f->cfg.out_w / lw, sy = f->cfg.out_h / lh;

    int x0, fx, y0, fy;
    grid_pos(ox, sx, lw, &x0, &fx);
    grid_pos(oy, sy, lh, &y0, &fy);
    const int x1 = x0 + 1 < lw ? x0 + 1 : x0;
    const int y1 = y0 + 1 < lh ? y0 + 1 : y0;

    const size_t idx[4] = { (size_t)y0 * lw + x0, (size_t)y0 * lw + x1,
                            (size_t)y1 * lw + x0, (size_t)y1 * lw + x1 };
    const int wgt[4] = { (256 - fx) * (256 - fy), fx * (256 - fy),
                         (256 - fx) * fy,         fx * fy };

    int nvalid = 0, best = -1, best_w = -1;
    for (int k = 0; k < 4; k++) {
        if (f->warp[2 * idx[k]] == FUSION_INVALID ||
            f->warp[2 * idx[k] + 1] == FUSION_INVALID) continue;
        nvalid++;
        if (wgt[k] > best_w) { best_w = wgt[k]; best = k; }
    }
    if (!nvalid) return 0;

    if (nvalid == 4) {
        int64_t u = 0, v = 0;
        for (int k = 0; k < 4; k++) {
            u += (int64_t)f->warp[2 * idx[k]] * wgt[k];
            v += (int64_t)f->warp[2 * idx[k] + 1] * wgt[k];
        }
        *tx_q8 = (int)(u >> 16);
        *ty_q8 = (int)(v >> 16);
    } else {
        *tx_q8 = f->warp[2 * idx[best]];
        *ty_q8 = f->warp[2 * idx[best] + 1];
    }
    return 1;
}

/* Bilinear sample of the prepared thermal frame. Returns a Q8 code and, through
   repaired, whether either source row was reconstructed rather than measured. */
static uint16_t temp_sample(const fusion_t *f, int tx_q8, int ty_q8,
                            int *px, int *py, int *repaired)
{
    const int tw = f->cfg.th_w, th = f->cfg.th_h;

    int x0 = tx_q8 >> 8, y0 = ty_q8 >> 8;
    if (x0 < 0) x0 = 0; else if (x0 > tw - 1) x0 = tw - 1;
    if (y0 < 0) y0 = 0; else if (y0 > th - 1) y0 = th - 1;

    const int fx = tx_q8 & 0xFF, fy = ty_q8 & 0xFF;
    const int x1 = x0 + 1 < tw ? x0 + 1 : x0;
    const int y1 = y0 + 1 < th ? y0 + 1 : y0;

    const uint8_t *r0 = f->t_prep + (size_t)y0 * tw;
    const uint8_t *r1 = f->t_prep + (size_t)y1 * tw;
    const int top = r0[x0] * (256 - fx) + r0[x1] * fx;
    const int bot = r1[x0] * (256 - fx) + r1[x1] * fx;

    *px = x0;
    *py = y0;
    *repaired = f->row_bad && (f->row_bad[y0] || f->row_bad[y1]);
    return (uint16_t)((top * (256 - fy) + bot * fy) >> 8);
}

int fusion_temp_at(const fusion_t *f, int ox, int oy, fusion_temp_t *out)
{
    memset(out, 0, sizeof(*out));

    if (!f->have_frame) return -1;
    if (ox < 0 || oy < 0 || ox >= f->cfg.out_w || oy >= f->cfg.out_h) return -1;

    int tx, ty;
    if (!temp_warp_at(f, ox, oy, &tx, &ty)) return 0;   /* uncovered, not an error */

    int px, py, repaired;
    const uint16_t code = temp_sample(f, tx, ty, &px, &py, &repaired);

    out->code_q8 = code;
    out->th_x = (int16_t)px;
    out->th_y = (int16_t)py;
    out->raw_milli_c = fusion_code_to_milli_c(f, code);
    out->milli_c = fusion_apply_emissivity(f, out->raw_milli_c);
    out->valid = 1;
    out->repaired = (uint8_t)(repaired != 0);
    return 0;
}

int fusion_temp_region(const fusion_t *f, int x0, int y0, int x1, int y1,
                       fusion_region_t *out)
{
    const int lw = f->cfg.low_w, lh = f->cfg.low_h;
    const int sx = f->cfg.out_w / lw, sy = f->cfg.out_h / lh;

    memset(out, 0, sizeof(*out));
    if (!f->have_frame) return -1;

    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > f->cfg.out_w) x1 = f->cfg.out_w;
    if (y1 > f->cfg.out_h) y1 = f->cfg.out_h;
    if (x1 <= x0 || y1 <= y0) return -1;

    int32_t lo = 0, hi = 0;
    int64_t sum = 0;

    for (int gy = 0; gy < lh; gy++) {
        /* centre of the output block this grid cell averages - the inverse of
           grid_pos(), so a sample reports the pixel a user could point at */
        const int oy = gy * sy + (sy - 1) / 2;
        if (oy < y0 || oy >= y1) continue;

        for (int gx = 0; gx < lw; gx++) {
            const int ox = gx * sx + (sx - 1) / 2;
            if (ox < x0 || ox >= x1) continue;

            const size_t i = (size_t)gy * lw + gx;
            const uint16_t qx = f->warp[2 * i], qy = f->warp[2 * i + 1];
            if (qx == FUSION_INVALID || qy == FUSION_INVALID) continue;

            int px, py, repaired;
            const uint16_t code = temp_sample(f, qx, qy, &px, &py, &repaired);
            const int32_t mc = fusion_code_to_milli_c(f, code);

            if (!out->samples || mc < lo) { lo = mc; out->min_x = ox; out->min_y = oy; }
            if (!out->samples || mc > hi) { hi = mc; out->max_x = ox; out->max_y = oy; }
            sum += mc;
            out->samples++;
            out->repaired += (repaired != 0);
        }
    }
    if (!out->samples) return -1;

    /* The emissivity correction is monotonic, so it cannot reorder the extremes
       and the hot/cold pixels found on raw values are the right ones. It is
       applied to the three reported figures rather than to every sample: for the
       mean that is correcting an average instead of averaging corrections, which
       differs only in the second-order term and costs one fourth root instead of
       thousands. */
    out->min_milli_c = fusion_apply_emissivity(f, lo);
    out->max_milli_c = fusion_apply_emissivity(f, hi);
    out->mean_milli_c = fusion_apply_emissivity(f, (int32_t)(sum / out->samples));
    return 0;
}

/* ------------------------------------------------------------------ palettes */

uint8_t fusion_ironbow[256][3];
uint8_t fusion_grayscale[256][3];
uint8_t fusion_whitehot[256][3];
uint8_t fusion_blackhot[256][3];

/*
 * Headroom left at each end of the monochrome ramps for the detail layer.
 *
 * At the default gain the detail term reaches roughly +-40 on hard edges, so a
 * 0..255 ramp would clip texture on anything near the top or bottom of the
 * temperature window - i.e. on the hot targets, which is where the texture is
 * wanted most. 24 gives most of that back for 19% of the tonal range, a trade
 * that reads as sharper, not flatter, because perceived sharpness comes from
 * local gradients rather than from the global span.
 */
#define FUSION_MONO_FLOOR 24
#define FUSION_MONO_CEIL  231

/* Ironbow: black -> violet -> red -> orange -> yellow -> white. Interpolated
 * from control points into a table, so the hot path is one indexed load. */
static void build_palettes(void)
{
    static int done = 0;
    if (done) return;
    done = 1;

    static const struct { int at; uint8_t r, g, b; } key[] = {
        {   0,   0,   0,   0 },
        {  38,  28,   0,  90 },
        {  77, 100,   0, 130 },
        { 115, 170,  20, 100 },
        { 153, 220,  70,  40 },
        { 191, 250, 140,   0 },
        { 230, 255, 210,  40 },
        { 255, 255, 255, 255 },
    };
    const int nkey = (int)(sizeof(key) / sizeof(key[0]));

    for (int k = 0; k < nkey - 1; k++) {
        int span = key[k + 1].at - key[k].at;
        for (int i = 0; i <= span; i++) {
            int idx = key[k].at + i;
            fusion_ironbow[idx][0] = (uint8_t)(key[k].r + (key[k + 1].r - key[k].r) * i / span);
            fusion_ironbow[idx][1] = (uint8_t)(key[k].g + (key[k + 1].g - key[k].g) * i / span);
            fusion_ironbow[idx][2] = (uint8_t)(key[k].b + (key[k + 1].b - key[k].b) * i / span);
        }
    }
    for (int i = 0; i < 256; i++)
        fusion_grayscale[i][0] = fusion_grayscale[i][1] = fusion_grayscale[i][2] = (uint8_t)i;

    const int span = FUSION_MONO_CEIL - FUSION_MONO_FLOOR;
    for (int i = 0; i < 256; i++) {
        uint8_t v = (uint8_t)(FUSION_MONO_FLOOR + (i * span + 127) / 255);
        fusion_whitehot[i][0] = fusion_whitehot[i][1] = fusion_whitehot[i][2] = v;
        fusion_blackhot[i][0] = fusion_blackhot[i][1] = fusion_blackhot[i][2] =
            (uint8_t)(FUSION_MONO_FLOOR + FUSION_MONO_CEIL - v);
    }
}

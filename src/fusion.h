/*
 * Thermal + visible frame fusion for the OpenMV N6.
 *
 * Portable C99: no MicroPython, no OpenMV, no floating point in the hot path.
 * The same translation unit builds for the host (x86/aarch64) and for the
 * Cortex-M55, so the algorithm can be iterated on a workstation against
 * recorded frames and then dropped into firmware unchanged.
 *
 * Pipeline, per fused frame:
 *
 *   Y  (out_w x out_h, 8bpp)          visible luma, straight off the PAG7936
 *    |-- 4x4 box decimate ----------> Y_low   (low_w x low_h)
 *    |-- Y - box_blur(Y) -----------> detail  (full res, signed)
 *
 *   T  (th_w x th_h, 8bpp)            Lepton, already AGC'd into [tmin,tmax]C
 *    |-- warp LUT + bilinear -------> T_reg   (low_w x low_h, registered to Y_low)
 *
 *   guided filter(guide=Y_low, in=T_reg) -> a,b  (low res)
 *   bilinear upsample a,b -> t_hi = (a*Y + b) >> 16      (full res, sharp)
 *   out = palette[t_hi] + gain*detail                    (full res, RGB)
 *
 * The point of the low-res stage: everything expensive - the warp, the guided
 * filter - runs on low_w*low_h pixels, and only the two coefficient planes are
 * upsampled. A full-res warp LUT for 640x400 would be 1.2MB; this one is 64KB.
 *
 * Colour stays bound to temperature: the detail layer is added to the palette
 * output, never to the index. Embossing edges must not move a pixel's hue,
 * because hue is the measurement the user reads.
 */
#ifndef FUSION_H
#define FUSION_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FUSION_INVALID 0xFFFFu  /* warp LUT sentinel: no thermal coverage here */

/*
 * fusion_set_homography() is the only floating-point code in the library, and it
 * is setup-only. Firmware compiles it out (-DFUSION_ENABLE_HOMOGRAPHY=0): on the
 * device the warp table always comes from calibration via fusion_set_warp(), so
 * the target build ends up entirely integer - which is also what OpenMV's
 * -fsingle-precision-constant -Wdouble-promotion -Werror firmware flags demand.
 */
#ifndef FUSION_ENABLE_HOMOGRAPHY
#define FUSION_ENABLE_HOMOGRAPHY 1
#endif

typedef struct {
    int out_w, out_h;     /* fused output, e.g. 640x400 (PAG7936 csi.VGA) */
    int low_w, low_h;     /* working grid, out_w/4 x out_h/4 */
    int th_w, th_h;       /* thermal source, 160x120 */

    int gf_radius;        /* guided-filter radius in low-res px (<= 8, see fusion.c) */
    int gf_eps;           /* guided-filter regularisation, 8-bit units squared */

    int detail_radius;    /* box radius for the high-pass, full-res px */
    int detail_gain;      /* Q8: 256 = unity */
    int detail_invert;    /* 1: subtract the detail layer instead of adding it.
                             For an inverted palette (black-hot) this keeps the
                             embossed texture reading the same way as the base
                             tone - without it the visible texture is a positive
                             laid over a negative and the two fight each other. */

    /*
     * Scene AGC: per-mille of the registered thermal histogram clipped at each
     * end before the range is stretched to 0..255. 0 disables it.
     *
     * This is what makes a monochrome palette usable. The Lepton frame arrives
     * already AGC'd into [tmin,tmax], but an indoor scene fills only a slice of
     * that window, so the fused plane lands in a narrow band - which ironbow
     * disguises with a fast hue sweep and a grey ramp shows for what it is.
     *
     * The cost is real and worth stating: with this on, tone is scene-relative,
     * not absolute. A pixel's grey level no longer maps to a fixed temperature
     * across frames. Leave it off when the image is being measured rather than
     * looked at.
     */
    int agc_permille;

    int th_seg_rows;      /* VoSPI segment height (Lepton 3.5: 30). 0 disables debanding */
    int badpix_thresh;    /* replace a pixel this far from its neighbours' median. 0 disables */

    /*
     * Dead-row repair. This unit returns 14 fixed rows - 6, 12, 28, 35, 36 and
     * the run 55..63 - carrying no scene at all, in every frame. Pinned to fixed
     * row numbers, so the sensor, not VoSPI tearing, which is random.
     *
     * They are NOT simply stuck at 255, which is what they look like at first
     * and what the host-side tooling originally assumed. Measured across 46
     * recorded frames: a dead row sits at a fixed high offset that clips at 255
     * only once the frame's own level rises. On the low-level frames of the same
     * sequence it reads ~238 with a spread of 9-14 codes, and a fixed
     * saturation threshold walks straight past it - on 13 of those 46 frames.
     * Those are precisely the frames where the AGC then stretches the untouched
     * rows into a full-contrast band.
     *
     * So the test is not "how bright" but "is there any scene in this row":
     *
     *   flat  the row's spread is at most deadrow_flat codes
     *   lifted its mean is at least deadrow_lift codes above the frame median
     *
     * The margin is enormous and was measured, not guessed: dead rows spread
     * 0-14 codes and sit ~190 above the median, ordinary rows spread 55-90 and
     * sit within ~5 of it. Both conditions are needed - flat alone condemns a
     * blank wall, lifted alone condemns any hot target.
     *
     * A condemned row is replaced by a linear blend of the nearest live rows
     * above and below. The 55..63 run means that blend can span nine rows; the
     * result is plausible, not measured, and fusion_temp_t.repaired says so.
     *
     * The one scene this can misread is a uniform hot object spanning the full
     * width to within deadrow_flat codes - an edge-to-edge pipe or busbar. Real
     * ones do not survive that test in practice (optics, emissivity and viewing
     * angle all put far more than 24 codes of spread across 160 pixels), but the
     * failure would be silent, so it is worth knowing that deadrow_flat = 0
     * turns the whole thing off.
     */
    int deadrow_flat;     /* max spread across a row for it to count as empty. 0 disables */
    int deadrow_lift;     /* codes above the frame median such a row must sit */

    int show_uncovered;   /* 1: draw plain grey where thermal has no coverage */
    int out_rgb565;       /* 1: write uint16 RGB565 instead of packed RGB888 */
} fusion_cfg_t;

typedef struct {
    fusion_cfg_t cfg;

    uint16_t *warp;       /* 2 per low-res px: thermal x,y in Q8. FUSION_INVALID = none */
    uint8_t  *y_low;      /* decimated guide */
    uint8_t  *t_prep;     /* thermal after debanding + bad-pixel repair */
    uint8_t  *t_reg;      /* thermal registered onto the low-res grid */
    uint8_t  *cover;      /* 0/1 thermal coverage on the low-res grid */

    int32_t  *a_q16;      /* guided-filter coefficients, low res */
    int32_t  *b_q16;

    int32_t  *s1, *s2;    /* box-filter scratch, low res */
    uint8_t  *blur;       /* box-blurred luma, full res */

    /* AGC state. Fixed-size, so it costs nothing when agc_permille is 0.
       lo/hi are carried between frames and eased toward each new estimate:
       a per-frame recompute makes the whole image breathe on live video as
       someone walks through the scene. */
    int32_t  agc_lo, agc_hi;
    int      agc_valid;

    /* Radiometry. See fusion_set_range() / fusion_set_emissivity(). */
    int32_t  tmin_mc, tmax_mc;    /* sensor range endpoints, milli-Celsius */
    int32_t  eps_q10;             /* emissivity, 1024 = 1.0 */
    int32_t  refl_mc;             /* reflected background temperature, milli-C */

    uint8_t  *row_bad;            /* th_h flags: this row was reconstructed */
    int      rows_rebuilt;        /* how many, this frame. A jump here is the
                                     sensor degrading, and worth surfacing. */
    int      have_frame;          /* a frame has been through fusion_process() */

    const uint8_t (*palette)[3];  /* 256 x RGB */
} fusion_t;

/*
 * Sizes of the two structs above, for out-of-process callers that have to
 * mirror the layout - live.py drives this library through ctypes, and
 * fusion_init() memsets sizeof(fusion_t) through the pointer it is handed, so a
 * mirror that is short by one field is a heap overwrite rather than a
 * wrong-looking image. Cheaper to assert than to diagnose.
 */
size_t fusion_sizeof_state(void);
size_t fusion_sizeof_cfg(void);

/* Sensible defaults for 640x400 out / 160x120 Lepton. */
void fusion_default_cfg(fusion_cfg_t *cfg);

/* Returns 0 on success, -1 on allocation failure or an invalid config. */
int  fusion_init(fusion_t *f, const fusion_cfg_t *cfg);
void fusion_free(fusion_t *f);

/*
 * Warp table. Two ways in:
 *
 *  - fusion_set_homography() for the pure planar case. H maps low-res output
 *    coords to thermal coords: [u v w] = H * [x y 1], sampled at (u/w, v/w).
 *
 *  - fusion_set_warp() to install a table built elsewhere. Use this for the
 *    real calibration, where OpenCV has already folded lens distortion of both
 *    cameras into the mapping - a homography alone cannot express that.
 *
 * Entries are Q8 thermal coordinates; either component FUSION_INVALID marks a
 * pixel with no thermal coverage.
 */
#if FUSION_ENABLE_HOMOGRAPHY
void fusion_set_homography(fusion_t *f, const double H[9]);
#endif
void fusion_set_warp(fusion_t *f, const uint16_t *warp);

/* 256-entry RGB palette. NULL restores the built-in ironbow. */
void fusion_set_palette(fusion_t *f, const uint8_t (*palette)[3]);

extern uint8_t fusion_ironbow[256][3];    /* filled by fusion_init() */
extern uint8_t fusion_grayscale[256][3];  /* full 0..255 ramp, no detail headroom */

/*
 * Monochrome mappings, the ones worth using when the aim is a sharp image
 * rather than a legible temperature scale. A grey ramp spends its whole
 * dynamic range on the quantity being sharpened, so injected detail survives
 * instead of being swamped by a hue change; ironbow's mid-band in particular
 * is nearly flat in luma, which is what makes a fused frame look soft there.
 *
 * Both ramps stop short of 0 and 255 (see FUSION_MONO_FLOOR/CEIL in fusion.c)
 * so the detail layer has room to modulate at both ends of the scale. Clipping
 * at the ends is exactly where texture on hot targets would otherwise be lost.
 *
 * black-hot is the white-hot ramp reversed. Pair it with cfg.detail_invert.
 */
extern uint8_t fusion_whitehot[256][3];
extern uint8_t fusion_blackhot[256][3];

/*
 * One fused frame.
 *   y       out_w*out_h   visible luma
 *   thermal th_w*th_h     Lepton frame
 *   out     packed RGB888 (out_w*out_h*3), or RGB565 uint16 if cfg.out_rgb565
 */
void fusion_process(fusion_t *f, const uint8_t *y, const uint8_t *thermal, uint8_t *out);

/* ------------------------------------------------------------------ radiometry
 *
 * The picture is the display; this is the measurement. The two are deliberately
 * separate paths onto the same frame:
 *
 *   display     t_prep -> warp -> AGC -> guided filter -> palette + detail
 *   measurement t_prep -> warp -> temperature
 *
 * The measurement path stops before the AGC, which is scene-relative and would
 * make a reading depend on what else is in shot, and before the guided filter,
 * which deliberately borrows the visible camera's edges - excellent for a sharp
 * picture, and not something to quote a number off. So the number a user reads
 * is not the pixel they see; it is the thermal pixel behind it.
 *
 * All temperatures are milli-Celsius integers. The whole path stays integer,
 * including the fourth root the emissivity correction needs (see fusion.c), so
 * radiometry is available on the target build as well as the host.
 */

typedef struct {
    int32_t  milli_c;      /* corrected for emissivity and reflected background */
    int32_t  raw_milli_c;  /* what the sensor reports, i.e. assuming eps = 1.0 */
    uint16_t code_q8;      /* the 8-bit thermal code it came from, Q8 */
    int16_t  th_x, th_y;   /* thermal pixel sampled, whole part */
    uint8_t  valid;        /* 0: this output pixel has no thermal coverage */
    uint8_t  repaired;     /* 1: sampled from a row fusion_thermal_prep rebuilt,
                              i.e. interpolated, not measured. Do not quote it. */
} fusion_temp_t;

typedef struct {
    int32_t min_milli_c, max_milli_c, mean_milli_c;
    int     min_x, min_y;  /* output coords of the coldest sample */
    int     max_x, max_y;  /* output coords of the hottest */
    int     samples;       /* covered samples that went into the statistics */
    int     repaired;      /* how many of them came off reconstructed rows */
} fusion_region_t;

/*
 * The sensor's own range, in milli-Celsius. IOCTL_LEPTON_SET_RANGE(tmin, tmax)
 * maps [tmin,tmax] onto codes 0..255, so this is the one fact that turns a code
 * back into a temperature - and capture.py re-picks the range per session to buy
 * resolution (0.588 -> 0.141 C/LSB measured). Whoever sets the range on the
 * sensor must set it here, or every reading is wrong by the ratio of the two
 * spans while looking entirely plausible.
 *
 * fusion_init() defaults to -10..140C, matching capture.py's own default.
 */
void fusion_set_range(fusion_t *f, int32_t tmin_mc, int32_t tmax_mc);

/*
 * Emissivity and the reflected background temperature.
 *
 * The Lepton reports the temperature a blackbody would need to be to emit what
 * it saw. A real surface emits less and reflects the rest, so what arrives is
 *
 *     L_measured = eps * L(T_object) + (1 - eps) * L(T_reflected)
 *
 * and the sensor's answer is somewhere between the object and the room. For an
 * electrical panel this is the dominant error, not the sensor's +-5C spec:
 * bright metal at eps ~= 0.1 next to a 20C wall reads tens of degrees cold, and
 * the reading gets *better*-looking as the fault gets worse. Painted or oxidised
 * surfaces are eps ~= 0.95 and need almost no correction; the trap is exactly
 * the shiny busbar you most want to measure.
 *
 * eps_q10 is Q10 (1024 = 1.0), clamped to [0.05, 1.0]. Below 0.05 the inversion
 * divides by almost nothing and amplifies sensor noise past any usefulness -
 * such a surface must be measured off a taped-on emissivity patch instead.
 *
 * refl_mc is the temperature of whatever the surface is reflecting, usually
 * ambient. Defaults: eps = 1.0, refl = 20C, i.e. no correction at all.
 */
void fusion_set_emissivity(fusion_t *f, int32_t eps_q10, int32_t refl_mc);

/*
 * Temperature at one output pixel. Returns 0 on success, -1 if no frame has been
 * processed yet or the pixel is outside the output.
 *
 * A pixel with no thermal coverage is not an error: out->valid is 0 and the
 * temperatures are left at 0. Check valid, not the return code.
 */
int fusion_temp_at(const fusion_t *f, int ox, int oy, fusion_temp_t *out);

/*
 * Statistics over an output-coordinate rectangle [x0,x1) x [y0,y1).
 *
 * Sampled on the low-res grid rather than per output pixel: the thermal data has
 * one value per 4x4 output block at best, so a per-pixel sweep would cost 16x
 * more to report the same numbers with a falsely precise hotspot location.
 *
 * Returns 0, or -1 if nothing in the rectangle has thermal coverage.
 *
 * The delta is the point. Absolute accuracy is +-5C or +-5%, but the difference
 * between two pixels of the same material in the same frame is far better than
 * that - "this terminal is 30C above its neighbours" is a finding; "this
 * terminal is 71C" is a number with a +-5C tail on it.
 */
int fusion_temp_region(const fusion_t *f, int x0, int y0, int x1, int y1,
                       fusion_region_t *out);

/* Stage hooks, exposed for tests and for measuring where the time goes. */
void fusion_decimate(const uint8_t *src, int sw, int sh, uint8_t *dst, int dw, int dh);
void fusion_warp_thermal(const fusion_t *f, const uint8_t *thermal);
void fusion_guided(fusion_t *f);
void fusion_agc(fusion_t *f);   /* stretches t_reg in place; no-op if disabled */
int32_t fusion_code_to_milli_c(const fusion_t *f, uint16_t code_q8);  /* raw, eps=1 */
int32_t fusion_apply_emissivity(const fusion_t *f, int32_t raw_mc);
void fusion_thermal_prep(fusion_t *f, const uint8_t *src, uint8_t *dst);
void fusion_box_u8(const uint8_t *src, uint8_t *dst, int w, int h, int r, int32_t *scratch);

#ifdef __cplusplus
}
#endif

#endif /* FUSION_H */

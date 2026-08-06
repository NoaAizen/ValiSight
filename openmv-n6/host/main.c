/*
 * Host harness for fusion.c.
 *
 * Reads a recorded (or synthetic) frame pair, runs the fusion pipeline and
 * writes a PPM. This is where the algorithm gets tuned - a rebuild-and-look
 * cycle here is under a second, against minutes for a firmware flash.
 *
 *   ./fuse y.raw thermal.raw out.ppm [options]
 */
#include "fusion.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void die(const char *msg)
{
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

static uint8_t *load(const char *path, size_t want)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "error: cannot open %s\n", path);
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    long have = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if ((size_t)have != want) {
        fprintf(stderr, "error: %s is %ld bytes, expected %zu\n", path, have, want);
        exit(1);
    }
    uint8_t *buf = malloc(want);
    if (!buf || fread(buf, 1, want, fp) != want) die("read failed");
    fclose(fp);
    return buf;
}

static void write_ppm(const char *path, const uint8_t *rgb, int w, int h)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) die("cannot write output");
    fprintf(fp, "P6\n%d %d\n255\n", w, h);
    fwrite(rgb, 1, (size_t)w * h * 3, fp);
    fclose(fp);
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void usage(void)
{
    printf("usage: fuse <y.raw> <thermal.raw> <out.ppm> [options]\n"
           "  --size WxH        fused output size      (default 640x400)\n"
           "  --low WxH         working grid           (default size/4)\n"
           "  --thermal WxH     thermal source size    (default 160x120)\n"
           "  --H a,b,c,...     3x3 homography, low-res coords -> thermal coords\n"
           "  --warp FILE       precomputed warp LUT (uint16 x,y in Q8) from calibration\n"
           "  --gain N          detail injection gain, Q8 (default 200)\n"
           "  --eps N           guided-filter regularisation (default 200)\n"
           "  --radius N        guided-filter radius, 1..8 (default 4)\n"
           "  --detail N        detail high-pass radius (default 2)\n"
           "  --palette NAME    ironbow | white | black | gray (default ironbow)\n"
           "                    white/black are the mono ramps with detail headroom;\n"
           "                    black also flips the detail layer unless --detail-sign says otherwise\n"
           "  --detail-sign S   +1 or -1, overrides the sign --palette black implies\n"
           "  --agc N           scene AGC, per-mille clipped each end (e.g. 20 = 2%%). 0 = off\n"
           "                    makes tone scene-relative rather than absolute\n"
           "  --gray            alias for --palette gray\n"
           "  --no-uncovered    colorise everywhere instead of showing plain luma\n"
           "  --bench N         run N times and report per-frame ms\n"
           "\n"
           " thermal prep:\n"
           "  --deadrow F,L     rebuild rows flat to within F codes and L above the frame\n"
           "                    median - the sensor's 14 dead rows (default 24,64)\n"
           "  --no-deadrow      leave the dead rows in\n"
           "\n"
           " radiometry - the numbers, as opposed to the picture:\n"
           "  --range MIN,MAX   sensor range in C that codes 0..255 span (default -10,140)\n"
           "                    MUST match IOCTL_LEPTON_SET_RANGE or every reading is wrong\n"
           "  --emissivity E    surface emissivity 0.05..1.0 (default 1.0, i.e. no correction)\n"
           "  --reflected T     reflected background temperature in C (default 20)\n"
           "  --probe X,Y       print the temperature at one output pixel\n"
           "  --stats           print min/max/mean over the whole covered frame\n");
    exit(0);
}

static int parse_wh(const char *s, int *w, int *h)
{
    return sscanf(s, "%dx%d", w, h) == 2;
}

int main(int argc, char **argv)
{
    if (argc < 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) usage();
    if (argc < 4) die("need <y.raw> <thermal.raw> <out.ppm>");

    const char *y_path = argv[1], *t_path = argv[2], *out_path = argv[3];
    const char *warp_path = NULL;

    fusion_cfg_t cfg;
    fusion_default_cfg(&cfg);

    int low_set = 0, bench = 0, have_H = 0;
    int sign_set = 0;
    const uint8_t (*palette)[3] = NULL;
    double H[9];

    int32_t tmin_mc = -10000, tmax_mc = 140000;
    int32_t eps_q10 = 1024, refl_mc = 20000;
    int probe_x = -1, probe_y = -1, want_stats = 0;

    for (int i = 4; i < argc; i++) {
        const char *a = argv[i];
        int last = (i + 1 >= argc);

        if (!strcmp(a, "--size") && !last) {
            if (!parse_wh(argv[++i], &cfg.out_w, &cfg.out_h)) die("bad --size");
        } else if (!strcmp(a, "--low") && !last) {
            if (!parse_wh(argv[++i], &cfg.low_w, &cfg.low_h)) die("bad --low");
            low_set = 1;
        } else if (!strcmp(a, "--thermal") && !last) {
            if (!parse_wh(argv[++i], &cfg.th_w, &cfg.th_h)) die("bad --thermal");
        } else if (!strcmp(a, "--H") && !last) {
            const char *p = argv[++i];
            for (int k = 0; k < 9; k++) {
                char *end;
                H[k] = strtod(p, &end);
                if (end == p) die("bad --H (need 9 comma-separated values)");
                p = (*end == ',') ? end + 1 : end;
            }
            have_H = 1;
        } else if (!strcmp(a, "--warp") && !last) {
            warp_path = argv[++i];
        } else if (!strcmp(a, "--gain") && !last) {
            cfg.detail_gain = atoi(argv[++i]);
        } else if (!strcmp(a, "--eps") && !last) {
            cfg.gf_eps = atoi(argv[++i]);
        } else if (!strcmp(a, "--radius") && !last) {
            cfg.gf_radius = atoi(argv[++i]);
        } else if (!strcmp(a, "--detail") && !last) {
            cfg.detail_radius = atoi(argv[++i]);
        } else if (!strcmp(a, "--agc") && !last) {
            cfg.agc_permille = atoi(argv[++i]);
        } else if (!strcmp(a, "--palette") && !last) {
            const char *p = argv[++i];
            if (!strcmp(p, "ironbow")) {
                palette = fusion_ironbow;
            } else if (!strcmp(p, "white") || !strcmp(p, "whitehot")) {
                palette = fusion_whitehot;
            } else if (!strcmp(p, "black") || !strcmp(p, "blackhot")) {
                palette = fusion_blackhot;
                cfg.detail_invert = 1;      /* --detail-sign can still override */
            } else if (!strcmp(p, "gray") || !strcmp(p, "grey")) {
                palette = fusion_grayscale;
            } else {
                die("bad --palette (ironbow|white|black|gray)");
            }
        } else if (!strcmp(a, "--detail-sign") && !last) {
            int s = atoi(argv[++i]);
            if (s != 1 && s != -1) die("bad --detail-sign (+1 or -1)");
            sign_set = (s < 0) ? -1 : 1;    /* applied after the loop, so the
                                               option order does not matter */
        } else if (!strcmp(a, "--deadrow") && !last) {
            if (sscanf(argv[++i], "%d,%d", &cfg.deadrow_flat, &cfg.deadrow_lift) != 2)
                die("bad --deadrow (need FLAT,LIFT in codes)");
        } else if (!strcmp(a, "--no-deadrow")) {
            cfg.deadrow_flat = 0;
        } else if (!strcmp(a, "--range") && !last) {
            double lo, hi;
            if (sscanf(argv[++i], "%lf,%lf", &lo, &hi) != 2 || hi <= lo)
                die("bad --range (need MIN,MAX in C with MAX > MIN)");
            tmin_mc = (int32_t)(lo * 1000.0);
            tmax_mc = (int32_t)(hi * 1000.0);
        } else if (!strcmp(a, "--emissivity") && !last) {
            double e = atof(argv[++i]);
            if (e < 0.05 || e > 1.0) die("bad --emissivity (0.05..1.0)");
            eps_q10 = (int32_t)(e * 1024.0 + 0.5);
        } else if (!strcmp(a, "--reflected") && !last) {
            refl_mc = (int32_t)(atof(argv[++i]) * 1000.0);
        } else if (!strcmp(a, "--probe") && !last) {
            if (sscanf(argv[++i], "%d,%d", &probe_x, &probe_y) != 2)
                die("bad --probe (need X,Y in output pixels)");
        } else if (!strcmp(a, "--stats")) {
            want_stats = 1;
        } else if (!strcmp(a, "--gray")) {
            palette = fusion_grayscale;
        } else if (!strcmp(a, "--no-uncovered")) {
            cfg.show_uncovered = 0;
        } else if (!strcmp(a, "--bench") && !last) {
            bench = atoi(argv[++i]);
        } else {
            fprintf(stderr, "error: unknown option %s\n", a);
            return 1;
        }
    }

    if (!low_set) {
        cfg.low_w = cfg.out_w / 4;
        cfg.low_h = cfg.out_h / 4;
    }
    if (sign_set) cfg.detail_invert = (sign_set < 0);

    fusion_t f;
    if (fusion_init(&f, &cfg) != 0)
        die("fusion_init failed (check --radius 1..8 and that size divides evenly by low)");
    /* the palette tables only exist after fusion_init() has built them */
    if (palette) fusion_set_palette(&f, palette);
    fusion_set_range(&f, tmin_mc, tmax_mc);
    fusion_set_emissivity(&f, eps_q10, refl_mc);

    if (warp_path) {
        size_t n = sizeof(uint16_t) * 2 * (size_t)cfg.low_w * cfg.low_h;
        uint8_t *lut = load(warp_path, n);
        fusion_set_warp(&f, (const uint16_t *)lut);
        free(lut);
    } else {
        if (!have_H) {
            /* Default: stretch the thermal frame across the whole output. Real
               work uses --warp from the calibration; this only exists so the
               pipeline can be exercised before a calibration exists. */
            memset(H, 0, sizeof(H));
            H[0] = (double)(cfg.th_w - 1) / (cfg.low_w - 1);
            H[4] = (double)(cfg.th_h - 1) / (cfg.low_h - 1);
            H[8] = 1.0;
        }
        fusion_set_homography(&f, H);
    }

    uint8_t *y = load(y_path, (size_t)cfg.out_w * cfg.out_h);
    uint8_t *t = load(t_path, (size_t)cfg.th_w * cfg.th_h);
    uint8_t *out = malloc((size_t)cfg.out_w * cfg.out_h * 3);
    if (!out) die("out of memory");

    int reps = bench > 0 ? bench : 1;
    double t0 = now_ms();
    for (int i = 0; i < reps; i++)
        fusion_process(&f, y, t, out);
    double el = now_ms() - t0;

    write_ppm(out_path, out, cfg.out_w, cfg.out_h);

    printf("%dx%d out, %dx%d grid, %dx%d thermal | r=%d eps=%d gain=%d detail=%d\n",
           cfg.out_w, cfg.out_h, cfg.low_w, cfg.low_h, cfg.th_w, cfg.th_h,
           cfg.gf_radius, cfg.gf_eps, cfg.detail_gain, cfg.detail_radius);
    printf("%.2f ms/frame over %d rep(s) -> %s\n", el / reps, reps, out_path);

    if (probe_x >= 0) {
        fusion_temp_t tp;
        if (fusion_temp_at(&f, probe_x, probe_y, &tp) != 0) {
            printf("probe %d,%d: outside the output frame\n", probe_x, probe_y);
        } else if (!tp.valid) {
            printf("probe %d,%d: no thermal coverage\n", probe_x, probe_y);
        } else {
            printf("probe %d,%d: %.2f C (raw %.2f C, code %.2f, thermal px %d,%d)%s\n",
                   probe_x, probe_y, tp.milli_c / 1000.0, tp.raw_milli_c / 1000.0,
                   tp.code_q8 / 256.0, tp.th_x, tp.th_y,
                   tp.repaired ? "  RECONSTRUCTED ROW - not a measurement" : "");
        }
    }

    if (want_stats) {
        fusion_region_t rg;
        if (fusion_temp_region(&f, 0, 0, cfg.out_w, cfg.out_h, &rg) != 0) {
            printf("stats: nothing in frame has thermal coverage\n");
        } else {
            printf("stats: min %.2f C at %d,%d | max %.2f C at %d,%d | mean %.2f C"
                   " | delta %.2f C | %d samples",
                   rg.min_milli_c / 1000.0, rg.min_x, rg.min_y,
                   rg.max_milli_c / 1000.0, rg.max_x, rg.max_y,
                   rg.mean_milli_c / 1000.0,
                   (rg.max_milli_c - rg.min_milli_c) / 1000.0, rg.samples);
            if (rg.repaired) printf(" (%d off rebuilt rows)", rg.repaired);
            printf("\n");
        }
        printf("deadrows: %d rebuilt\n", f.rows_rebuilt);
    }

    free(y);
    free(t);
    free(out);
    fusion_free(&f);
    return 0;
}

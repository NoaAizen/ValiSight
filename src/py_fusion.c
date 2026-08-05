/*
 * MicroPython binding for the thermal/visible fusion pipeline.
 *
 * Deliberately thin: every decision lives in fusion.c, which knows nothing about
 * MicroPython and is iterated on a workstation. This file only marshals objects.
 *
 *   import fusion
 *   f = fusion.Fusion(warp="/sdcard/warp.lut")
 *   img = f.process(rgb_snapshot, thermal_snapshot)   # RGB565 image
 *   print(f.bench(rgb_snapshot, thermal_snapshot, n=10), "ms")
 *
 *   f.set_range(TMIN, TMAX)          # whatever IOCTL_LEPTON_SET_RANGE was given
 *   f.set_emissivity(0.95, 22)       # surface, and what it reflects
 *   f.temp_at(320, 200)              # (C, raw_C, repaired) or None
 *   f.temp_region()                  # (min, max, mean, hot_x, hot_y) or None
 */
#if MICROPY_PY_FUSION

#include <string.h>

#include "py/runtime.h"
#include "py/obj.h"
#include "py/mphal.h"

#include "imlib.h"
#include "py_image.h"
#include "py_helper.h"

#include "umalloc.h"
#include "fusion.h"

typedef struct _py_fusion_obj_t {
    mp_obj_base_t base;
    fusion_t f;
    image_t out;
    bool ready;
} py_fusion_obj_t;

const mp_obj_type_t py_fusion_type;

static void check_ready(py_fusion_obj_t *self) {
    if (!self->ready) {
        mp_raise_msg(&mp_type_RuntimeError, MP_ERROR_TEXT("fusion object is not initialised"));
    }
}

// Load a warp LUT from a path or any buffer-like object. The table is what the
// calibration produces; without one the object falls back to stretching the
// thermal frame across the output, which is only useful for smoke tests.
static void load_warp(py_fusion_obj_t *self, mp_obj_t src) {
    const size_t want = sizeof(uint16_t) * 2 *
                        (size_t) self->f.cfg.low_w * self->f.cfg.low_h;

    if (mp_obj_is_str(src)) {
        mp_obj_t open_args[2] = {src, MP_OBJ_NEW_QSTR(MP_QSTR_rb)};
        mp_obj_t file = mp_call_function_n_kw(
            mp_load_attr(mp_import_name(MP_QSTR_builtins, mp_const_none, MP_OBJ_NEW_SMALL_INT(0)),
                         MP_QSTR_open), 2, 0, open_args);
        mp_obj_t data = mp_call_function_0(mp_load_attr(file, MP_QSTR_read));
        mp_call_function_0(mp_load_attr(file, MP_QSTR_close));
        src = data;
    }

    mp_buffer_info_t bufinfo;
    mp_get_buffer_raise(src, &bufinfo, MP_BUFFER_READ);
    if (bufinfo.len != want) {
        mp_raise_msg_varg(&mp_type_ValueError,
                          MP_ERROR_TEXT("warp LUT is %d bytes, expected %d"),
                          (int) bufinfo.len, (int) want);
    }
    fusion_set_warp(&self->f, (const uint16_t *) bufinfo.buf);
}

static mp_obj_t py_fusion_make_new(const mp_obj_type_t *type, size_t n_args,
                                   size_t n_kw, const mp_obj_t *all_args) {
    enum {
        ARG_width, ARG_height, ARG_thermal_width, ARG_thermal_height,
        ARG_radius, ARG_eps, ARG_gain, ARG_detail, ARG_uncovered, ARG_warp
    };
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_width,          MP_ARG_INT, {.u_int = 640} },
        { MP_QSTR_height,         MP_ARG_INT, {.u_int = 400} },
        { MP_QSTR_thermal_width,  MP_ARG_INT, {.u_int = 160} },
        { MP_QSTR_thermal_height, MP_ARG_INT, {.u_int = 120} },
        { MP_QSTR_radius,         MP_ARG_INT, {.u_int = 4} },
        { MP_QSTR_eps,            MP_ARG_INT, {.u_int = 200} },
        { MP_QSTR_gain,           MP_ARG_INT, {.u_int = 200} },
        { MP_QSTR_detail,         MP_ARG_INT, {.u_int = 2} },
        { MP_QSTR_uncovered,      MP_ARG_BOOL, {.u_bool = true} },
        { MP_QSTR_warp,           MP_ARG_OBJ, {.u_obj = mp_const_none} },
    };

    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all_kw_array(n_args, n_kw, all_args, MP_ARRAY_SIZE(allowed_args),
                              allowed_args, args);

    py_fusion_obj_t *self = mp_obj_malloc_with_finaliser(py_fusion_obj_t, &py_fusion_type);
    self->ready = false;

    fusion_cfg_t cfg;
    fusion_default_cfg(&cfg);
    cfg.out_w = args[ARG_width].u_int;
    cfg.out_h = args[ARG_height].u_int;
    cfg.low_w = cfg.out_w / 4;
    cfg.low_h = cfg.out_h / 4;
    cfg.th_w = args[ARG_thermal_width].u_int;
    cfg.th_h = args[ARG_thermal_height].u_int;
    cfg.gf_radius = args[ARG_radius].u_int;
    cfg.gf_eps = args[ARG_eps].u_int;
    cfg.detail_gain = args[ARG_gain].u_int;
    cfg.detail_radius = args[ARG_detail].u_int;
    cfg.show_uncovered = args[ARG_uncovered].u_bool;
    cfg.out_rgb565 = 1;

    if (fusion_init(&self->f, &cfg) != 0) {
        mp_raise_msg(&mp_type_ValueError,
                     MP_ERROR_TEXT("bad fusion config (radius must be 1..8, size divisible by 4)"));
    }
    self->ready = true;

    self->out.w = cfg.out_w;
    self->out.h = cfg.out_h;
    self->out.pixfmt = PIXFORMAT_RGB565;
    // the output frame is written every pass and then blitted, so it wants fast RAM too
    self->out.data = uma_calloc((size_t) cfg.out_w * cfg.out_h * sizeof(uint16_t),
                                UMA_FAST | UMA_PERSIST | UMA_MAYBE);

    if (args[ARG_warp].u_obj != mp_const_none) {
        load_warp(self, args[ARG_warp].u_obj);
    }
    return MP_OBJ_FROM_PTR(self);
}

static void check_inputs(py_fusion_obj_t *self, image_t *rgb, image_t *thermal) {
    if (rgb->w != self->f.cfg.out_w || rgb->h != self->f.cfg.out_h) {
        mp_raise_msg_varg(&mp_type_ValueError,
                          MP_ERROR_TEXT("visible frame is %dx%d, expected %dx%d"),
                          rgb->w, rgb->h, self->f.cfg.out_w, self->f.cfg.out_h);
    }
    if (thermal->w != self->f.cfg.th_w || thermal->h != self->f.cfg.th_h) {
        mp_raise_msg_varg(&mp_type_ValueError,
                          MP_ERROR_TEXT("thermal frame is %dx%d, expected %dx%d"),
                          thermal->w, thermal->h, self->f.cfg.th_w, self->f.cfg.th_h);
    }
    // Both must be 8bpp: the pipeline wants luma, and the visible camera is put
    // in GRAYSCALE anyway - fusion only ever needs the guide channel from it.
    if (rgb->pixfmt != PIXFORMAT_GRAYSCALE || thermal->pixfmt != PIXFORMAT_GRAYSCALE) {
        mp_raise_msg(&mp_type_ValueError,
                     MP_ERROR_TEXT("both frames must be GRAYSCALE"));
    }
}

static mp_obj_t py_fusion_process(mp_obj_t self_in, mp_obj_t rgb_in, mp_obj_t thermal_in) {
    py_fusion_obj_t *self = MP_OBJ_TO_PTR(self_in);
    check_ready(self);

    image_t *rgb = py_helper_arg_to_image(rgb_in, 0);
    image_t *thermal = py_helper_arg_to_image(thermal_in, 0);
    check_inputs(self, rgb, thermal);

    fusion_process(&self->f, rgb->data, thermal->data, self->out.data);
    return py_image_from_struct(&self->out);
}
static MP_DEFINE_CONST_FUN_OBJ_3(py_fusion_process_obj, py_fusion_process);

static mp_obj_t py_fusion_bench(size_t n_args, const mp_obj_t *pos_args, mp_map_t *kw_args) {
    static const mp_arg_t allowed_args[] = {
        { MP_QSTR_n, MP_ARG_INT, {.u_int = 10} },
    };
    mp_arg_val_t args[MP_ARRAY_SIZE(allowed_args)];
    mp_arg_parse_all(n_args - 3, pos_args + 3, kw_args, MP_ARRAY_SIZE(allowed_args),
                     allowed_args, args);

    py_fusion_obj_t *self = MP_OBJ_TO_PTR(pos_args[0]);
    check_ready(self);

    image_t *rgb = py_helper_arg_to_image(pos_args[1], 0);
    image_t *thermal = py_helper_arg_to_image(pos_args[2], 0);
    check_inputs(self, rgb, thermal);

    int reps = args[0].u_int < 1 ? 1 : args[0].u_int;
    mp_uint_t t0 = mp_hal_ticks_us();
    for (int i = 0; i < reps; i++) {
        fusion_process(&self->f, rgb->data, thermal->data, self->out.data);
    }
    mp_uint_t el = mp_hal_ticks_us() - t0;
    return mp_obj_new_float((mp_float_t) el / 1000.0f / (mp_float_t) reps);
}
static MP_DEFINE_CONST_FUN_OBJ_KW(py_fusion_bench_obj, 3, py_fusion_bench);

static mp_obj_t py_fusion_set_warp_m(mp_obj_t self_in, mp_obj_t src) {
    py_fusion_obj_t *self = MP_OBJ_TO_PTR(self_in);
    check_ready(self);
    load_warp(self, src);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(py_fusion_set_warp_obj, py_fusion_set_warp_m);

static mp_obj_t py_fusion_set_gain(mp_obj_t self_in, mp_obj_t gain) {
    py_fusion_obj_t *self = MP_OBJ_TO_PTR(self_in);
    check_ready(self);
    self->f.cfg.detail_gain = mp_obj_get_int(gain);
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(py_fusion_set_gain_obj, py_fusion_set_gain);

// ---------------------------------------------------------------- radiometry
//
// The picture is the display, these are the measurement. See the radiometry
// section of fusion.h for why the two are separate paths over the same frame.

// The sensor range that codes 0..255 span. This has to follow whatever was
// passed to IOCTL_LEPTON_SET_RANGE - auto-range re-picks it per session, and it
// is the only thing that turns a code back into a temperature.
static mp_obj_t py_fusion_set_range(mp_obj_t self_in, mp_obj_t lo, mp_obj_t hi) {
    py_fusion_obj_t *self = MP_OBJ_TO_PTR(self_in);
    check_ready(self);
    fusion_set_range(&self->f,
                     (int32_t) (mp_obj_get_float(lo) * 1000.0f),
                     (int32_t) (mp_obj_get_float(hi) * 1000.0f));
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_3(py_fusion_set_range_obj, py_fusion_set_range);

// Emissivity 0.05..1.0 and the temperature the surface reflects (usually
// ambient). 1.0 is what the sensor already assumes; bright metal is ~0.1 and
// reads tens of degrees cold without this.
static mp_obj_t py_fusion_set_emissivity(size_t n_args, const mp_obj_t *args) {
    py_fusion_obj_t *self = MP_OBJ_TO_PTR(args[0]);
    check_ready(self);

    mp_float_t refl = (n_args > 2) ? mp_obj_get_float(args[2]) : (mp_float_t) 20.0f;
    fusion_set_emissivity(&self->f,
                          (int32_t) (mp_obj_get_float(args[1]) * 1024.0f + 0.5f),
                          (int32_t) (refl * 1000.0f));
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(py_fusion_set_emissivity_obj, 2, 3,
                                           py_fusion_set_emissivity);

// Temperature at one output pixel, as (celsius, raw_celsius, repaired), or None
// where the thermal camera does not see. raw_celsius is before the emissivity
// correction; repaired means the sample came off a row the library rebuilt and
// is interpolated rather than measured.
static mp_obj_t py_fusion_temp_at(mp_obj_t self_in, mp_obj_t x_in, mp_obj_t y_in) {
    py_fusion_obj_t *self = MP_OBJ_TO_PTR(self_in);
    check_ready(self);

    fusion_temp_t t;
    if (fusion_temp_at(&self->f, mp_obj_get_int(x_in), mp_obj_get_int(y_in), &t) != 0) {
        mp_raise_msg(&mp_type_ValueError,
                     MP_ERROR_TEXT("no frame processed yet, or pixel is outside the frame"));
    }
    if (!t.valid) {
        return mp_const_none;
    }

    mp_obj_t items[3] = {
        mp_obj_new_float((mp_float_t) t.milli_c / 1000.0f),
        mp_obj_new_float((mp_float_t) t.raw_milli_c / 1000.0f),
        mp_obj_new_bool(t.repaired),
    };
    return mp_obj_new_tuple(3, items);
}
static MP_DEFINE_CONST_FUN_OBJ_3(py_fusion_temp_at_obj, py_fusion_temp_at);

// Statistics over an output rectangle, defaulting to the whole frame, as
// (min, max, mean, hot_x, hot_y) or None where nothing is covered.
//
// The delta is what an inspection rests on: absolute accuracy is +-5C, but the
// difference between two points of the same material in one frame is far better
// than that.
static mp_obj_t py_fusion_temp_region(size_t n_args, const mp_obj_t *args) {
    py_fusion_obj_t *self = MP_OBJ_TO_PTR(args[0]);
    check_ready(self);

    int x0 = 0, y0 = 0, x1 = self->f.cfg.out_w, y1 = self->f.cfg.out_h;
    if (n_args >= 5) {
        x0 = mp_obj_get_int(args[1]);
        y0 = mp_obj_get_int(args[2]);
        x1 = mp_obj_get_int(args[3]);
        y1 = mp_obj_get_int(args[4]);
    }

    fusion_region_t r;
    if (fusion_temp_region(&self->f, x0, y0, x1, y1, &r) != 0) {
        return mp_const_none;
    }

    mp_obj_t items[5] = {
        mp_obj_new_float((mp_float_t) r.min_milli_c / 1000.0f),
        mp_obj_new_float((mp_float_t) r.max_milli_c / 1000.0f),
        mp_obj_new_float((mp_float_t) r.mean_milli_c / 1000.0f),
        MP_OBJ_NEW_SMALL_INT(r.max_x),
        MP_OBJ_NEW_SMALL_INT(r.max_y),
    };
    return mp_obj_new_tuple(5, items);
}
static MP_DEFINE_CONST_FUN_OBJ_VAR_BETWEEN(py_fusion_temp_region_obj, 1, 5,
                                           py_fusion_temp_region);

// How many rows the dead-row repair rebuilt in the last frame. Worth surfacing:
// on this unit it is 0 or 14 with nothing in between, and a change means the
// sensor's state changed, not the scene.
static mp_obj_t py_fusion_rows_rebuilt(mp_obj_t self_in) {
    py_fusion_obj_t *self = MP_OBJ_TO_PTR(self_in);
    check_ready(self);
    return MP_OBJ_NEW_SMALL_INT(self->f.rows_rebuilt);
}
static MP_DEFINE_CONST_FUN_OBJ_1(py_fusion_rows_rebuilt_obj, py_fusion_rows_rebuilt);

// "ironbow" | "white" | "black" | "gray". The mono ramps are the ones to use
// when the aim is a sharp picture rather than a legible temperature scale;
// black-hot also flips the detail sign, or the embossed texture fights the tone.
static mp_obj_t py_fusion_set_palette_m(mp_obj_t self_in, mp_obj_t name_in) {
    py_fusion_obj_t *self = MP_OBJ_TO_PTR(self_in);
    check_ready(self);

    const char *name = mp_obj_str_get_str(name_in);
    if (!strcmp(name, "ironbow")) {
        fusion_set_palette(&self->f, fusion_ironbow);
        self->f.cfg.detail_invert = 0;
    } else if (!strcmp(name, "white")) {
        fusion_set_palette(&self->f, fusion_whitehot);
        self->f.cfg.detail_invert = 0;
    } else if (!strcmp(name, "black")) {
        fusion_set_palette(&self->f, fusion_blackhot);
        self->f.cfg.detail_invert = 1;
    } else if (!strcmp(name, "gray")) {
        fusion_set_palette(&self->f, fusion_grayscale);
        self->f.cfg.detail_invert = 0;
    } else {
        mp_raise_msg(&mp_type_ValueError,
                     MP_ERROR_TEXT("palette must be ironbow, white, black or gray"));
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_2(py_fusion_set_palette_obj, py_fusion_set_palette_m);

static mp_obj_t py_fusion_del(mp_obj_t self_in) {
    py_fusion_obj_t *self = MP_OBJ_TO_PTR(self_in);
    if (self->ready) {
        fusion_free(&self->f);
        if (self->out.data) {
            uma_free(self->out.data);
            self->out.data = NULL;
        }
        self->ready = false;
    }
    return mp_const_none;
}
static MP_DEFINE_CONST_FUN_OBJ_1(py_fusion_del_obj, py_fusion_del);

static const mp_rom_map_elem_t py_fusion_locals_dict_table[] = {
    { MP_ROM_QSTR(MP_QSTR_process),  MP_ROM_PTR(&py_fusion_process_obj) },
    { MP_ROM_QSTR(MP_QSTR_bench),    MP_ROM_PTR(&py_fusion_bench_obj) },
    { MP_ROM_QSTR(MP_QSTR_set_warp), MP_ROM_PTR(&py_fusion_set_warp_obj) },
    { MP_ROM_QSTR(MP_QSTR_set_gain), MP_ROM_PTR(&py_fusion_set_gain_obj) },
    { MP_ROM_QSTR(MP_QSTR_set_palette), MP_ROM_PTR(&py_fusion_set_palette_obj) },

    // radiometry - the numbers, as opposed to the picture
    { MP_ROM_QSTR(MP_QSTR_set_range),      MP_ROM_PTR(&py_fusion_set_range_obj) },
    { MP_ROM_QSTR(MP_QSTR_set_emissivity), MP_ROM_PTR(&py_fusion_set_emissivity_obj) },
    { MP_ROM_QSTR(MP_QSTR_temp_at),        MP_ROM_PTR(&py_fusion_temp_at_obj) },
    { MP_ROM_QSTR(MP_QSTR_temp_region),    MP_ROM_PTR(&py_fusion_temp_region_obj) },
    { MP_ROM_QSTR(MP_QSTR_rows_rebuilt),   MP_ROM_PTR(&py_fusion_rows_rebuilt_obj) },

    { MP_ROM_QSTR(MP_QSTR___del__),  MP_ROM_PTR(&py_fusion_del_obj) },
};
static MP_DEFINE_CONST_DICT(py_fusion_locals_dict, py_fusion_locals_dict_table);

MP_DEFINE_CONST_OBJ_TYPE(
    py_fusion_type,
    MP_QSTR_Fusion,
    MP_TYPE_FLAG_NONE,
    make_new, py_fusion_make_new,
    locals_dict, &py_fusion_locals_dict
    );

static const mp_rom_map_elem_t fusion_module_globals_table[] = {
    { MP_ROM_QSTR(MP_QSTR___name__), MP_ROM_QSTR(MP_QSTR_fusion) },
    { MP_ROM_QSTR(MP_QSTR_Fusion),   MP_ROM_PTR(&py_fusion_type) },
};
static MP_DEFINE_CONST_DICT(fusion_module_globals, fusion_module_globals_table);

const mp_obj_module_t fusion_user_cmodule = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t *) &fusion_module_globals,
};

MP_REGISTER_MODULE(MP_QSTR_fusion, fusion_user_cmodule);

#endif // MICROPY_PY_FUSION

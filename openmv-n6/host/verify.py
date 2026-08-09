#!/usr/bin/env python3
"""Assert the fusion output against the synthetic ground truth.

Checks, in order of what they would catch:

  1. uncovered region is passed through as plain luma  -> coverage mask, grid mapping
  2. the hot blob lands where the warp says it should  -> registration, warp LUT, bilinear
  3. hot reads hotter than cold on the palette         -> guided filter reconstructs levels
  4. output high-frequency tracks the luma's           -> detail injection is actually wired
  5. detail gain 0 removes that correlation            -> the coupling is the gain, not a leak
  6. dead rows are rebuilt back to the clean frame     -> saturated-row repair
  7. probed temperatures match the range mapping       -> radiometry, emissivity
"""
import json
import subprocess
import sys

import numpy as np

FAILS = []


def check(name, ok, detail=""):
    print("  %-52s %s%s" % (name, "PASS" if ok else "FAIL", "  " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


def read_ppm(path):
    with open(path, "rb") as f:
        data = f.read()
    parts, off = [], 0
    while len(parts) < 4:
        end = data.index(b"\n", off)
        line = data[off:end]
        off = end + 1
        if line.startswith(b"#"):
            continue
        parts += line.split()
    w, h = int(parts[1]), int(parts[2])
    return np.frombuffer(data[off:off + w * h * 3], np.uint8).reshape(h, w, 3)


def highpass(a):
    """Cheap 3x3 high-pass; enough to tell injected structure from none."""
    a = a.astype(np.float64)
    b = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]) / 4.0
    return a[1:-1, 1:-1] - b


def fuse(out, thermal="synth/thermal.raw", *extra):
    """Run the harness and hand back stdout, so the printed measurements can be
    asserted on directly rather than re-derived here."""
    r = subprocess.run(["./fuse", "synth/y.raw", thermal, out, "--warp", "synth/warp.lut",
                        *extra], check=True, capture_output=True, text=True)
    return r.stdout


def thermal_noise(frames=8, noise=2, nframes=80, *extra):
    """RMS of the filtered thermal plane against the clean frame, in codes, plus
    how many pixels took the motion path on the last frame.

    Measured on the thermal plane rather than the fused image on purpose: the
    guided filter averages ~(2r+1)^2 low-res samples, so by the output the
    temporal filter's contribution is buried under a spatial one and the number
    is meaningless. This is also the plane fusion_temp_at() quotes from."""
    out = fuse("synth/temporal.ppm", "synth/thermal.raw",
               "--range", "20,40", "--gain", "0",
               "--temporal", str(frames), "--noise", str(noise),
               "--frames", str(nframes), *extra)
    line = [l for l in out.splitlines() if l.startswith("thermal noise:")][0]
    rms = float(line.split()[2])
    moved = int(line.split("frame(s),")[1].split()[0])
    return rms, moved


def check_temporal():
    """The temporal filter: does it remove noise, does it keep working as it is
    asked to smooth harder, does the motion gate gate, and does it leave the
    measurement alone."""

    off, _ = thermal_noise(frames=1)
    on, moved = thermal_noise(frames=8)
    check("temporal filter cuts thermal noise", on < off / 2.0,
          "%.3f -> %.3f codes rms (%.1fx)" % (off, on, off / max(on, 1e-9)))

    # A static scene must not be reading as motion, or the filter is doing
    # nothing and the reduction above came from somewhere else.
    check("static scene does not trip the motion gate", moved < 100,
          "%d of 19200 px" % moved)

    # The regression test for the stall. An IIR held at input precision quietly
    # stops converging once the blend weight drops past ~1/4: the first version
    # measured 2.0x at 4 frames, 1.2x at 8 and exactly 1.0x at 16 and 32. The
    # bug's whole signature is non-monotonicity, so that is what is asserted -
    # a single-point check at 8 frames would have passed it.
    series = [thermal_noise(frames=n)[0] for n in (2, 4, 8, 16)]
    check("more frames means less noise, monotonically",
          all(b < a for a, b in zip(series, series[1:])),
          " -> ".join("%.3f" % v for v in series))

    # The gate is specified in milli-Celsius, so the same noise in codes must be
    # judged differently when the sensor range changes - that is the entire
    # reason it is not stated in codes. At -10..140C one code is 0.588C, so 2
    # codes of noise is a large excursion and belongs on the motion path; at
    # 20..40C the same 2 codes is 0.157C and is squarely noise.
    def moved_at(rng):
        out = fuse("synth/temporal_%s.ppm" % rng.replace(",", "_"), "synth/thermal.raw",
                   "--range", rng, "--gain", "0", "--temporal", "8",
                   "--noise", "2", "--frames", "40")
        line = [l for l in out.splitlines() if l.startswith("thermal noise:")][0]
        return int(line.split("frame(s),")[1].split()[0])

    narrow, wide = moved_at("20,40"), moved_at("-10,140")
    check("the motion gate follows the sensor range, not the codes",
          wide > 1000 and wide > 10 * max(narrow, 1),
          "%d px moved at -10..140C vs %d at 20..40C" % (wide, narrow))

    # The filter sits upstream of the radiometry, so a clean static scene must
    # come back with the same temperature filtered or not. If it does not, the
    # filter is biasing the measurement rather than denoising it.
    t_off = probe("synth/temporal_r0.ppm", "synth/thermal.raw", 400, 125,
                  "--range", "20,45", "--temporal", "1", "--frames", "20")
    t_on = probe("synth/temporal_r1.ppm", "synth/thermal.raw", 400, 125,
                 "--range", "20,45", "--temporal", "8", "--frames", "20")
    check("filter does not shift a clean static reading",
          abs(t_off["c"] - t_on["c"]) < 0.05,
          "%.3f vs %.3f C" % (t_off["c"], t_on["c"]))


def probe(out, thermal, x, y, *extra):
    line = [l for l in fuse(out, thermal, "--probe", "%d,%d" % (x, y), *extra).splitlines()
            if l.startswith("probe")][0]
    if "no thermal coverage" in line:
        return None
    return {"c": float(line.split(": ")[1].split()[0]),
            "raw": float(line.split("raw ")[1].split()[0]),
            "repaired": "RECONSTRUCTED" in line}


def stats(out, thermal, *extra):
    line = [l for l in fuse(out, thermal, "--stats", *extra).splitlines()
            if l.startswith("stats")][0]
    f = lambda k: float(line.split(k + " ")[1].split()[0])  # noqa: E731
    return {"min": f("min"), "max": f("max"), "mean": f("mean"), "delta": f("delta")}


def check_row_repair(meta):
    """The defect is a real one on this unit: see DEAD_ROWS in synth.py, which
    builds the two damaged thermal frames these checks run against."""
    fuse("synth/clean.ppm", "synth/thermal.raw")
    fuse("synth/dead_on.ppm", "synth/thermal_dead.raw")
    fuse("synth/dead_off.ppm", "synth/thermal_dead.raw", "--no-deadrow")

    clean = read_ppm("synth/clean.ppm").astype(int)
    on = read_ppm("synth/dead_on.ppm").astype(int)
    off = read_ppm("synth/dead_off.ppm").astype(int)

    err_on, err_off = np.abs(on - clean).mean(), np.abs(off - clean).mean()
    check("dead rows are repaired back to the clean frame", err_on < 1.0,
          "mean px error %.2f" % err_on)

    # Without this the test above would pass on a pipeline that ignored the rows
    # entirely, if the rows happened not to matter much. They matter enormously.
    check("...and the damage they do is real", err_off > 20.0,
          "mean px error unrepaired %.1f" % err_off)

    # The rows do not merely add hot stripes. Each one that lands on a VoSPI seam
    # is read as a ~200-code DC step and subtracted from a whole 30-row segment,
    # so an unrepaired frame reads *cold* over a quarter of itself - a stripe is
    # obvious, a segment 35C too cold is a plausible-looking wrong answer.
    s_on, s_off = stats("synth/s1.ppm", "synth/thermal_dead.raw", "--range", "-10,140"), \
        stats("synth/s2.ppm", "synth/thermal_dead.raw", "--range", "-10,140", "--no-deadrow")
    s_clean = stats("synth/s3.ppm", "synth/thermal.raw", "--range", "-10,140")
    check("repair restores the frame's temperature statistics",
          abs(s_on["mean"] - s_clean["mean"]) < 0.5 and abs(s_on["max"] - s_clean["max"]) < 0.5,
          "mean %.2f vs %.2f C" % (s_on["mean"], s_clean["mean"]))
    check("unrepaired dead rows corrupt the deband, not just the stripe",
          s_off["min"] < s_clean["min"] - 10,
          "min %.1f C vs %.1f C clean" % (s_off["min"], s_clean["min"]))

    # A rebuilt row is interpolated, not measured, and must say so.
    p = probe("synth/p1.ppm", "synth/thermal_dead.raw", 400, 125, "--range", "-10,140")
    check("a reading off a rebuilt row is flagged", p and p["repaired"])
    p = probe("synth/p2.ppm", "synth/thermal_dead.raw", 400, 180, "--range", "-10,140")
    check("a reading off a live row is not flagged", p and not p["repaired"])

    # Safety valve: a scene that genuinely pins the top of the range is not a
    # defect, and interpolating it away would erase the hottest target in the
    # frame - the one failure of this repair that could hide a fault.
    s_hot = stats("synth/s4.ppm", "synth/thermal_hot.raw", "--range", "-10,140")
    s_hot_off = stats("synth/s5.ppm", "synth/thermal_hot.raw", "--range", "-10,140", "--no-deadrow")
    check("a saturated scene is left alone rather than interpolated away",
          abs(s_hot["max"] - s_hot_off["max"]) < 0.01,
          "max %.1f C with repair, %.1f C without" % (s_hot["max"], s_hot_off["max"]))


def check_radiometry(meta):
    hx0, hy0, hx1, hy1 = meta["hot_output_box"]
    px, py = (hx0 + hx1) // 2, (hy0 + hy1) // 2
    hot_code = meta["codes"]["hot"]

    # 1. the code -> temperature mapping is the sensor range, nothing else
    for lo, hi in ((-10, 140), (20, 45)):
        p = probe("synth/r1.ppm", "synth/thermal.raw", px, py, "--range", "%d,%d" % (lo, hi))
        want = lo + hot_code * (hi - lo) / 255.0
        check("code %d over range %d..%dC reads %.1fC" % (hot_code, lo, hi, want),
              p is not None and abs(p["raw"] - want) < 0.7,
              "got %.2f C" % p["raw"] if p else "no coverage")

    # A narrow range is the whole reason for the sensor's auto-range: the same
    # code carries 6x less temperature, so a reading that ignored the range would
    # be out by 100C here rather than by a plausible-looking amount.
    p_wide = probe("synth/r2.ppm", "synth/thermal.raw", px, py, "--range", "-10,140")
    p_narrow = probe("synth/r3.ppm", "synth/thermal.raw", px, py, "--range", "20,45")
    check("range width actually changes the reading", p_wide["raw"] - p_narrow["raw"] > 50,
          "%.1f C vs %.1f C" % (p_wide["raw"], p_narrow["raw"]))

    # 2. emissivity correction against the float model it approximates
    for eps in (0.95, 0.5, 0.1):
        p = probe("synth/r4.ppm", "synth/thermal.raw", px, py,
                  "--range", "-10,140", "--emissivity", str(eps), "--reflected", "20")
        q = round(eps * 1024) / 1024.0          # the C side stores emissivity in Q10
        ta, tr = p["raw"] + 273.15, 20 + 273.15
        want = ((ta ** 4 - (1 - q) * tr ** 4) / q) ** 0.25 - 273.15
        check("emissivity %.2f inverts to within 0.05C of the model" % eps,
              abs(p["c"] - want) < 0.05, "%.2f C vs %.2f C" % (p["c"], want))

    p1 = probe("synth/r5.ppm", "synth/thermal.raw", px, py, "--range", "-10,140")
    check("emissivity 1.0 leaves the reading untouched", abs(p1["c"] - p1["raw"]) < 0.001)

    # 3. an uncovered pixel must report no coverage, not a fabricated temperature
    ux1 = meta["uncovered_output_box"][2]
    check("an uncovered pixel reports no temperature",
          probe("synth/r6.ppm", "synth/thermal.raw", ux1 // 2, 200, "--range", "-10,140") is None)

    # 4. region statistics find the blob the warp put there
    s = stats("synth/r7.ppm", "synth/thermal.raw", "--range", "-10,140")
    codes = meta["codes"]
    span = 150.0 / 255.0
    check("region max matches the hot code", abs(s["max"] - (-10 + codes["hot"] * span)) < 1.5,
          "%.1f C" % s["max"])
    check("region min matches the cold code", abs(s["min"] - (-10 + codes["cold"] * span)) < 1.5,
          "%.1f C" % s["min"])
    check("region delta is hot minus cold",
          abs(s["delta"] - (codes["hot"] - codes["cold"]) * span) < 2.0, "%.1f C" % s["delta"])


def main():
    meta = json.load(open("synth/meta.json"))
    ow, oh = meta["out_w"], meta["out_h"]

    y = np.fromfile("synth/y.raw", np.uint8).reshape(oh, ow)
    rgb = read_ppm("synth/out.ppm")
    print("output %dx%d" % (rgb.shape[1], rgb.shape[0]))

    # 1. uncovered strip must be exactly the luma, in grey.
    #    The last two columns before the footprint straddle it: their bilinear
    #    stencil already touches a covered grid cell, so they blend. That soft
    #    edge is deliberate - a hard cut at the coverage boundary reads worse -
    #    so the strict check stops short of it.
    ux1 = meta["uncovered_output_box"][2] - 2
    strip, ystrip = rgb[:, :ux1], y[:, :ux1]
    grey = (strip[..., 0] == strip[..., 1]).all() and (strip[..., 1] == strip[..., 2]).all()
    exact = np.array_equal(strip[..., 0], ystrip)
    check("uncovered strip is neutral grey", grey)
    check("uncovered strip equals the source luma", exact,
          "" if exact else "max diff %d" % np.abs(strip[..., 0].astype(int) - ystrip).max())

    # 2 & 3. the hot blob lands where the warp puts it, and reads hot
    hx0, hy0, hx1, hy1 = meta["hot_output_box"]
    hot = rgb[hy0:hy1, hx0:hx1].reshape(-1, 3).mean(0)

    fx0, fy0, fx1, fy1 = meta["flat_box"]
    cold_x1 = meta["coverage_low"][0] * (ow // meta["low_w"]) + 60
    cold = rgb[oh // 2 - 20:oh // 2 + 20, cold_x1:cold_x1 + 60].reshape(-1, 3).mean(0)

    check("hot blob is bright on the palette", hot[0] > 180,
          "R=%.0f G=%.0f B=%.0f" % tuple(hot))
    check("hot reads hotter than cold", hot[0] > cold[0] + 60,
          "hotR=%.0f coldR=%.0f" % (hot[0], cold[0]))
    check("cold region stays on the low end", cold[0] < 150,
          "R=%.0f G=%.0f B=%.0f" % tuple(cold))

    # the two-column transition itself must stay bounded, not wild
    edge = rgb[:, ux1:ux1 + 2, 0].astype(int)
    check("coverage edge blends rather than jumps",
          np.abs(edge - y[:, ux1:ux1 + 2].astype(int)).max() < 200)

    # 4. injected detail must correlate with the source luma's structure
    cov = slice(60, oh - 60), slice(ux1 + 60, ow - 60)
    lum = rgb[..., :3].mean(2)
    hp_out, hp_y = highpass(lum[cov]), highpass(y[cov].astype(np.float64))
    r = np.corrcoef(hp_out.ravel(), hp_y.ravel())[0, 1]
    check("detail correlates with source luma", r > 0.7, "r=%.3f" % r)

    # 4b. Correlation is scale-invariant, so it says nothing about whether the
    #     detail is actually *visible*. Measured across palettes on a real frame,
    #     washed-out and vivid renderings scored identically (0.888-0.911) while
    #     their luma contrast differed fourfold. Amplitude is the property that
    #     matters, so check it separately and check that it tracks the gain.
    amp = hp_out.std()
    check("injected detail has real amplitude", amp > 2.0, "std=%.2f" % amp)

    subprocess.run(["./fuse", "synth/y.raw", "synth/thermal.raw", "synth/out_g2.ppm",
                    "--warp", "synth/warp.lut", "--gain", "400"],
                   check=True, stdout=subprocess.DEVNULL)
    amp2 = highpass(read_ppm("synth/out_g2.ppm")[..., :3].mean(2)[cov]).std()
    check("detail amplitude scales with gain", amp2 > amp * 1.4,
          "gain 200 -> %.2f, gain 400 -> %.2f" % (amp, amp2))

    # 5. and that correlation must be the gain, not something leaking through
    subprocess.run(["./fuse", "synth/y.raw", "synth/thermal.raw", "synth/out_g0.ppm",
                    "--warp", "synth/warp.lut", "--gain", "0"],
                   check=True, stdout=subprocess.DEVNULL)
    lum0 = read_ppm("synth/out_g0.ppm")[..., :3].mean(2)
    r0 = np.corrcoef(highpass(lum0[cov]).ravel(), hp_y.ravel())[0, 1]
    check("gain 0 drops the correlation", abs(r0) < r - 0.3, "r=%.3f vs %.3f" % (r0, r))

    # 6. Scene AGC. The synthetic thermal already spans the full range, so it is
    #    squeezed into a narrow band first - which is what an indoor scene
    #    actually looks like coming off the Lepton, and the case the AGC exists
    #    for. Measured on a real frame this was worth ~2.5x in luma contrast,
    #    more than the palette choice.
    th = np.fromfile("synth/thermal.raw", np.uint8)
    flat = (110 + (th.astype(np.int32) - th.mean()) * 0.10).clip(0, 255).astype(np.uint8)
    flat.tofile("synth/thermal_flat.raw")

    def render(name, *extra):
        subprocess.run(["./fuse", "synth/y.raw", "synth/thermal_flat.raw", name,
                        "--warp", "synth/warp.lut", *extra],
                       check=True, stdout=subprocess.DEVNULL)
        return read_ppm(name)[..., :3].mean(2)[cov]

    # Measured with gain 0. The AGC acts on the thermal tone, and at normal gain
    # the injected detail dominates the variance and hides the effect entirely -
    # the first version of this check read 16.5 -> 17.2 and looked like a
    # regression when nothing was wrong with the AGC at all.
    off = render("synth/agc_off.ppm", "--agc", "0", "--gain", "0")
    on = render("synth/agc_on.ppm", "--agc", "20", "--gain", "0")
    check("AGC lifts contrast on a low-contrast scene", on.std() > off.std() * 1.5,
          "tone std %.1f -> %.1f" % (off.std(), on.std()))

    off_d = render("synth/agc_off_d.ppm", "--agc", "0")
    on_d = render("synth/agc_on_d.ppm", "--agc", "20")
    check("AGC does not crush the detail layer",
          highpass(on_d).std() > highpass(off_d).std() * 0.8,
          "detail std %.2f -> %.2f" % (highpass(off_d).std(), highpass(on_d).std()))

    # 7. Monochrome ramps must keep headroom at both ends, or texture on the
    #    hottest and coldest targets is clipped away rather than modulated.
    # Headroom is a property of the ramp, so it is checked at gain 0. With detail
    # injected the output is *meant* to reach the ends - that is what the
    # headroom is reserved for.
    w0 = render("synth/white0.ppm", "--palette", "white", "--agc", "20", "--gain", "0")
    b0 = render("synth/black0.ppm", "--palette", "black", "--agc", "20", "--gain", "0")
    check("white-hot ramp keeps headroom at both ends", w0.min() > 1 and w0.max() < 254,
          "%.0f..%.0f" % (w0.min(), w0.max()))
    check("black-hot ramp is the inverse of white-hot",
          abs((w0.mean() + b0.mean()) - 255) < 25,
          "means %.0f / %.0f" % (w0.mean(), b0.mean()))

    w = render("synth/white.ppm", "--palette", "white", "--agc", "20")
    b = render("synth/black.ppm", "--palette", "black", "--agc", "20")
    check("black-hot keeps its detail amplitude", highpass(b).std() > highpass(w).std() * 0.7,
          "detail std %.2f vs %.2f" % (highpass(b).std(), highpass(w).std()))

    # 8. the sensor defect, and the measurement path that has to survive it
    check_row_repair(meta)
    check_radiometry(meta)

    # 9. the temporal noise filter
    check_temporal()

    print()
    if FAILS:
        print("FAILED: %s" % ", ".join(FAILS))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

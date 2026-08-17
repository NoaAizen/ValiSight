#!/usr/bin/env python3
"""Turn a bridge recording into the static "holds" Yael's IMU-camera solver takes.

    imu_holds.py out/imu.csv [--min-seconds 2] [--json holds.json]

Reads the imu.csv that bridge_rx writes (unwrapped ticks, mg, mdps), finds the
stretches where the rig was held still, averages each, and emits one entry per
hold in the shape of mapinit.calibration.imu.RigOrientation:

    {"label": "hold_01", "imu_accel_mg": [ax, ay, az], "camera_gravity": null,
     "t_start_s": ..., "t_end_s": ..., "n_samples": ..., "accel_sd_mg": ..., "gyro_max_dps": ...}

`camera_gravity` is deliberately left null: it must come from the CAMERA seeing
surveyed targets in that same hold, independent of the IMU — that is Yael's
side (and the whole point of her method). This tool only prepares the IMU half
and tells her which time window of the recording each hold spans, so she can
pull the matching thermal frames (<seq>_thermal.bin with the same timestamps).

Static test = the same criteria her solver and src/attitude.py use:
|a| within 60 mg of 1000 mg, and gyro below 3 deg/s, sustained.
Also reports whether the holds differ enough in tilt (>= 15 deg pairwise) to
determine the rotation — so a bad recording is caught here, not after the drive.
"""
import argparse, csv, json, math, sys

GRAVITY_MG = 1000.0
GRAVITY_TOL_MG = 60.0          # == mapinit GRAVITY_TOLERANCE_MG, attitude.STATIONARY_ACCEL_TOL_MG
STATIONARY_GYRO_DPS = 3.0      # == attitude.STATIONARY_GYRO_DPS
MIN_TILT_SEP_DEG = 15.0        # == mapinit MIN_TILT_SEPARATION_DEG


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            src = int(r["ts_src"])
            spt = 500e-9 if src == 1 else 1e-6
            rows.append((int(r["ts_ticks"]) * spt,
                         (float(r["ax_mg"]), float(r["ay_mg"]), float(r["az_mg"])),
                         (float(r["gx_mdps"]) / 1000.0, float(r["gy_mdps"]) / 1000.0, float(r["gz_mdps"]) / 1000.0)))
    return rows


def is_static(a, g):
    mag = math.sqrt(sum(c * c for c in a))
    return abs(mag - GRAVITY_MG) <= GRAVITY_TOL_MG and max(abs(c) for c in g) <= STATIONARY_GYRO_DPS


def find_holds(rows, min_seconds):
    holds, start = [], None
    for i, (t, a, g) in enumerate(rows):
        st = is_static(a, g)
        if st and start is None:
            start = i
        elif not st and start is not None:
            holds.append((start, i)); start = None
    if start is not None:
        holds.append((start, len(rows)))
    return [(s, e) for s, e in holds if rows[e - 1][0] - rows[s][0] >= min_seconds and e - s >= 5]


def summarize(rows, s, e, k):
    seg = rows[s:e]
    n = len(seg)
    mean = [sum(a[i] for _, a, _ in seg) / n for i in range(3)]
    sd = math.sqrt(sum(sum((a[i] - mean[i]) ** 2 for i in range(3)) for _, a, _ in seg) / n)
    gmax = max(max(abs(c) for c in g) for _, _, g in seg)
    return {"label": "hold_%02d" % k, "imu_accel_mg": [round(m, 1) for m in mean], "camera_gravity": None,
            "t_start_s": round(seg[0][0], 3), "t_end_s": round(seg[-1][0], 3), "n_samples": n,
            "accel_sd_mg": round(sd, 2), "gyro_max_dps": round(gmax, 3)}


def tilt_deg(a, b):
    na, nb = math.sqrt(sum(c * c for c in a)), math.sqrt(sum(c * c for c in b))
    d = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / (na * nb)))
    return math.degrees(math.acos(d))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("imu_csv"); ap.add_argument("--min-seconds", type=float, default=2.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rows = load(a.imu_csv)
    if not rows:
        sys.exit("no IMU rows in %s" % a.imu_csv)
    span = rows[-1][0] - rows[0][0]
    rate = (len(rows) - 1) / span if span > 0 else 0
    print("%d samples over %.1f s (%.0f Hz)" % (len(rows), span, rate))
    holds = [summarize(rows, s, e, k + 1) for k, (s, e) in enumerate(find_holds(rows, a.min_seconds))]
    for h in holds:
        print("  %s  %6.1f–%6.1f s  n=%-4d  a=(%7.1f %7.1f %7.1f) mg  sd=%.1f  gyro_max=%.2f dps"
              % (h["label"], h["t_start_s"], h["t_end_s"], h["n_samples"], *h["imu_accel_mg"], h["accel_sd_mg"], h["gyro_max_dps"]))
    # observability check, same rule as her solver
    if len(holds) >= 2:
        seps = [(hi["label"], hj["label"], tilt_deg(hi["imu_accel_mg"], hj["imu_accel_mg"]))
                for i, hi in enumerate(holds) for hj in holds[i + 1:]]
        best = max(s for _, _, s in seps)
        distinct = sum(1 for _, _, s in seps if s >= MIN_TILT_SEP_DEG)
        print("tilt separations: max %.1f deg, %d/%d pairs >= %.0f deg -> %s"
              % (best, distinct, len(seps), MIN_TILT_SEP_DEG,
                 "OK for the solver" if best >= MIN_TILT_SEP_DEG and len(holds) >= 3 else "NOT enough: tilt the rig more / add holds"))
    else:
        print("fewer than 2 holds — the rotation cannot be determined from this recording")
    if a.json:
        json.dump({"source": a.imu_csv, "holds": holds}, open(a.json, "w"), indent=1)
        print("wrote", a.json)


if __name__ == "__main__":
    main()

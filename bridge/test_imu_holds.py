#!/usr/bin/env python3
"""imu_holds: a synthetic imu.csv with 3 tilted holds separated by motion must
yield exactly 3 holds with the right mean vectors, and a recording with two
holds at the same tilt must be flagged as insufficient."""
import csv, json, math, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))

def write_csv(path, segments, hz=200):
    """segments: list of (seconds, accel_mg(3), gyro_dps(3))"""
    t = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow("seq,ts_ticks,ts_src,ax_mg,ay_mg,az_mg,gx_mdps,gy_mdps,gz_mdps".split(","))
        seq = 0
        for secs, a, g in segments:
            for i in range(int(secs * hz)):
                w.writerow([seq, int(t * 1e6), 0, *[int(x) for x in a], *[int(x * 1000) for x in g]])
                seq += 1; t += 1.0 / hz

def run(path, *args):
    p = subprocess.run([sys.executable, os.path.join(HERE, "imu_holds.py"), path, *args], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    return p.stdout

def test_three_holds():
    with tempfile.TemporaryDirectory() as d:
        c, j = os.path.join(d, "imu.csv"), os.path.join(d, "h.json")
        motion = (1.0, (300, 900, 600), (40, 10, -20))       # moving: gyro way above 3 dps
        write_csv(c, [(3, (0, 0, 1000), (0.1, 0, 0)), motion,
                      (3, (0, 643, 766), (0, 0.2, 0)), motion,        # 40 deg about x
                      (3, (643, 0, 766), (0, 0, 0.1))])               # 40 deg about y
        out = run(c, "--json", j)
        h = json.load(open(j))["holds"]
        assert len(h) == 3, out
        assert h[0]["imu_accel_mg"] == [0.0, 0.0, 1000.0]
        assert h[1]["imu_accel_mg"] == [0.0, 643.0, 766.0]
        assert all(x["camera_gravity"] is None for x in h)
        assert "OK for the solver" in out, out
        print("PASS test_three_holds")

def test_same_tilt_is_flagged():
    with tempfile.TemporaryDirectory() as d:
        c = os.path.join(d, "imu.csv")
        write_csv(c, [(3, (0, 0, 1000), (0, 0, 0)), (1.0, (0, 0, 1000), (30, 0, 0)), (3, (0, 5, 1000), (0, 0, 0))])
        out = run(c)
        assert "NOT enough" in out, out
        print("PASS test_same_tilt_is_flagged")

if __name__ == "__main__":
    test_three_holds(); test_same_tilt_is_flagged()

#!/usr/bin/env python3
"""Render one capture through several palette/gain combinations and tile them."""
import subprocess, sys, os
from PIL import Image, ImageDraw

# Script-relative: the old hardcoded ROOT pointed at a project location two
# moves ago and the output at a scratch directory that no longer exists.
ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", ".."))
FUSE = f"{ROOT}/host/fuse"
OUT = f"{ROOT}/captures/views"          # preview renders, gitignored
os.makedirs(OUT, exist_ok=True)

scene = sys.argv[1] if len(sys.argv) > 1 else "person2"
frame = sys.argv[2] if len(sys.argv) > 2 else "0000"
cap = f"{ROOT}/captures/{scene}"
y = f"{cap}/{frame}_rgb0.raw"
t = f"{cap}/{frame}_thermal.raw"

VARIANTS = [
    ("ironbow  gain 200  no agc (current)", ["--palette", "ironbow", "--gain", "200"]),
    ("ironbow  gain 200  agc 20",  ["--palette", "ironbow", "--gain", "200", "--agc", "20"]),
    ("white-hot  gain 320  no agc", ["--palette", "white", "--gain", "320"]),
    ("white-hot  gain 320  agc 20", ["--palette", "white", "--gain", "320", "--agc", "20"]),
    ("white-hot  gain 420  agc 20", ["--palette", "white", "--gain", "420", "--agc", "20"]),
    ("black-hot  gain 320  agc 20", ["--palette", "black", "--gain", "320", "--agc", "20"]),
]

tiles = []
for label, opts in VARIANTS:
    ppm = f"{OUT}/{scene}_{len(tiles)}.ppm"
    subprocess.run([FUSE, y, t, ppm] + opts, check=True, stdout=subprocess.DEVNULL)
    tiles.append((label, Image.open(ppm).convert("RGB")))

w, h = tiles[0][1].size
cols, rows = 2, (len(tiles) + 1) // 2
sheet = Image.new("RGB", (cols * w, rows * h), "black")
d = ImageDraw.Draw(sheet)
for i, (label, im) in enumerate(tiles):
    x, yy = (i % cols) * w, (i // cols) * h
    sheet.paste(im, (x, yy))
    d.rectangle([x + 4, yy + 4, x + 8 + 7 * len(label), yy + 20], fill="black")
    d.text((x + 8, yy + 8), label, fill="white")

path = f"{ROOT}/captures/fused/{scene}_mono.png"
sheet.save(path)
print(path, sheet.size)

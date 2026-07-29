"""Put src/ on the path. Imported first by every test module.

Tests live at the project root rather than under src/ so that `discover` never
imports the adapters, and so src/ keeps holding exactly the modules that ship.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
RECORDINGS = os.path.join(ROOT, "data", "recordings")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

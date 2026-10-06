"""Resource loading and legacy imports must survive running outside the repo."""
import ast
import os
from pathlib import Path
import subprocess
import sys

TOOLS = Path(__file__).resolve().parents[1]


def test_viewer_and_pages_load_from_an_unrelated_working_directory(tmp_path):
    code = """
import ast
import capture
import live
import label_web
import radar_calib_web
from viewer import native, rendering, streaming, telemetry

assert live.Pipeline is native.Pipeline
assert live.Renderer is rendering.Renderer
assert live.Streamer is streaming.Streamer
assert live.health is telemetry.health
assert live.PAGE.startswith(b'<!doctype html>')
assert label_web.PAGE.startswith('<!doctype html>')
assert radar_calib_web.PAGE.startswith('<!doctype html>')
for source in (capture.RECORD_CODE, capture.RAM_CODE, capture.FETCH_CODE,
               capture.STREAM_CODE, live.SETUP_CODE):
    ast.parse(source)
assert live.SETUP_CODE.startswith(capture._BRINGUP)
assert '#F %d %d %d %d %d' in live.SETUP_CODE
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        (str(TOOLS), str(TOOLS / "calib"))))
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                   check=True, capture_output=True, text=True)


def test_board_templates_are_valid_python_source():
    for path in (TOOLS / "board" / "templates").glob("*.py.tmpl"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

# Viewer components

`../live.py` owns CLI startup, HTTP routing and map integration. It re-exports
the original component names so existing callers keep working.

| Module | Responsibility |
| --- | --- |
| `constants.py` | Frame dimensions, shared library path and measured supervision thresholds |
| `native.py` | ctypes layouts, ABI checks and synchronized C pipeline access |
| `display.py` | Registration views and display enhancement |
| `rendering.py` | Latest-frame handoff, scene lighting, overlays and rendering worker |
| `streaming.py` | Board payload setup, framing, serial reads and restart supervision |
| `telemetry.py` | Health checks and board/host timing summaries |

Components do not import `live`. Keep the serial reader independent of the
renderer: the latest-frame handoff must never block draining the board's USB
stream. Display enhancement must not feed sensor measurements or model inputs.
The native structures must follow `src/fusion.h` exactly.

Run `make test-scripts` from the repository root to exercise the pipeline,
recorded frames, stream recovery, rendering and HTTP contracts without hardware.

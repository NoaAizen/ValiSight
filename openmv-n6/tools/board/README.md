# Board-side code

The `.py` probes in this directory are standalone MicroPython programs for the
OpenMV board. Their deployment commands remain in the host tools documentation.

`templates/*.py.tmpl` contains code transmitted by host scripts:

| Template | Host owner |
| --- | --- |
| `bringup`, `record`, `ram`, `fetch`, `capture_stream` | `../capture.py` |
| `live_stream` | `../viewer/streaming.py`, launched through `../live.py` |
| `radar_stage1` | `../diag/radar_stage1_n6.py` |
| `radar_sync_probe` | `../diag/radar_sync_probe.py` |

The host loads templates relative to its source location, then applies the
existing `__PLACEHOLDER__` substitutions before transmission. These files are
source text, not CPython modules. Keep them with the host scripts when deploying.
Changes to framing or bring-up need hardware validation in addition to the
offline syntax, payload and stream recovery tests.

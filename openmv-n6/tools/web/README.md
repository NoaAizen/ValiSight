# Browser pages

| Asset | Server |
| --- | --- |
| `live.html` | `../live.py` |
| `label.html` | `../label_web.py` |
| `radar_calibration.html` | `../calib/radar_calib_web.py` |
| `radar_doppler.html` | `../diag/radar_doppler_web.py` |

The pages contain their CSS and JavaScript and are still served by the existing
HTTP handlers. `assets.py` resolves resources relative to this directory.
`live.PAGE` remains bytes; labeling/calibration `PAGE` values remain text.
The label page uses Python `%` substitution, so literal percent signs there
must remain escaped as `%%`.

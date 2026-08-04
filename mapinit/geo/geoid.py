#!/usr/bin/env python3
"""Geoid undulation lookup backed by pyproj/PROJ, with no silent fallbacks.

The geoid undulation N relates the two height reference surfaces:

    h = H + N

where h is the ellipsoidal height and H is the orthometric height.

PROJ resolves EPSG:4979 -> EPSG:3855 using a vertical shift grid. If that grid
is absent PROJ does not fail: it quietly degrades to a "ballpark" transformation
that returns the input height unchanged. Every entry point here treats that
degraded state as an error, so a missing grid can never be mistaken for a
successful lookup.

Note that PROJ only reads GeoTIFF (.tif) and legacy .gtx grids. The
GeographicLib .pgm distribution of EGM2008 is *not* usable by PROJ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pyproj

# Enforce strict offline execution mode for pyproj to avoid network lookups
pyproj.network.set_network_enabled(False)

#: Grid PROJ needs for the EPSG:4979 -> EPSG:3855 (EGM2008) vertical transform.
EGM2008_GRID_NAME = "us_nga_egm08_25.tif"
EGM2008_GRID_URL = f"https://cdn.proj.org/{EGM2008_GRID_NAME}"


class GeoidGridUnavailable(RuntimeError):
    """Raised when PROJ cannot perform a real (non-ballpark) vertical transform."""


class GeoidModel:
    """Look up EGM2008 geoid undulation for a geographic point."""

    def __init__(self, extra_data_dir: Optional[Path | str] = None) -> None:
        # Let a project-local directory supply the grid alongside the PROJ user dir
        if extra_data_dir is not None:
            extra_data_dir = Path(extra_data_dir).expanduser().resolve()
            if extra_data_dir.is_dir():
                pyproj.datadir.append_data_dir(str(extra_data_dir))

        self._transformer = pyproj.Transformer.from_crs(
            "EPSG:4979", "EPSG:3855", always_xy=True
        )
        self._assert_grid_available()

    def _assert_grid_available(self) -> None:
        """Reject a transformer that PROJ downgraded to a ballpark approximation.

        PROJ reports ballpark operations with a negative accuracy and flags them
        in the pipeline description; either signal means no grid was loaded.
        """
        accuracy = self._transformer.accuracy
        description = self._transformer.description or ""

        if accuracy is not None and accuracy >= 0 and "ballpark" not in description.lower():
            return

        search_dirs = "\n  ".join(
            filter(None, (pyproj.datadir.get_user_data_dir(), pyproj.datadir.get_data_dir()))
        )
        raise GeoidGridUnavailable(
            f"PROJ has no EGM2008 vertical shift grid, so EPSG:4979 -> EPSG:3855 "
            f"degrades to a ballpark transform that would return the input height "
            f"unchanged (pipeline: {description!r}).\n"
            f"Install {EGM2008_GRID_NAME} into one of:\n  {search_dirs}\n"
            f"e.g. curl -o \"$(python -c 'import pyproj;print(pyproj.datadir.get_user_data_dir())')/"
            f"{EGM2008_GRID_NAME}\" {EGM2008_GRID_URL}\n"
            f"Note: GeographicLib .pgm grids are not readable by PROJ."
        )

    @property
    def description(self) -> str:
        """PROJ pipeline description, naming the operation that supplies the shift."""
        return self._transformer.description or "unknown"

    def undulation(self, latitude: float, longitude: float) -> float:
        """Return the geoid undulation N in meters (positive = geoid above ellipsoid).

        Transforming an ellipsoidal height of 0 yields the orthometric height
        H = 0 - N, so the undulation is the negated result.
        """
        *_, orthometric_at_zero = self._transformer.transform(longitude, latitude, 0.0)
        undulation = -float(orthometric_at_zero)

        # A grid that covers the globe but not this point returns infinity
        if undulation != undulation or abs(undulation) == float("inf"):
            raise GeoidGridUnavailable(
                f"Geoid grid does not cover lat={latitude}, lon={longitude}."
            )
        return undulation

    def orthometric_height(self, latitude: float, longitude: float, ellipsoidal_height_m: float) -> float:
        """Convert an ellipsoidal height h to an orthometric height H = h - N."""
        return ellipsoidal_height_m - self.undulation(latitude, longitude)

#!/usr/bin/env python3
"""Sample terrain height from a cached GLO-30 tile.

This turns the DEM from a file on disk into the two things initialization
actually needs from it: a ground plane, and a prior on our own altitude.

Three properties of GLO-30 shape every method here.

**It is a DSM, not a DTM.** The recorded height is the top of whatever is
there — roofs, canopy — not bare earth. Sampling directly over a building
returns its roof, so ``ground_elevation`` estimates ground from a low
percentile of a ring around the point rather than from the point itself. Adding
a building height to a value that already includes the building double-counts
it, which is why ``surface_elevation_m`` and ``ground_elevation`` are separate
methods with names that say which one you are holding.

**Its posting is 30 m.** One pixel spans thirty meters, so nothing smaller than
that exists in the data. It supports a ground plane and an ego-altitude prior;
it cannot support obstacle-level residuals, and ``assert_scale_supported``
exists to make that refusal explicit rather than leaving it to a comment.

**Its heights are orthometric**, referenced to EGM2008. They share a datum with
surveyed targets, and differ from a GNSS ellipsoidal height by the geoid
undulation at that point — about 19.8 m in Jerusalem. Mixing the two silently
biases altitude by that amount.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

#: Nominal grid spacing of GLO-30, in meters.
GLO30_POSTING_M = 30.0

#: Copernicus DEM specified absolute vertical accuracy, LE90, in meters.
GLO30_VERTICAL_ACCURACY_M = 4.0

#: Value Copernicus writes where it has no measurement.
GLO30_NODATA = -32767.0


class DemUnavailable(RuntimeError):
    """Raised when the DEM cannot answer for the requested point."""


class ScaleNotSupported(ValueError):
    """Raised when a caller asks the DEM for detail finer than its posting."""


@dataclass(frozen=True)
class GroundEstimate:
    """Ground height beneath a point, and the evidence behind it."""

    #: Estimated bare-ground orthometric height, in meters.
    ground_m: float
    #: Raw DSM height at the point, including any structure standing there.
    surface_m: float
    #: Spread of the ring samples the estimate was drawn from.
    sigma_m: float
    #: How far the surface stands above the estimated ground.
    structure_m: float
    ring_radius_m: float
    ring_samples: int

    def __str__(self) -> str:
        return (
            f"ground {self.ground_m:.1f} m, surface {self.surface_m:.1f} m "
            f"(+{self.structure_m:.1f} m structure), sigma {self.sigma_m:.2f} m "
            f"from {self.ring_samples} ring samples"
        )


@dataclass(frozen=True)
class EgoAltitudePrior:
    """Prior on our own altitude, in both height systems.

    Carrying both is deliberate: consumers comparing against GNSS need the
    ellipsoidal value, consumers comparing against the map need the orthometric
    one, and a single unlabelled number invites the two to be confused.
    """

    orthometric_m: float
    ellipsoidal_m: float
    geoid_undulation_m: float
    sigma_m: float
    ground: GroundEstimate

    def __str__(self) -> str:
        return (
            f"H = {self.orthometric_m:.1f} m orthometric, "
            f"h = {self.ellipsoidal_m:.1f} m ellipsoidal "
            f"(N = {self.geoid_undulation_m:+.2f} m), sigma {self.sigma_m:.1f} m"
        )


class DemSampler:
    """Reads heights out of one cached DEM tile."""

    def __init__(
        self,
        dem_path: Path,
        nodata: Optional[float] = None,
        posting_m: float = GLO30_POSTING_M,
        vertical_accuracy_m: float = GLO30_VERTICAL_ACCURACY_M,
    ) -> None:
        try:
            import rasterio
        except ImportError as exc:
            raise DemUnavailable(
                "rasterio is required to sample the DEM. Install it with: pip install rasterio"
            ) from exc

        self.dem_path = Path(dem_path)
        if not self.dem_path.is_file():
            raise DemUnavailable(f"DEM file not found: {self.dem_path}")

        self._dataset = rasterio.open(self.dem_path)
        self.posting_m = posting_m
        self.vertical_accuracy_m = vertical_accuracy_m
        self.nodata = nodata if nodata is not None else (self._dataset.nodata or GLO30_NODATA)

    def close(self) -> None:
        self._dataset.close()

    def __enter__(self) -> "DemSampler":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @property
    def bounds(self):
        return self._dataset.bounds

    def covers(self, latitude: float, longitude: float) -> bool:
        b = self._dataset.bounds
        return b.left <= longitude <= b.right and b.bottom <= latitude <= b.top

    # -- scale ------------------------------------------------------------

    def assert_scale_supported(self, feature_size_m: float) -> None:
        """Refuse to be used below the resolution the data actually carries.

        A 1.2 m clustering epsilon is forty times finer than the posting. Asking
        this DEM for residuals at that scale returns interpolation artifacts
        that look like signal, so the refusal is explicit rather than implied.
        """
        if feature_size_m < self.posting_m:
            raise ScaleNotSupported(
                f"Requested feature size {feature_size_m:.2f} m is finer than the "
                f"{self.posting_m:.0f} m DEM posting. This DEM supports a ground "
                f"plane and an ego-altitude prior only; obstacle-level residuals "
                f"would be interpolation artifacts, not terrain."
            )

    # -- point sampling ---------------------------------------------------

    def _read_window(self, row: int, col: int, size: int = 2):
        """Read a small block, clamped to the raster so edges do not wrap."""
        import rasterio.windows

        row = max(0, min(row, self._dataset.height - size))
        col = max(0, min(col, self._dataset.width - size))
        window = rasterio.windows.Window(col, row, size, size)
        return self._dataset.read(1, window=window).astype(float)

    def surface_elevation_m(self, latitude: float, longitude: float) -> float:
        """Bilinearly interpolated DSM height at a point, in meters orthometric.

        This is the top of whatever stands there. For bare ground use
        ``ground_elevation``.
        """
        if not self.covers(latitude, longitude):
            raise DemUnavailable(
                f"Point lat={latitude}, lon={longitude} is outside {self.dem_path.name} "
                f"(bounds {self._dataset.bounds})."
            )

        # Fractional pixel position; the 0.5 shift puts the origin at pixel centres
        col_f, row_f = ~self._dataset.transform * (longitude, latitude)
        col_f, row_f = col_f - 0.5, row_f - 0.5
        col, row = math.floor(col_f), math.floor(row_f)
        dx, dy = col_f - col, row_f - row

        block = self._read_window(row, col)
        if block.shape != (2, 2):
            raise DemUnavailable(f"Could not read a 2x2 block at row={row}, col={col}")

        # A single nodata neighbour would poison the interpolation, so fall back
        # to the mean of whatever real samples the block does hold
        valid = block != self.nodata
        if not valid.any():
            raise DemUnavailable(
                f"DEM has no data at lat={latitude}, lon={longitude} "
                f"(all samples are nodata {self.nodata})."
            )
        if not valid.all():
            return float(block[valid].mean())

        top = block[0, 0] * (1 - dx) + block[0, 1] * dx
        bottom = block[1, 0] * (1 - dx) + block[1, 1] * dx
        return float(top * (1 - dy) + bottom * dy)

    def sample_many(self, latitudes, longitudes):
        """Sample many points at once, reading the covering window only once.

        Marching rays out to kilometres asks for tens of thousands of heights,
        and going through the file for each costs seconds. Pulling the covering
        block into memory and interpolating there is the same arithmetic two
        orders of magnitude faster.

        Returns a float array with NaN wherever the DEM has no data.
        """
        import numpy as np
        import rasterio.windows

        latitudes = np.asarray(latitudes, dtype=float)
        longitudes = np.asarray(longitudes, dtype=float)

        cols, rows = ~self._dataset.transform * (longitudes, latitudes)
        cols, rows = np.asarray(cols) - 0.5, np.asarray(rows) - 0.5

        col_min = int(math.floor(np.nanmin(cols))) - 1
        row_min = int(math.floor(np.nanmin(rows))) - 1
        col_max = int(math.ceil(np.nanmax(cols))) + 2
        row_max = int(math.ceil(np.nanmax(rows))) + 2

        col_min, row_min = max(col_min, 0), max(row_min, 0)
        col_max = min(col_max, self._dataset.width)
        row_max = min(row_max, self._dataset.height)
        if col_max <= col_min or row_max <= row_min:
            return np.full(latitudes.shape, np.nan)

        window = rasterio.windows.Window(
            col_min, row_min, col_max - col_min, row_max - row_min
        )
        block = self._dataset.read(1, window=window).astype(float)
        block[block == self.nodata] = np.nan

        local_col = cols - col_min
        local_row = rows - row_min
        c0 = np.clip(np.floor(local_col).astype(int), 0, block.shape[1] - 2)
        r0 = np.clip(np.floor(local_row).astype(int), 0, block.shape[0] - 2)
        dx = np.clip(local_col - c0, 0.0, 1.0)
        dy = np.clip(local_row - r0, 0.0, 1.0)

        top = block[r0, c0] * (1 - dx) + block[r0, c0 + 1] * dx
        bottom = block[r0 + 1, c0] * (1 - dx) + block[r0 + 1, c0 + 1] * dx
        heights = top * (1 - dy) + bottom * dy

        # Outside the raster the interpolation is meaningless, whatever the
        # clipped indices happened to land on
        outside = (
            (local_col < 0) | (local_col > block.shape[1] - 1)
            | (local_row < 0) | (local_row > block.shape[0] - 1)
        )
        heights[outside] = np.nan
        return heights

    def _ring_samples(self, latitude: float, longitude: float, radius_m: float, count: int) -> List[float]:
        """Heights sampled evenly around a circle, skipping points off the tile."""
        samples = []
        metres_per_deg_lat = 111_320.0
        metres_per_deg_lon = metres_per_deg_lat * math.cos(math.radians(latitude))

        for i in range(count):
            angle = 2 * math.pi * i / count
            lat = latitude + (radius_m * math.sin(angle)) / metres_per_deg_lat
            lon = longitude + (radius_m * math.cos(angle)) / max(metres_per_deg_lon, 1e-6)
            try:
                samples.append(self.surface_elevation_m(lat, lon))
            except DemUnavailable:
                continue
        return samples

    def ground_elevation(
        self,
        latitude: float,
        longitude: float,
        ring_radius_m: float = 60.0,
        percentile: float = 20.0,
        ring_samples: int = 16,
    ) -> GroundEstimate:
        """Estimate bare-ground height from a low percentile of a perimeter ring.

        GLO-30 records surfaces, so the height directly overhead may be a roof.
        A ring at least a couple of postings out is likely to touch open ground
        somewhere, and a low percentile of it picks that up while ignoring the
        structures the ring also crosses.

        Defaults put the ring at twice the posting and take the 20th percentile.
        Both are engineering choices, exposed as arguments so a deployment can
        state its own rather than inherit one buried here.
        """
        surface = self.surface_elevation_m(latitude, longitude)
        samples = self._ring_samples(latitude, longitude, ring_radius_m, ring_samples)

        if not samples:
            raise DemUnavailable(
                f"No usable DEM samples on a {ring_radius_m:.0f} m ring around "
                f"lat={latitude}, lon={longitude}."
            )

        ordered = sorted(samples)
        index = min(int(len(ordered) * percentile / 100.0), len(ordered) - 1)
        ground = ordered[index]

        mean = sum(samples) / len(samples)
        sigma = math.sqrt(sum((s - mean) ** 2 for s in samples) / len(samples))

        return GroundEstimate(
            ground_m=ground,
            surface_m=surface,
            sigma_m=sigma,
            structure_m=surface - ground,
            ring_radius_m=ring_radius_m,
            ring_samples=len(samples),
        )

    # -- priors -----------------------------------------------------------

    def ego_altitude_prior(
        self,
        latitude: float,
        longitude: float,
        geoid_undulation_m: float,
        rig_height_agl_m: float = 0.0,
        **ground_kwargs,
    ) -> EgoAltitudePrior:
        """Prior on our own altitude, given how high the rig sits above ground.

        ``geoid_undulation_m`` must come from a verified geoid; passing zero
        because none was available reintroduces exactly the bias the geoid guard
        exists to prevent.
        """
        ground = self.ground_elevation(latitude, longitude, **ground_kwargs)
        orthometric = ground.ground_m + rig_height_agl_m

        # The DEM's own accuracy and the local terrain spread are independent
        # sources of error, so they combine in quadrature
        sigma = math.hypot(self.vertical_accuracy_m, ground.sigma_m)

        return EgoAltitudePrior(
            orthometric_m=orthometric,
            ellipsoidal_m=orthometric + geoid_undulation_m,
            geoid_undulation_m=geoid_undulation_m,
            sigma_m=sigma,
            ground=ground,
        )

    def ground_plane(
        self,
        latitude: float,
        longitude: float,
        radius_m: float = 90.0,
        samples: int = 16,
    ) -> Tuple[float, float, float]:
        """Fit a local ground plane, returning (slope %, aspect deg, rms residual m).

        Slope and aspect describe the terrain the rig stands on; the residual
        says how well a plane describes it at all. A large residual means the
        ground is not locally planar and a plane prior will mislead.
        """
        ring = []
        metres_per_deg_lat = 111_320.0
        metres_per_deg_lon = metres_per_deg_lat * math.cos(math.radians(latitude))

        for i in range(samples):
            angle = 2 * math.pi * i / samples
            east = radius_m * math.cos(angle)
            north = radius_m * math.sin(angle)
            lat = latitude + north / metres_per_deg_lat
            lon = longitude + east / max(metres_per_deg_lon, 1e-6)
            try:
                ring.append((east, north, self.surface_elevation_m(lat, lon)))
            except DemUnavailable:
                continue

        if len(ring) < 3:
            raise DemUnavailable(
                f"Need at least 3 usable samples to fit a plane, got {len(ring)}."
            )

        # Least squares for z = a*east + b*north + c, via the normal equations.
        # The ring is symmetric about the centre, so the system stays well
        # conditioned without needing a general solver.
        n = len(ring)
        sum_e = sum(p[0] for p in ring)
        sum_n = sum(p[1] for p in ring)
        sum_z = sum(p[2] for p in ring)
        sum_ee = sum(p[0] * p[0] for p in ring)
        sum_nn = sum(p[1] * p[1] for p in ring)
        sum_en = sum(p[0] * p[1] for p in ring)
        sum_ez = sum(p[0] * p[2] for p in ring)
        sum_nz = sum(p[1] * p[2] for p in ring)

        # Centre the samples so the cross terms drop out and a, b decouple
        mean_e, mean_n, mean_z = sum_e / n, sum_n / n, sum_z / n
        cov_ee = sum_ee - n * mean_e * mean_e
        cov_nn = sum_nn - n * mean_n * mean_n
        cov_en = sum_en - n * mean_e * mean_n
        cov_ez = sum_ez - n * mean_e * mean_z
        cov_nz = sum_nz - n * mean_n * mean_z

        determinant = cov_ee * cov_nn - cov_en * cov_en
        if abs(determinant) < 1e-9:
            raise DemUnavailable("Sample geometry is degenerate; cannot fit a plane.")

        a = (cov_ez * cov_nn - cov_nz * cov_en) / determinant
        b = (cov_nz * cov_ee - cov_ez * cov_en) / determinant
        c = mean_z - a * mean_e - b * mean_n

        residual = math.sqrt(
            sum((z - (a * e + b * north + c)) ** 2 for e, north, z in ring) / n
        )
        slope_percent = math.hypot(a, b) * 100.0
        aspect_deg = math.degrees(math.atan2(-a, -b)) % 360.0

        return (slope_percent, aspect_deg, residual)

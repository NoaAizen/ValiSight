#!/usr/bin/env python3
"""Sources for the spatial prior layers used to initialize position.

The provider interface is the seam between "where the priors live" and "how
initialization uses them". Today they come off local disk; once spatial
database infrastructure exists, swapping the implementation must not change
which tile a given coordinate resolves to — which is why the local provider
performs a real point-in-tile lookup rather than taking whatever file it finds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .tiles import select_tile

#: Minimum plausible sizes; anything smaller is a truncated or failed download.
MIN_DEM_BYTES = 1024 * 50
MIN_VECTOR_BYTES = 1024 * 10


@dataclass
class PriorPaths:
    """Data Transfer Object (DTO) encapsulating paths to local prior datasets."""

    glo30_path: Path
    overture_path: Path


class BasePriorDataProvider(ABC):
    """Abstract Base Class defining the contract for spatial prior data providers."""

    @abstractmethod
    def get_priors(self, latitude: float, longitude: float) -> PriorPaths:
        """
        Retrieve file paths or data streams for DEM (GLO-30) and Vector (Overture) priors.

        Args:
            latitude (float): Target latitude in decimal degrees.
            longitude (float): Target longitude in decimal degrees.

        Returns:
            PriorPaths: Container with paths to the resolved spatial layers.
        """


class LocalFilePriorProvider(BasePriorDataProvider):
    """
    Temporary provider implementation: fetches prior files directly from local disk.

    Expected folder structure:
        priors_dir/
            ├── glo30/    -> Contains .tif/.tiff DEM grid files
            └── overture/ -> Contains .parquet/.geojson vector layers

    Files are selected by the tile tag in their name (the Copernicus
    ``Copernicus_DSM_COG_10_N31_00_E035_00_DEM.tif`` form, or a bare
    ``N31E035``), so the requested point picks the tile rather than being
    ignored. A directory holding a single untagged file is still accepted as an
    untiled global export.
    """

    DEM_SUFFIXES = {".tif", ".tiff"}
    VECTOR_SUFFIXES = {".parquet", ".geojson", ".json"}

    def __init__(self, priors_dir: Path, tile_size_deg: float = 1.0) -> None:
        self.priors_dir = Path(priors_dir)
        self.glo30_dir = self.priors_dir / "glo30"
        self.overture_dir = self.priors_dir / "overture"
        self.tile_size_deg = tile_size_deg

    def get_priors(self, latitude: float, longitude: float) -> PriorPaths:
        # 1. Validate subdirectory existence
        if not self.glo30_dir.exists() or not self.overture_dir.exists():
            raise FileNotFoundError(
                f"Priors directory structure missing under {self.priors_dir}. "
                f"Expected 'glo30/' and 'overture/' subdirectories."
            )

        # 2. Resolve the DEM and vector tiles that actually cover the point
        glo30_path = select_tile(
            (f for f in self.glo30_dir.iterdir() if f.suffix.lower() in self.DEM_SUFFIXES),
            latitude, longitude, dataset="GLO-30 DEM", size_deg=self.tile_size_deg,
        )
        overture_path = select_tile(
            (f for f in self.overture_dir.iterdir() if f.suffix.lower() in self.VECTOR_SUFFIXES),
            latitude, longitude, dataset="Overture vector", size_deg=self.tile_size_deg,
        )

        # 3. Perform basic integrity checks (guarantee non-empty files)
        if glo30_path.stat().st_size < MIN_DEM_BYTES:
            raise RuntimeError(f"GLO-30 file {glo30_path.name} is corrupted or too small.")
        if overture_path.stat().st_size < MIN_VECTOR_BYTES:
            raise RuntimeError(f"Overture file {overture_path.name} is corrupted or too small.")

        return PriorPaths(glo30_path=glo30_path, overture_path=overture_path)


class DatabasePriorProvider(BasePriorDataProvider):
    """
    Future implementation: retrieves prior layers dynamically from a local/remote spatial
    database (e.g., PostGIS, SpatiaLite, or Blob storage).
    """

    def __init__(self, db_connection_string: str) -> None:
        self.db_connection_string = db_connection_string

    def get_priors(self, latitude: float, longitude: float) -> PriorPaths:
        # TODO: Implement database connection, spatial query (bounding box/point-in-polygon),
        # download/extract cached tiles to temporary path, and return PriorPaths.
        raise NotImplementedError("Database provider will be implemented once DB infrastructure is live.")

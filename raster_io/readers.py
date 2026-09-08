"""Reads GeoTIFF/TIFF (georeferenced) and PNG/JPEG (benchmark) images into
a common RasterImage, always preferring metadata over assumption.

GSD, CRS, and the affine transform are read from the file's own metadata
whenever GDAL can find it (GeoTIFF/TIFF with georeferencing tags). PNG/JPEG
-- the formats VRSBench/RSVQA-LR/CDVQA ship benchmark crops in -- carry no
georeferencing at all, so their GSD must be supplied by the caller (who
knows it from the benchmark's own documentation); read_image() never
invents one. The transform is kept on RasterImage (not discarded after
use) specifically so any pixel can later be converted to lon/lat via
pixel_to_lonlat().
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Geod
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning
from rasterio.transform import Affine
from rasterio.warp import transform as warp_transform

SUPPORTED_EXTENSIONS = frozenset({".tif", ".tiff", ".png", ".jpg", ".jpeg"})

_GEOD = Geod(ellps="WGS84")


class ReaderError(ValueError):
    """Raised when a file can't be read, or isn't a supported format."""


@dataclass
class RasterImage:
    """One loaded image. `data` is (bands, height, width), matching
    rasterio's convention. `transform` and `crs` are None for a format or
    file with no georeferencing (plain PNG/JPEG, or a non-geo TIFF);
    `gsd_metres` is None whenever it truly can't be resolved -- callers
    must not treat None as "assume 10 m", they must refuse (see
    raster_io/validate.py)."""

    data: np.ndarray
    transform: Affine | None
    crs: str | None
    gsd_metres: float | None
    band_count: int
    width: int
    height: int
    path: Path


def _compute_gsd_metres(transform: Affine, crs: CRS) -> float:
    """The real ground distance covered by one pixel step, computed
    geodesically so it's correct for both projected (already-metric) and
    geographic (degrees) CRSs alike -- never assumed from the CRS's
    reported units."""
    x0, y0 = transform @ (0, 0)
    x1, y1 = transform @ (1, 0)
    x2, y2 = transform @ (0, 1)
    lons, lats = warp_transform(crs, "EPSG:4326", [x0, x1, x2], [y0, y1, y2])
    _, _, dist_x = _GEOD.inv(lons[0], lats[0], lons[1], lats[1])
    _, _, dist_y = _GEOD.inv(lons[0], lats[0], lons[2], lats[2])
    return (dist_x + dist_y) / 2.0


def read_image(path: str | Path, *, gsd_metres_override: float | None = None) -> RasterImage:
    """Read one image.

    `gsd_metres_override` is used only for a file with no embedded
    georeferencing (PNG/JPEG, or a plain non-geo TIFF) -- pass it when you
    already know the GSD from the source dataset's own documentation
    (e.g. a benchmark's stated ground resolution). It is ignored, not
    merged or averaged, whenever the file has real georeferencing: the
    file's own metadata always wins.
    """
    path = Path(path)
    if not path.exists():
        raise ReaderError(f"file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ReaderError(
            f"unsupported format {path.suffix!r}; supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as src:
            data = src.read()
            crs = src.crs
            transform = (
                src.transform
                if src.transform is not None and not src.transform.is_identity
                else None
            )

    band_count, height, width = data.shape
    crs_str = crs.to_string() if crs else None

    gsd_metres = None
    if crs is not None and transform is not None:
        gsd_metres = _compute_gsd_metres(transform, crs)
    elif gsd_metres_override is not None:
        gsd_metres = gsd_metres_override

    return RasterImage(
        data=data,
        transform=transform,
        crs=crs_str,
        gsd_metres=gsd_metres,
        band_count=band_count,
        width=width,
        height=height,
        path=path,
    )


def pixel_to_lonlat(image: RasterImage, row: int, col: int) -> tuple[float, float]:
    """Convert a pixel (row, col) to (lon, lat) using the image's own
    preserved transform and CRS. Raises ReaderError if the image has no
    georeferencing to convert from."""
    if image.transform is None or image.crs is None:
        raise ReaderError(
            f"{image.path.name}: no georeferencing (transform/CRS) -- cannot convert pixel to lon/lat"
        )
    x, y = image.transform @ (col + 0.5, row + 0.5)  # pixel center
    lons, lats = warp_transform(image.crs, "EPSG:4326", [x], [y])
    return lons[0], lats[0]
